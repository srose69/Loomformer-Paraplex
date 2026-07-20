#pragma once
#include "../common.cuh"
#include "../carrier.cuh"

template <typename scalar_t>
__global__ void tria_step_gate_forward_kernel(
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ carry_prev, const scalar_t* __restrict__ w,
    scalar_t* __restrict__ carry_new, scalar_t* __restrict__ p_out,
    float* __restrict__ scale_out, float alpha, int axis, int64_t n) {
    const int64_t idx=blockIdx.x*(int64_t)blockDim.x+threadIdx.x;
    if(idx>=n) return;
    float m[9],cp[9],pre[9];
    tria_carrier_build9((float)r[idx],(float)i_[idx],(float)o[idx],alpha,axis,m);
    #pragma unroll
    for(int k=0;k<9;++k) cp[k]=(float)carry_prev[idx*9+k];
    tria_matmul9(m,cp,pre);
    const float scale=tria_rms9(pre),inv=1.0f/scale;
    scalar_t* out=carry_new+idx*9; float p=0.0f;
    #pragma unroll
    for(int k=0;k<9;++k){ const float cv=pre[k]*inv; out[k]=(scalar_t)cv; p=fmaf(cv,(float)w[k],p); }
    p_out[idx]=(scalar_t)p; scale_out[idx]=scale;
}

template <typename scalar_t>
__global__ void tria_step_gate_backward_kernel(
    const scalar_t* __restrict__ grad_carry_new, const scalar_t* __restrict__ grad_p_out,
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ carry_prev, const scalar_t* __restrict__ w,
    const float* __restrict__ scale,
    scalar_t* __restrict__ grad_r, scalar_t* __restrict__ grad_i, scalar_t* __restrict__ grad_o,
    scalar_t* __restrict__ grad_carry_prev, float* __restrict__ grad_w_partial,
    float alpha, int axis, int64_t n) {
    const int64_t idx=blockIdx.x*(int64_t)blockDim.x+threadIdx.x;
    float local_w[9]={0.0f};
    if(idx<n){
        const float rv=(float)r[idx],iv=(float)i_[idx],ov=(float)o[idx];
        float m[9],cp[9],pre[9],gout[9],gpre[9],gm[9],gcp[9];
        tria_carrier_build9(rv,iv,ov,alpha,axis,m);
        #pragma unroll
        for(int k=0;k<9;++k) cp[k]=(float)carry_prev[idx*9+k];
        tria_matmul9(m,cp,pre);
        const float gp=(float)grad_p_out[idx],inv=1.0f/scale[idx];
        #pragma unroll
        for(int k=0;k<9;++k){ gout[k]=(float)grad_carry_new[idx*9+k]+gp*(float)w[k]; local_w[k]=gp*pre[k]*inv; }
        tria_rms_backward9(gout,pre,scale[idx],gpre);
        tria_matmul_right_transpose9(gpre,cp,gm);
        tria_matmul_left_transpose9(m,gpre,gcp);
        float da,db,dc; tria_carrier_grad_abc(gm,alpha,axis,rv,iv,ov,da,db,dc);
        grad_r[idx]=(scalar_t)(da*iv+db*ov);
        grad_i[idx]=(scalar_t)(da*rv+dc*ov);
        grad_o[idx]=(scalar_t)(db*rv+dc*iv);
        #pragma unroll
        for(int k=0;k<9;++k) grad_carry_prev[idx*9+k]=(scalar_t)gcp[k];
    }
    gate_mix_block_reduce9(local_w);
    if(threadIdx.x<9) grad_w_partial[(int64_t)threadIdx.x*gridDim.x+blockIdx.x]=local_w[threadIdx.x];
}

// Gated counterpart of tria_step_reverse_backward_kernel -- see that
// function's comment for the O(1) analytic-reverse rationale.
template <typename scalar_t>
__global__ void tria_step_gate_reverse_backward_kernel(
    const scalar_t* __restrict__ grad_carry_new, const scalar_t* __restrict__ grad_p_out,
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ current, const scalar_t* __restrict__ w,
    scalar_t* __restrict__ grad_r, scalar_t* __restrict__ grad_i, scalar_t* __restrict__ grad_o,
    scalar_t* __restrict__ grad_previous, scalar_t* __restrict__ previous_out,
    float* __restrict__ grad_w_partial,
    float alpha, int axis, int64_t n) {
    const int64_t idx=blockIdx.x*(int64_t)blockDim.x+threadIdx.x;
    float local_w[9]={0.0f};
    if(idx<n){
        const float rv=(float)r[idx],iv=(float)i_[idx],ov=(float)o[idx];
        float m[9],cur[9],prev[9],pre[9],gout[9],gpre[9],gm[9],gprev[9];
        tria_carrier_build9(rv,iv,ov,alpha,axis,m);
        #pragma unroll
        for(int k=0;k<9;++k) cur[k]=(float)current[idx*9+k];
        tria_reverse_prev9(m,cur,prev);
        tria_matmul9(m,prev,pre);
        const float scale=tria_rms9(pre),inv=1.0f/scale;
        const float gp=(float)grad_p_out[idx];
        #pragma unroll
        for(int k=0;k<9;++k){ gout[k]=(float)grad_carry_new[idx*9+k]+gp*(float)w[k]; local_w[k]=gp*pre[k]*inv; }
        tria_rms_backward9(gout,pre,scale,gpre);
        tria_matmul_right_transpose9(gpre,prev,gm);
        tria_matmul_left_transpose9(m,gpre,gprev);
        float da,db,dc; tria_carrier_grad_abc(gm,alpha,axis,rv,iv,ov,da,db,dc);
        grad_r[idx]=(scalar_t)(da*iv+db*ov);
        grad_i[idx]=(scalar_t)(da*rv+dc*ov);
        grad_o[idx]=(scalar_t)(db*rv+dc*iv);
        #pragma unroll
        for(int k=0;k<9;++k){
            grad_previous[idx*9+k]=(scalar_t)gprev[k];
            previous_out[idx*9+k]=(scalar_t)prev[k];
        }
    }
    gate_mix_block_reduce9(local_w);
    if(threadIdx.x<9) grad_w_partial[(int64_t)threadIdx.x*gridDim.x+blockIdx.x]=local_w[threadIdx.x];
}
