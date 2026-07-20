#pragma once
#include "../common.cuh"
#include "../carrier.cuh"

template <typename scalar_t>
__global__ void tria_init_gate_forward_kernel(
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ w, scalar_t* __restrict__ carry_1,
    scalar_t* __restrict__ p_out, float* __restrict__ scale_out,
    float alpha, int axis, int64_t n) {
    const int64_t idx=blockIdx.x*(int64_t)blockDim.x+threadIdx.x;
    if(idx>=n) return;
    float vals[9];
    tria_carrier_build9((float)r[idx],(float)i_[idx],(float)o[idx],alpha,axis,vals);
    const float scale=tria_rms9(vals), inv=1.0f/scale;
    scalar_t* out=carry_1+idx*9;
    float p=0.0f;
    #pragma unroll
    for(int k=0;k<9;++k){ const float cv=vals[k]*inv; out[k]=(scalar_t)cv; p=fmaf(cv,(float)w[k],p); }
    p_out[idx]=(scalar_t)p; scale_out[idx]=scale;
}

template <typename scalar_t>
__global__ void tria_init_gate_backward_kernel(
    const scalar_t* __restrict__ grad_carry_1, const scalar_t* __restrict__ grad_p_out,
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ w, const float* __restrict__ scale,
    scalar_t* __restrict__ grad_r, scalar_t* __restrict__ grad_i, scalar_t* __restrict__ grad_o,
    float* __restrict__ grad_w_partial, float alpha, int axis, int64_t n) {
    const int64_t idx=blockIdx.x*(int64_t)blockDim.x+threadIdx.x;
    float local_w[9]={0.0f};
    if(idx<n){
        const float rv=(float)r[idx],iv=(float)i_[idx],ov=(float)o[idx];
        float vals[9], gout[9], gm[9];
        tria_carrier_build9(rv,iv,ov,alpha,axis,vals);
        const float gp=(float)grad_p_out[idx], inv=1.0f/scale[idx];
        const scalar_t* g=grad_carry_1+idx*9;
        #pragma unroll
        for(int k=0;k<9;++k){ gout[k]=(float)g[k]+gp*(float)w[k]; local_w[k]=gp*vals[k]*inv; }
        tria_rms_backward9(gout,vals,scale[idx],gm);
        float da,db,dc; tria_carrier_grad_abc(gm,alpha,axis,rv,iv,ov,da,db,dc);
        grad_r[idx]=(scalar_t)(da*iv+db*ov);
        grad_i[idx]=(scalar_t)(da*rv+dc*ov);
        grad_o[idx]=(scalar_t)(db*rv+dc*iv);
    }
    gate_mix_block_reduce9(local_w);
    if(threadIdx.x<9) grad_w_partial[(int64_t)threadIdx.x*gridDim.x+blockIdx.x]=local_w[threadIdx.x];
}
