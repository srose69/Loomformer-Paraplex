#pragma once
#include "../carrier.cuh"

template <typename scalar_t>
__global__ void tria_step_forward_kernel(
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ carry_prev, scalar_t* __restrict__ carry_new,
    float* __restrict__ scale_out, float alpha, int axis, int64_t n) {
    const int64_t idx=blockIdx.x*(int64_t)blockDim.x+threadIdx.x;
    if(idx>=n) return;
    float m[9],cp[9],pre[9];
    tria_carrier_build9((float)r[idx],(float)i_[idx],(float)o[idx],alpha,axis,m);
    #pragma unroll
    for(int k=0;k<9;++k) cp[k]=(float)carry_prev[idx*9+k];
    tria_matmul9(m,cp,pre);
    const float scale=tria_rms9(pre),inv=1.0f/scale;
    scalar_t* out=carry_new+idx*9;
    #pragma unroll
    for(int k=0;k<9;++k) out[k]=(scalar_t)(pre[k]*inv);
    scale_out[idx]=scale;
}

template <typename scalar_t>
__global__ void tria_step_backward_kernel(
    const scalar_t* __restrict__ grad_carry_new,
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ carry_prev, const float* __restrict__ scale,
    scalar_t* __restrict__ grad_r, scalar_t* __restrict__ grad_i, scalar_t* __restrict__ grad_o,
    scalar_t* __restrict__ grad_carry_prev, float alpha, int axis, int64_t n) {
    const int64_t idx=blockIdx.x*(int64_t)blockDim.x+threadIdx.x;
    if(idx>=n) return;
    const float rv=(float)r[idx],iv=(float)i_[idx],ov=(float)o[idx];
    float m[9],cp[9],pre[9],gout[9],gpre[9],gm[9],gcp[9];
    tria_carrier_build9(rv,iv,ov,alpha,axis,m);
    #pragma unroll
    for(int k=0;k<9;++k){ cp[k]=(float)carry_prev[idx*9+k]; gout[k]=(float)grad_carry_new[idx*9+k]; }
    tria_matmul9(m,cp,pre);
    tria_rms_backward9(gout,pre,scale[idx],gpre);
    tria_matmul_right_transpose9(gpre,cp,gm);  // dM = dpre @ cp^T
    tria_matmul_left_transpose9(m,gpre,gcp);   // dcp = M^T @ dpre
    float da,db,dc; tria_carrier_grad_abc(gm,alpha,axis,rv,iv,ov,da,db,dc);
    grad_r[idx]=(scalar_t)(da*iv+db*ov);
    grad_i[idx]=(scalar_t)(da*rv+dc*ov);
    grad_o[idx]=(scalar_t)(db*rv+dc*iv);
    #pragma unroll
    for(int k=0;k<9;++k) grad_carry_prev[idx*9+k]=(scalar_t)gcp[k];
}

// O(1) analytic-reverse backward: recovers carry_prev from `current`
// (=carry_new, the FP32 state this step produced) via tria_reverse_prev9
// instead of taking it as a saved input -- the FP32 counterpart of
// depth_replay_backward's half-precision forward-replay. Also emits
// `previous_out` so the caller's tape can feed it to the earlier layer's
// own reverse-backward call as its `current`.
template <typename scalar_t>
__global__ void tria_step_reverse_backward_kernel(
    const scalar_t* __restrict__ grad_carry_new,
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ current,
    scalar_t* __restrict__ grad_r, scalar_t* __restrict__ grad_i, scalar_t* __restrict__ grad_o,
    scalar_t* __restrict__ grad_previous, scalar_t* __restrict__ previous_out,
    float alpha, int axis, int64_t n) {
    const int64_t idx=blockIdx.x*(int64_t)blockDim.x+threadIdx.x;
    if(idx>=n) return;
    const float rv=(float)r[idx],iv=(float)i_[idx],ov=(float)o[idx];
    float m[9],cur[9],prev[9],pre[9],gout[9],gpre[9],gm[9],gprev[9];
    tria_carrier_build9(rv,iv,ov,alpha,axis,m);
    #pragma unroll
    for(int k=0;k<9;++k){ cur[k]=(float)current[idx*9+k]; gout[k]=(float)grad_carry_new[idx*9+k]; }
    tria_reverse_prev9(m,cur,prev);
    tria_matmul9(m,prev,pre);
    const float scale=tria_rms9(pre);
    tria_rms_backward9(gout,pre,scale,gpre);
    tria_matmul_right_transpose9(gpre,prev,gm);  // dM = dpre @ prev^T
    tria_matmul_left_transpose9(m,gpre,gprev);   // dprev = M^T @ dpre
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
