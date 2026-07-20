#pragma once
#include <stdint.h>
#include <math.h>
#include "../carrier.cuh"

template <typename scalar_t>
__device__ __forceinline__ float tc_load(const scalar_t* p) { return (float)(*p); }
template <typename scalar_t>
__device__ __forceinline__ void tc_store(scalar_t* p, float v) { *p = (scalar_t)v; }

template <typename scalar_t>
__global__ void temporal_carry_forward_kernel(
    const scalar_t* __restrict__ depth_carry,
    const bool* __restrict__ reset_mask,
    float* __restrict__ document_carry,
    float* __restrict__ scale_out,
    int64_t B, int64_t T, int64_t H) {
    const int64_t stream=blockIdx.x*(int64_t)blockDim.x+threadIdx.x;
    if(stream>=B*H) return;
    const int64_t b=stream/H, h=stream-b*H;
    const int64_t base=b*T*H*9+h*9, step=H*9;
    const int64_t rbase=b*T, sbase=b*T*H+h, sstep=H;
    float acc[9]={0,0,0,0,0,0,0,0,0};
    bool have=false;
    for(int64_t t=0;t<T;++t){
        const int64_t off=base+t*step;
        float local[9],pre[9];
        #pragma unroll
        for(int k=0;k<9;++k) local[k]=tc_load(depth_carry+off+k);
        const bool reset=reset_mask[rbase+t]||!have;
        if(reset){
            #pragma unroll
            for(int k=0;k<9;++k) pre[k]=local[k];
            have=true;
        } else {
            tria_matmul9(local,acc,pre);
        }
        const float scale=tria_rms9(pre);
        const float inv=1.0f/scale;
        #pragma unroll
        for(int k=0;k<9;++k) acc[k]=pre[k]*inv;
        #pragma unroll
        for(int k=0;k<9;++k) document_carry[off+k]=acc[k];
        scale_out[sbase+t*sstep]=scale;
    }
}

template <typename scalar_t>
__global__ void temporal_carry_backward_kernel(
    const float* __restrict__ grad_document_carry,
    const scalar_t* __restrict__ depth_carry,
    const float* __restrict__ document_carry,
    const float* __restrict__ scale,
    const bool* __restrict__ reset_mask,
    scalar_t* __restrict__ grad_depth_carry,
    int64_t B, int64_t T, int64_t H) {
    const int64_t stream=blockIdx.x*(int64_t)blockDim.x+threadIdx.x;
    if(stream>=B*H) return;
    const int64_t b=stream/H, h=stream-b*H;
    const int64_t base=b*T*H*9+h*9, step=H*9;
    const int64_t rbase=b*T, sbase=b*T*H+h, sstep=H;
    float grad_from_future[9]={0,0,0,0,0,0,0,0,0};
    for(int64_t t=T-1;t>=0;--t){
        const int64_t off=base+t*step;
        float gy[9];
        #pragma unroll
        for(int k=0;k<9;++k)
            gy[k]=grad_document_carry[off+k]+grad_from_future[k];
        const bool reset=reset_mask[rbase+t]||(t==0);
        float local[9],pre[9],gpre[9];
        #pragma unroll
        for(int k=0;k<9;++k) local[k]=tc_load(depth_carry+off+k);
        if(reset){
            #pragma unroll
            for(int k=0;k<9;++k) pre[k]=local[k];
            tria_rms_backward9(gy,pre,scale[sbase+t*sstep],gpre);
            #pragma unroll
            for(int k=0;k<9;++k){
                tc_store(grad_depth_carry+off+k,gpre[k]);
                grad_from_future[k]=0.0f;
            }
        } else {
            float prev[9],glocal[9],gprev[9];
            #pragma unroll
            for(int k=0;k<9;++k) prev[k]=document_carry[(off-step)+k];
            tria_matmul9(local,prev,pre);
            tria_rms_backward9(gy,pre,scale[sbase+t*sstep],gpre);
            tria_matmul_right_transpose9(gpre,prev,glocal);
            tria_matmul_left_transpose9(local,gpre,gprev);
            #pragma unroll
            for(int k=0;k<9;++k){
                tc_store(grad_depth_carry+off+k,glocal[k]);
                grad_from_future[k]=gprev[k];
            }
        }
    }
}
