#pragma once
#include <stdint.h>
#include <math.h>

// Full-rank non-learned local Tria factor.
// a=r*i, b=r*o, c=i*o parameterize K=[[0,-c,b],[c,0,-a],[-b,a,0]].
// M=(I+alpha*K)@R_axis, with fixed +90 degree signed permutations around
// z/x/y for axis 0/1/2. Everything is float so half/bfloat inputs accumulate
// in FP32; callers cast only final stored values.
//
// a,b,c are JOINTLY RMS-normalized before tanh (NOT independently
// hard-clipped by a fixed abc_bound): rms=sqrt(mean(ri^2,ro^2,io^2)+eps),
// then tanh(x/rms) per component, BEFORE alpha scaling -- ADDED after a real
// 32,500-step checkpoint showed |o| growing from 0.51 at init to 1.25
// post-training, with |r*o| reaching 512 in outlier neurons/positions.
// normalizing the accumulated carry afterward does NOT fix this: it
// only rescales the whole matrix by a scalar, which leaves every singular-
// value RATIO (the condition number) exactly unchanged (verified
// numerically). Weight decay does not fix it either -- r/i/o are FFN
// activations, not weights, and nothing in the main LM loss penalizes their
// magnitude on Tria's behalf. This makes the "a,b,c stay small" assumption
// (that the single-factor stability bound kappa(M)<=sqrt(1+3*alpha^2)
// relies on) a structural guarantee instead of something training can
// silently violate. The earlier fixed abc_bound=1.0f clip was near-identity
// at typical activation scale and only engaged once |r*i| etc. crossed the
// absolute threshold -- which stopped catching outliers once activation
// scale itself drifted upward over a long training run. The joint-RMS form
// is self-referential/scale-invariant instead: it bounds a,b,c relative to
// EACH OTHER, not to a fixed constant, so the guarantee holds identically at
// any activation scale. eps=1e-6f is the default, matching the PyTorch
// reference (tria.py's build_tria_pytorch) exactly.
//
__device__ __forceinline__ void tria_carrier_build9(
    float r, float i, float o, float alpha, int axis, float m[9],
    float eps = 1e-6f) {
    const float x0 = r * i, x1 = r * o, x2 = i * o;
    const float rms = sqrtf((x0 * x0 + x1 * x1 + x2 * x2) * (1.0f / 3.0f) + eps);
    const float a = alpha * tanhf(x0 / rms);
    const float b = alpha * tanhf(x1 / rms);
    const float c = alpha * tanhf(x2 / rms);
    if (axis == 0) {              // @ Rz(+90)
        m[0] = -c; m[1] = -1.0f; m[2] =  b;
        m[3] = 1.0f; m[4] = -c;  m[5] = -a;
        m[6] =  a; m[7] =  b;    m[8] = 1.0f;
    } else if (axis == 1) {       // @ Rx(+90)
        m[0] = 1.0f; m[1] =  b;   m[2] =  c;
        m[3] =  c;   m[4] = -a;   m[5] = -1.0f;
        m[6] = -b;   m[7] = 1.0f; m[8] = -a;
    } else {                      // @ Ry(+90)
        m[0] = -b;   m[1] = -c;   m[2] = 1.0f;
        m[3] =  a;   m[4] = 1.0f; m[5] =  c;
        m[6] = -1.0f;m[7] =  a;   m[8] = -b;
    }
}

// Chain dL/dM -> dL/d(ri,ro,io) through the joint-RMS tanh. Unlike the old
// per-component abc_bound clip, rms=sqrt(mean(x0^2,x1^2,x2^2)+eps) depends on
// ALL THREE of x=(ri,ro,io), so the Jacobian d(a,b,c)/d(ri,ro,io) is no
// longer diagonal -- each output channel's tanh derivative couples back into
// every input through rms. Let u_k=x_k/rms, W_k=alpha*G_k*(1-tanh(u_k)^2)
// where G_k=dL/d(a,b,c)_k (the plain gm linear-combo below, alpha NOT yet
// applied). Differentiating u_k=x_k/rms gives
// du_k/dx_j = delta_kj/rms - x_k*x_j/(3*rms^3), which collapses via
// dot=sum_k(W_k*x_k) to the compact form used below:
//   dL/dx_j = W_j/rms - x_j*dot/(3*rms^3)
// r,i,o are needed (not just gm) to rebuild rms and u_k -- pass the SAME
// r,i,o used to build the forward matrix this gradient corresponds to.
__device__ __forceinline__ void tria_carrier_grad_abc(
    const float gm[9], float alpha, int axis,
    float r, float i, float o,
    float& d_ri, float& d_ro, float& d_io, float eps = 1e-6f) {
    float ga, gb, gc;  // dL/d(a,b,c), alpha NOT yet applied
    if (axis == 0) {
        ga = -gm[5] + gm[6];
        gb =  gm[2] + gm[7];
        gc = -gm[0] - gm[4];
    } else if (axis == 1) {
        ga = -gm[4] - gm[8];
        gb =  gm[1] - gm[6];
        gc =  gm[2] + gm[3];
    } else {
        ga =  gm[3] + gm[7];
        gb = -gm[0] - gm[8];
        gc = -gm[1] + gm[5];
    }
    const float x0 = r * i, x1 = r * o, x2 = i * o;
    const float rms2 = (x0 * x0 + x1 * x1 + x2 * x2) * (1.0f / 3.0f) + eps;
    const float rms = sqrtf(rms2);
    const float t0 = tanhf(x0 / rms), t1 = tanhf(x1 / rms), t2 = tanhf(x2 / rms);
    const float w0 = alpha * ga * (1.0f - t0 * t0);
    const float w1 = alpha * gb * (1.0f - t1 * t1);
    const float w2 = alpha * gc * (1.0f - t2 * t2);
    const float dot = w0 * x0 + w1 * x1 + w2 * x2;
    const float inv_rms = 1.0f / rms;
    const float cross = dot / (3.0f * rms2 * rms);
    d_ri = w0 * inv_rms - x0 * cross;
    d_ro = w1 * inv_rms - x1 * cross;
    d_io = w2 * inv_rms - x2 * cross;
}

__device__ __forceinline__ float tria_absmax9(const float v[9]) {
    float amax = fabsf(v[0]);
    #pragma unroll
    for (int k = 1; k < 9; ++k) amax = fmaxf(amax, fabsf(v[k]));
    return amax;
}

// Smooth carry normalization. RMS is differentiable in all entries and avoids
// max-abs's systematic tie at the structural +/-1 entries. It remains
// positively homogeneous (rms(c*v)=|c|*rms(v)), so the O(1) reverse trick in
// tria_reverse_prev9 still works.
__device__ __forceinline__ float tria_rms9(const float v[9], float eps = 1e-6f) {
    float sumsq = 0.0f;
    #pragma unroll
    for (int k = 0; k < 9; ++k) sumsq += v[k] * v[k];
    return sqrtf(sumsq * (1.0f / 9.0f) + eps);
}

// d(carry_k)/d(pre_j) = delta_kj/rms - pre_k*pre_j/(9*rms^3) (same derivation
// pattern as tria_carrier_grad_abc's joint-RMS Jacobian), collapsing via
// dot=sum_k(grad_out_k*pre_k) to:
//   dL/dpre_j = grad_out_j/rms - pre_j*dot/(9*rms^3)
// No tie-detection branch needed -- unlike tria_maxabs_backward9, this is the
// exact analytic gradient everywhere, not a subgradient convention.
__device__ __forceinline__ void tria_rms_backward9(
    const float grad_out[9], const float pre[9], float rms, float grad_pre[9]) {
    float dot = 0.0f;
    #pragma unroll
    for (int k = 0; k < 9; ++k) dot += grad_out[k] * pre[k];
    const float inv_rms = 1.0f / rms;
    const float cross = dot / (9.0f * rms * rms * rms);
    #pragma unroll
    for (int k = 0; k < 9; ++k) grad_pre[k] = grad_out[k] * inv_rms - pre[k] * cross;
}

__device__ __forceinline__ void tria_matmul9(
    const float a[9], const float b[9], float out[9]) {
    #pragma unroll
    for (int row = 0; row < 3; ++row) {
        #pragma unroll
        for (int col = 0; col < 3; ++col) {
            out[row * 3 + col] = fmaf(a[row * 3 + 2], b[6 + col],
                fmaf(a[row * 3 + 1], b[3 + col], a[row * 3] * b[col]));
        }
    }
}

__device__ __forceinline__ void tria_matmul_right_transpose9(
    const float a[9], const float b[9], float out[9]) {
    // out = a @ b^T
    #pragma unroll
    for (int row = 0; row < 3; ++row) {
        #pragma unroll
        for (int col = 0; col < 3; ++col) {
            out[row * 3 + col] = fmaf(a[row * 3 + 2], b[col * 3 + 2],
                fmaf(a[row * 3 + 1], b[col * 3 + 1], a[row * 3] * b[col * 3]));
        }
    }
}

__device__ __forceinline__ void tria_matmul_left_transpose9(
    const float a[9], const float b[9], float out[9]) {
    // out = a^T @ b
    #pragma unroll
    for (int row = 0; row < 3; ++row) {
        #pragma unroll
        for (int col = 0; col < 3; ++col) {
            out[row * 3 + col] = fmaf(a[6 + row], b[6 + col],
                fmaf(a[3 + row], b[3 + col], a[row] * b[col]));
        }
    }
}

// General 3x3 inverse via cofactor/adjugate. M=(I+K)@R_axis is always
// invertible here (det=1+|v|^2>0, see tria_carrier_build9), so this never
// needs a singularity guard.
__device__ __forceinline__ void tria_invert9(const float m[9], float inv[9]) {
    const float a = m[0], b = m[1], c = m[2];
    const float d = m[3], e = m[4], f = m[5];
    const float g = m[6], h = m[7], k = m[8];
    const float A =  e * k - f * h, B = -(d * k - f * g), C =  d * h - e * g;
    const float D = -(b * k - c * h), E =  a * k - c * g, F = -(a * h - b * g);
    const float G =  b * f - c * e, H = -(a * f - c * d), I =  a * e - b * d;
    const float inv_det = 1.0f / (a * A + b * B + c * C);
    inv[0] = A * inv_det; inv[1] = D * inv_det; inv[2] = G * inv_det;
    inv[3] = B * inv_det; inv[4] = E * inv_det; inv[5] = H * inv_det;
    inv[6] = C * inv_det; inv[7] = F * inv_det; inv[8] = I * inv_det;
}

// O(1) analytic reverse of the depth/temporal recurrence: given the local
// factor `local` and the RMS-normalized post-step state `cur`
// (cur=pre/s, pre=local@prev, s=rms(pre)), recover the RMS-normalized
// pre-step state `prev`. The unknown scale s cancels exactly: prev_raw =
// local^-1 @ (cur*s) = s*(local^-1@cur), and renormalizing prev_raw by its
// own RMS reproduces prev regardless of s -- so s never needs to be
// computed. Used by temporal_carry_endpoint's backward instead of storing
// the whole [B,T,H,3,3] forward trajectory.
__device__ __forceinline__ void tria_reverse_prev9(
    const float local[9], const float cur[9], float prev[9]) {
    float inv[9], raw[9];
    tria_invert9(local, inv);
    tria_matmul9(inv, cur, raw);
    const float s = tria_rms9(raw);
    const float invs = 1.0f / s;
    #pragma unroll
    for (int k = 0; k < 9; ++k) prev[k] = raw[k] * invs;
}

__device__ __forceinline__ void tria_maxabs_backward9(
    const float grad_out[9], const float pre[9], float scale, float grad_pre[9]) {
    // Match torch.amax(abs(pre)) exactly, including its equal split across
    // ties. Carrier matrices contain several literal +/-1 entries, so a
    // first-argmax subgradient would be a systematic CUDA/PyTorch mismatch,
    // not a rare corner case.
    const float inv = 1.0f / scale;
    float amax = fabsf(pre[0]);
    float dot = grad_out[0] * pre[0];
    #pragma unroll
    for (int k = 1; k < 9; ++k) {
        amax = fmaxf(amax, fabsf(pre[k]));
        dot += grad_out[k] * pre[k];
    }
    int ties = 0;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        grad_pre[k] = grad_out[k] * inv;
        ties += (fabsf(pre[k]) == amax) ? 1 : 0;
    }
    if (amax > 1.0e-6f && ties > 0) {
        const float corr = -dot / (scale * scale * (float)ties);
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            if (fabsf(pre[k]) == amax) {
                const float sign = pre[k] < 0.0f ? -1.0f : 1.0f;
                grad_pre[k] += sign * corr;
            }
        }
    }
}
