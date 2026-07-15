#pragma once
#include <stdint.h>
#include <math.h>

// Full-rank non-learned local Tria factor.
// a=r*i, b=r*o, c=i*o parameterize K=[[0,-c,b],[c,0,-a],[-b,a,0]].
// M=(I+alpha*K)@R_axis, with fixed +90 degree signed permutations around
// z/x/y for axis 0/1/2. Everything is float so half/bfloat inputs accumulate
// in FP32; callers cast only final stored values.
//
// abc_bound: a,b,c are soft-clipped to [-abc_bound,abc_bound] via
// abc_bound*tanh(x/abc_bound) BEFORE alpha scaling -- ADDED after a real
// 32,500-step checkpoint showed |o| growing from 0.51 at init to 1.25
// post-training, with |r*o| reaching 512 in outlier neurons/positions.
// maxabs-normalizing the accumulated carry afterward does NOT fix this: it
// only rescales the whole matrix by a scalar, which leaves every singular-
// value RATIO (the condition number) exactly unchanged (verified
// numerically). Weight decay does not fix it either -- r/i/o are FFN
// activations, not weights, and nothing in the main LM loss penalizes their
// magnitude on Tria's behalf. This makes the "a,b,c stay small" assumption
// (that the single-factor stability bound kappa(M)<=sqrt(1+3*alpha^2)
// relies on) a structural guarantee instead of something training can
// silently violate. abc_bound=1.0f is the default, matching the PyTorch
// reference (tria.py's build_tria_pytorch) exactly; verified end-to-end on
// the real checkpoint: calibration's 90%-population stable_horizon went
// from 9/128 (broken) to 128/128 (full window) with this bound in place,
// while real held-out perplexity moved by <0.001 nats (negligible) --
// the bound only engages for rare outliers, near-identity everywhere else.
//
// NOT YET VALIDATED ON REAL GPU HARDWARE (this environment has none) --
// the PyTorch reference path (tria.py) is the one that was actually run and
// measured; this CUDA change mirrors it by construction (same abc_bound,
// same tanh, same chain rule) but needs a real-hardware forward/backward
// parity check (matching this codebase's own existing practice of gating
// _CUDA_TRIA_ENABLED behind exactly that kind of validation) before
// trusting it in production the way the PyTorch path already is.
__device__ __forceinline__ void tria_carrier_build9(
    float r, float i, float o, float alpha, int axis, float m[9],
    float abc_bound = 1.0f) {
    const float a = alpha * abc_bound * tanhf((r * i) / abc_bound);
    const float b = alpha * abc_bound * tanhf((r * o) / abc_bound);
    const float c = alpha * abc_bound * tanhf((i * o) / abc_bound);
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

// d(abc_bound*tanh(x/abc_bound))/dx = 1 - tanh(x/abc_bound)^2, evaluated at
// each of the three raw products. Backward needs this extra chain-rule
// factor now that a,b,c are no longer linear in r,i,o.
__device__ __forceinline__ void tria_carrier_tanh_derivs(
    float r, float i, float o, float abc_bound,
    float& d_a_raw, float& d_b_raw, float& d_c_raw) {
    const float ta = tanhf((r * i) / abc_bound);
    const float tb = tanhf((r * o) / abc_bound);
    const float tc = tanhf((i * o) / abc_bound);
    d_a_raw = 1.0f - ta * ta;
    d_b_raw = 1.0f - tb * tb;
    d_c_raw = 1.0f - tc * tc;
}

// Chain dL/dM -> dL/d(ri,ro,io). alpha is already included here. r,i,o are
// now needed (not just gm) to chain through the abc_bound tanh -- pass the
// SAME r,i,o used to build the forward matrix this gradient corresponds to.
__device__ __forceinline__ void tria_carrier_grad_abc(
    const float gm[9], float alpha, int axis,
    float r, float i, float o,
    float& d_a, float& d_b, float& d_c, float abc_bound = 1.0f) {
    if (axis == 0) {
        d_a = alpha * (-gm[5] + gm[6]);
        d_b = alpha * ( gm[2] + gm[7]);
        d_c = alpha * (-gm[0] - gm[4]);
    } else if (axis == 1) {
        d_a = alpha * (-gm[4] - gm[8]);
        d_b = alpha * ( gm[1] - gm[6]);
        d_c = alpha * ( gm[2] + gm[3]);
    } else {
        d_a = alpha * ( gm[3] + gm[7]);
        d_b = alpha * (-gm[0] - gm[8]);
        d_c = alpha * (-gm[1] + gm[5]);
    }
    float dt_a, dt_b, dt_c;
    tria_carrier_tanh_derivs(r, i, o, abc_bound, dt_a, dt_b, dt_c);
    d_a *= dt_a;
    d_b *= dt_b;
    d_c *= dt_c;
}

__device__ __forceinline__ float tria_absmax9(const float v[9]) {
    float amax = fabsf(v[0]);
    #pragma unroll
    for (int k = 1; k < 9; ++k) amax = fmaxf(amax, fabsf(v[k]));
    return amax;
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
// factor `local` and the maxabs-normalized post-step state `cur`
// (cur=pre/s, pre=local@prev, s=max|pre|), recover the maxabs-normalized
// pre-step state `prev`. The unknown scale s cancels exactly: prev_raw =
// local^-1 @ (cur*s) = s*(local^-1@cur), and renormalizing prev_raw by its
// own max-abs reproduces prev regardless of s -- so s never needs to be
// computed. Used by temporal_carry_endpoint's backward instead of storing
// the whole [B,T,H,3,3] forward trajectory.
__device__ __forceinline__ void tria_reverse_prev9(
    const float local[9], const float cur[9], float prev[9]) {
    float inv[9], raw[9];
    tria_invert9(local, inv);
    tria_matmul9(inv, cur, raw);
    const float s = fmaxf(tria_absmax9(raw), 1.0e-6f);
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
