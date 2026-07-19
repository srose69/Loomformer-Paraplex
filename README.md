<!-- markdownlint-disable MD013 MD024 MD033 MD036 MD041 -->

<div align="center">

# LoomFormer-Paraplex

**Autoregressive language model with Paraplex neurons, depth-wise attention and Tria operator carry**

Causal GQA remains the token mixer. Paraplex changes the FFN. DepthAttn changes the residual route. Tria adds a small operator state that is composed through depth and across selected temporal boundaries.

by **srose69** (SimpleRose)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)
[![CUDA fused paths](https://img.shields.io/badge/CUDA-fused%20paths-green)](https://developer.nvidia.com/cuda-toolkit)
[![Attention GQA](https://img.shields.io/badge/attention-causal%20GQA-black)](#causal-gqa)
[![FFN Paraplex](https://img.shields.io/badge/FFN-Paraplex-blueviolet)](#paraplex)
[![Residual DepthAttn](https://img.shields.io/badge/residual-DepthAttn-cyan)](#depthattn)
[![Operator carry Tria](https://img.shields.io/badge/operator%20carry-Tria-orange)](#tria)
[![Activation PvPowLU](https://img.shields.io/badge/activation-PvPowLU-ff69b4)](#pvpowlu)
[![Checkpoint 113M](https://img.shields.io/badge/checkpoint-113M-yellow)](https://huggingface.co/srs6901/LoomFormer-Paraplex/)
[![SFT supported](https://img.shields.io/badge/SFT-supported-purple)](#training-sft-and-inference)
[![Status experimental](https://img.shields.io/badge/status-experimental-orange)](#status)

> Some badges are clickable.

<a href="#en">English</a> · <a href="#ru">Русский</a>

</div>

---

> **Note:** LoomFormer is active research code. The equations below describe the current implementation, but checkpoint compatibility, configuration names and fused CUDA paths may still change.
>
> **Примечание:** LoomFormer — активно развивающийся исследовательский код. Формулы ниже описывают текущую реализацию, но совместимость чекпойнтов, имена параметров и fused CUDA-пути ещё могут меняться.

<a name="en"></a>

## Contents

- [What is LoomFormer?](#what-is-loomformer)
- [Reference checkpoint](#reference-checkpoint)
- [Architecture at a glance](#architecture-at-a-glance)
- [How it works](#how-it-works)
  - [Causal GQA](#causal-gqa)
  - [DepthAttn](#depthattn)
  - [Paraplex](#paraplex)
  - [One neuron from L1H1 to L2H1](#one-neuron-from-l1h1-to-l2h1)
  - [Tria](#tria)
  - [One Tria step from L1 to L2](#one-tria-step-from-l1-to-l2)
  - [Boundary-to-boundary temporal carry](#boundary-to-boundary-temporal-carry)
  - [Final Tria readout](#final-tria-readout)
- [Runtime state and VRAM](#runtime-state-and-vram)
- [CUDA kernels and replay](#cuda-kernels-and-replay)
- [Depth and connectivity](#depth-and-connectivity)
- [Training, SFT and inference](#training-sft-and-inference)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Design constraints](#design-constraints)
- [Status](#status)
- [Citation](#citation)

---

## What is LoomFormer?

LoomFormer is a decoder-only autoregressive language model. It is still a Transformer in the literal sense: tokens are mixed by causal self-attention, generation is token-by-token, and inference uses a KV cache. The architectural experiment begins after that point.

I built LoomFormer around three coupled changes:

1. **Paraplex** replaces the ordinary scalar FFN activation with a real path, a bounded phase path and a phase-conditioned output nonlinearity.
2. **DepthAttn** replaces the fixed residual source with a learned softmax read over states produced earlier in network depth.
3. **Tria** converts the internal Paraplex coordinates into a `3×3` operator, composes that operator through layers and time, and returns selected summaries to later computation.

The implementation works entirely with real tensors and produces three internal coordinates per FFN neuron:

- `R` — the real/preactivation coordinate;
- `I` — the bounded phase coordinate;
- `O` — the activated Paraplex output before the FFN down-projection.

I use the words **pseudo-complex** and **pseudo-paravector** to describe the structure of the effective Paraplex weight. The implementation uses real tensors, but each hidden neuron is parameterized by a paired real and imaginary weight:

```math
W_{\mathrm{eff}}=\left(W_{\mathrm{real}},W_{\mathrm{imag}}
\right).
```

This is the pseudo-complex part. `W_real` produces the real coordinate, while `W_imag` produces the phase argument from the same FFN input together with attention and depth context:

```math
R=W_{\mathrm{real}}u+b_{\mathrm{real}},
```

```math
\beta
=
W_{\mathrm{imag}}
\begin{bmatrix}
Q\\
K_{\mathrm{ctx}}\\
C\\
u\\
D
\end{bmatrix}
+b_{\mathrm{imag}}.
```

The full `U` sector inside `W_imag` is a second projection of the same model-stream input `u`; the remaining sectors condition that projection on query, attended key, attention context and depth context. The two weight parts therefore form one effective pseudo-complex parameterization rather than two unrelated branches. No complex dtype or closed complex multiplication is required for this pairing.

`W_imag` is itself pseudo-paravector-valued. Its own-stream `U` sector acts as the scalar component, while the context sectors form the vector component:

```math
W_{\mathrm{imag}}=\left(W^{I,U},\left[W^{I,Q},W^{I,K},W^{I,C},W^{I,D}\right]\right).
```

The complete nesting is therefore

```math
W_{\mathrm{eff}}=\left(W_{\mathrm{real}},\left(W^{I,U},\mathbf W^{I,\mathrm{ctx}}\right)\right).
```

I call this a pseudo-paravector because it has the scalar-plus-vector organization of a paravector, while the implementation does not claim a full Clifford product or a complete Clifford basis.

The name **LoomFormer** refers to the way these paths are interwoven: ordinary causal attention, a depth history, a phase trace and an operator carry all participate in the same forward pass. The `-former` part is there because causal attention is still present; Tria does not replace attention.

## Reference checkpoint

Published checkpoints are available on Hugging Face:

**[Hugging Face checkpoint repository](https://huggingface.co/srs6901/LoomFormer-Paraplex/)**

The reference run currently described in this README used:

| Property | Value |
| --- | ---: |
| Parameters | `112,999,621` |
| Blocks | `10` |
| Model width | `768` |
| Query heads | `8 × 96` |
| GQA | group size `4`, therefore `2` KV heads |
| FFN hidden width | `3072` |
| Sequence length | `1024` |
| Tria temporal window | `128` tokens / `8` windows per sequence |
| Precision | `bf16` |
| Embeddings | untied |
| Dataset | FSS1STR |
| Training hardware | one NVIDIA L40S |
| Observed throughput | approximately `27.3k tok/s` |

At step `150,000`, I measured a full sequential evaluation over `45,040,704` tokens:

| Metric | Value |
| --- | ---: |
| Full evaluation loss | `3.2584` nats/token |
| Bits per token | `4.7009` |
| Bits per byte | `1.3288` |

The nominal `120,000`-step training budget was `4,608,000,000` tokens, or approximately `40.8` training tokens per parameter. The source dataset contains approximately `4,684,230,241` tokens, or `41.5` data tokens per parameter. The published run was continued beyond the nominal budget to step `150,000`.

These numbers describe one dataset, tokenizer, configuration and hardware setup. They are not presented as a cross-model benchmark.

## Architecture at a glance

For block `l`, token `t`, the main path is:

```text
h_l,t
  │
  ├── causal GQA ───────────────► attention output, q, attended-k, v-context
  │
  ├── DepthAttn over depth history ─► attention skip
  │
  └── LayerNorm(skip + attention)
                    │
                    ▼
                  u_l,t
                    │
                    ├── second DepthAttn ─► FFN skip and depth context d_l,t
                    │
                    ├── Paraplex(u, q, attended-k, context, d, Tria gate)
                    │         └── R_l,t,h , I_l,t,h , O_l,t,h
                    │
                    ├── Tria depth composition ─► C_l,t,h and p_l,t,h
                    │
                    └── LayerNorm(FFN skip + W2 O)
                              │
                              ▼
                            h_l+1,t
```

After the last block, each token has a finished depth carrier `D_t` of shape `[H,3,3]`. Outside the block stack, these carriers are composed in token order until a fixed window boundary or an explicit `<CARRY>` boundary fires. The fired endpoint has two uses:

```text
D_s, D_s+1, ... , D_b
          │
          └── temporal composition ─► A_b
                                       ├── sparse final cross-attention key/value
                                       └── seed for Tria layer 1 of the next segment
```

The refeed seed is consumed once at the next segment start. Temporal accumulation is then restarted from that newly depth-composed token, which prevents the boundary carrier from being multiplied into the path twice.

## How it works

### Causal GQA

The attention path is ordinary causal grouped-query attention with YaRN-scaled rotary position embeddings. Query heads may share fewer key/value heads.

For one query head, suppressing batch and head indices:

```math
A_t = \mathrm{softmax}\!\left(\frac{Q_tK_{\le t}^{\top}}{\sqrt{d_h}}\right),
\qquad
V^{\mathrm{ctx}}_t = A_tV_{\le t},
\qquad
K^{\mathrm{ctx}}_t = A_tK_{\le t}.
```

LoomFormer keeps both the attention context `Vctx` and the attended key context `Kctx`. The ordinary attention output is projected back into the residual stream, while `Q`, `Kctx` and `Vctx` are also passed into Paraplex.

The causal-attention implementation supports flat, chunked and token-by-token execution. Packed SFT rows use explicit block-diagonal masks so examples packed into one tensor cannot attend across their boundaries.

### DepthAttn

A standard residual block always adds the immediately preceding hidden state. LoomFormer instead keeps keys and values for the sequence of states already produced in network depth.

For sublayer `s`, a learned query `q_s` reads that history:

```math
\pi_{s,j} = \mathrm{softmax}_{j}
\left(\frac{\langle q_s,k_j\rangle}{\sqrt{d_h}}\right),
\qquad
D_s = \sum_{j\le s}\pi_{s,j}v_j,
\qquad
\mathrm{skip}_s = W^{\mathrm{depth}}_{o,s}D_s.
```

The softmax axis is **depth**, not token position. There are two reads per block:

- one read supplies the skip used around causal attention;
- the second supplies both the FFN skip and the depth context given to Paraplex.

The readout projection may be shared by all sublayers or separated per sublayer. Optional RMS fixing is available for the depth Q/K/V inputs, and an optional RMS cap can limit unusually large residual branches without amplifying quiet ones.

### Paraplex

For a token representation `u` and hidden neuron `h`, the ordinary FFN up-projection first produces the weight-dependent term

```math
X_{t,h}=(W_{\mathrm{real}}u_t)_h.
```

When no Tria signal enters the layer,

```math
R_{t,h}=X_{t,h}+b_h.
```

When a Tria gate enters from the previous layer, it scales only `X`, not the bias:

```math
R_{t,h}=X_{t,h}\left(1+\gamma_l p_{t,h}\right)+b_h,
\qquad
\gamma_l=0.25\tanh(\widehat\gamma_l).
```

In vector form,

```math
R_t=
\mathrm{diag}\!\left(\mathbf 1+\gamma_l p_t\right)
W_{\mathrm{real},l}u_t+b_l.
```

The stored matrix `W_real` is not rewritten by Tria. The carrier creates a token-dependent diagonal gain on its output rows. Because every carrier is max-absolute normalized and the nine-slot selector is a softmax mixture, `|p_{t,h}|≤1`; therefore the multiplicative factor stays between `0.75` and `1.25`. The bias remains an untranslated origin rather than being scaled together with the input-dependent response.

For the gate itself,

```math
\frac{\partial R_{t,h}}{\partial X_{t,h}}=1+\gamma_l p_{t,h},
\qquad
\frac{\partial R_{t,h}}{\partial b_h}=1.
```

The carrier changes the sensitivity to `Wx`; it does not scale the bias derivative.

The phase argument is a structured projection of five sources:

```math
\beta_{t,h}=
W^{\mathrm{imag}}_h
\begin{bmatrix}
Q_t\\
K^{\mathrm{ctx}}_t\\
V^{\mathrm{ctx}}_t\\
u_t\\
d_t
\end{bmatrix}
+b^{\mathrm{imag}}_h.
```

The phase weights are sector-constrained. In `head` mode, a hidden group reads the Q/Kctx/Vctx/Depth channels associated with its own query head while retaining the full model-state source `u`. In `open` mode, selected context sectors may cross head boundaries. The compact `w1_imag` parameter stores only live sector weights; the dense matrix used by the GEMM is a transient scatter result, not a second trainable matrix.

The current phase is bounded by a saturating sinusoidal map:

```math
I^{\mathrm{base}}_{t,h}
=
\sin\!\left(
\frac{\pi}{2}
\frac{\beta_{t,h}}{\sqrt{1+\beta_{t,h}^{2}}}
\right).
```

The same neuron's previous-token base phase may enter through a learned trace coefficient:

```math
z_{t,h}=I^{\mathrm{base}}_{t,h}+w^{\mathrm{trace}}_h I^{\mathrm{base}}_{t-1,h},
\qquad
I_{t,h}=\frac{z_{t,h}}{\sqrt{1+z_{t,h}^{2}}}.
```

Document boundaries reset that trace. The trace remains parallel during full-sequence training because position `t` reads the shifted `Ibase` from `t-1`, not the already enriched `I_t`.

The displayed phase equations are the exact forward function. Backward is configurable: `phase_grad_mode: floor` lower-bounds the cosine factor in the local derivative, while `secant` uses a secant slope to an EMA phase-radius anchor away from a small local neighborhood.

#### PvPowLU

In the reference Paraplex/PvPowLU path, a positive amplitude is built as

```math
A_{t,h}=\mathrm{softplus}(g_{t,h})>0.
```

By default the gate is self-referential, `g=R`. An optional independent `gate_proj` exists for donor-model transplantation.

The real and phase coordinates are combined before the output activation:

```math
P_{t,h}=R_{t,h}+A_{t,h}I_{t,h}.
```

For exponent parameter `m`, PvPowLU uses the positive PowLU gate

```math
G(A)=A^{\frac{m}{\sqrt{A}+1}}\sigma(A),
```

and the Paraplex output coordinate is

```math
\begin{aligned}
O_{t,h}
&=P_{t,h}G(A_{t,h}) \\
&=R_{t,h}G(A_{t,h})+I_{t,h}A_{t,h}G(A_{t,h}).
\end{aligned}
```

The FFN branch returned to the model stream is

```math
F_t=W_2O_t.
```

This is not an ordinary GLU. The gate controls both the external nonlinear gain and the amount of phase mixed into the real coordinate before that gain is applied.

#### The effective pseudo-complex weight

For every Paraplex neuron, `W_real` and `W_imag` participate in the same forward transformation. Their shared-input part can be written as

```math
W^{(u)}_{\mathrm{eff},h}=\left(W^{R}_{h},W^{I,U}_{h}\right).
```

with

```math
R_h=W^{R}_{h}u+b^{R}_{h},
```

```math
\begin{aligned}
\beta_h={}&W^{I,U}_{h}u+W^{I,Q}_{h}Q+W^{I,K}_{h}K_{\mathrm{ctx}} \\
&+W^{I,C}_{h}C+W^{I,D}_{h}D+b^{I}_{h}.
\end{aligned}
```

The phase map converts `β_h` into `I_h`, and PvPowLU combines `R_h` and `I_h` into the neuron output. Thus `W_real` is the real part of the effective weight, while `W_imag` is its pseudo-paravector imaginary part. The `U` sector supplies the imaginary projection of the same `u`; the other sectors provide its context-dependent vector coordinates.

For the default self-referential mode, `A(R)=softplus(R)` and `A'(R)=σ(R)`. The direct Paraplex derivatives are

```math
\frac{\partial O}{\partial R}
=
\left(1+I\sigma(R)\right)G(A)
+
\left(R+AI\right)G'(A)\sigma(R),
```

```math
\frac{\partial O}{\partial I}=A\,G(A).
```

Therefore the update of a real projection row contains the phase-dependent coefficient

```math
\frac{\partial \mathcal L}{\partial W_{\mathrm{real},h}}
=
\frac{\partial \mathcal L}{\partial R_h}
\left(1+\gamma_l p_h\right)u^{\top},
```

while the phase projection receives gradients through `I(β)` and through all Tria uses of `R`, `I` and `O`. The learned `W_real` is consequently co-adapted with `w_imag`, even though they remain separate parameter tensors. In this precise sense, the `W` later used in `Wx+b` has been learned under the influence of the phase path.

There is also a cross-layer route. `w_imag,l` changes `I_l`, then `O_l`, then `W_{2,l}O_l`, which changes the next residual stream. The next block's real projection multiplies that changed input:

```math
w^{\mathrm{imag}}_l
\longrightarrow I_l
\longrightarrow O_l
\longrightarrow W_{2,l}O_l
\longrightarrow u_{l+1}
\longrightarrow W_{\mathrm{real},l+1}u_{l+1}.
```

The code also supports GELU and an ungated single-input PowLU path. The equations above describe the Paraplex/PvPowLU configuration.

### One neuron from L1H1 to L2H1

Consider one token `t` and one aligned hidden channel `h=1`. I omit `t` and `h` from the notation below.

The first layer has no incoming Tria gate:

```math
R_1=w^{\mathrm{real}}_1u_1+b_1.
```

Its phase and output are

```math
\beta_1=w^{\mathrm{imag}}_1
[Q_1,K^{\mathrm{ctx}}_1,V^{\mathrm{ctx}}_1,u_1,d_1]+b^{\mathrm{imag}}_1,
```

```math
I_1=\mathrm{phase}(\beta_1,I^{\mathrm{base}}_{1,t-1}),
\qquad
A_1=\mathrm{softplus}(R_1),
```

```math
P_1=R_1+A_1I_1,
\qquad
O_1=P_1G(A_1).
```

The ordinary FFN route is dense across channels:

```math
F_1=W_{2,1}O_1,
\qquad
h_2=\mathrm{LN}_{\mathrm{ffn},1}
\left(S^{\mathrm{ffn}}_1+F_1\right).
```

Consequently, the phase of hidden neuron `H1` can affect many model-width coordinates through `W2`. Block 2 then builds its own attention output and depth skip from `h2`:

```math
u_2=\mathrm{LN}_{\mathrm{attn},2}
\left(S^{\mathrm{attn}}_2+\mathrm{Attn}_2(h_2)\right).
```

In parallel, `R1`, `I1` and `O1` produce the first Tria carrier `C1`. The first layer has one learned distribution over its nine carrier slots:

```math
w^{\mathrm{slot}}_1=\mathrm{softmax}(\lambda_1),
\qquad
p_1=\left\langle w^{\mathrm{slot}}_1,\mathrm{vec}(C_1)\right\rangle.
```

The same hidden index in the second layer receives that scalar through the identity-anchored gate:

```math
X_2=w^{\mathrm{real}}_2u_2,
\qquad
R_2=X_2\left(1+\gamma_2p_1\right)+b_2.
```

It then builds its own phase and output:

```math
\beta_2=w^{\mathrm{imag}}_2
[Q_2,K^{\mathrm{ctx}}_2,V^{\mathrm{ctx}}_2,u_2,d_2]+b^{\mathrm{imag}}_2,
```

```math
I_2=\mathrm{phase}(\beta_2,I^{\mathrm{base}}_{2,t-1}),
\qquad
A_2=\mathrm{softplus}(R_2),
```

```math
P_2=R_2+A_2I_2,
\qquad
O_2=P_2G(A_2).
```

The concrete `L1H1 → L2H1` route therefore has two simultaneous paths:

```text
L1H1 phase/output ──► dense W2 ──► residual stream ──► attention/depth ──► u2 ──► Wreal,2
          │
          └─────────► C1[H1,3,3] ──► nine-slot selector ──► gain on X2[H1]
```

The immediate Tria gate preserves the hidden-channel index. Cross-channel mixing is supplied by `W2`, the next `W_real`, attention, DepthAttn and the final population pool.

### Tria

Tria takes the three Paraplex coordinates `R`, `I` and `O` for every token and hidden channel. The implementation materializes three unique pairwise relations:

```math
a=\tanh(RI),
\qquad
b=\tanh(RO),
\qquad
c=\tanh(IO).
```

Before the `tanh` bound, the products obey

```math
(RI)(RO)(IO)=R^2I^2O^2\ge 0.
```

They are invariant to a simultaneous sign flip `(R,I,O)→(-R,-I,-O)`. Tria therefore reacts to relative sign and magnitude relations, not to an arbitrary global sign convention.

The code does **not** store a separate raw `3×3` outer-product matrix. It stores the three bounded relations above, inserts them into a skew-symmetric generator, multiplies by a fixed axis rotation, and then composes the resulting carrier. The nine slots read by selectors are the nine entries of that carrier after rotation and composition.

The generator is

```math
K(a,b,c)=
\begin{bmatrix}
0 & -c & b\\
c & 0 & -a\\
-b & a & 0
\end{bmatrix},
\qquad K^{\top}=-K.
```

The local Tria operator is

```math
T_{l,t,h}=\left(I_3+\alpha K_{l,t,h}\right)R_{\mathrm{axis}(l)},
```

where `Raxis` is a fixed `+90°` rotation around one coordinate axis. The axis cycles with layer depth. `α` is a small carrier coefficient, configured directly or selected by the startup calibration path.

Each matrix composition is normalized by its largest absolute entry:

```math
\mathcal N(M)=
\frac{M}{\max\!\left(\max_{i,j}|M_{ij}|,\varepsilon\right)}.
```

A useful invariant follows directly from this construction. If `R=I=O=0`, then `a=b=c=0`, but

```math
T=R_{\mathrm{axis}},
```

not the zero matrix. Zero local RIO modulation is the base transport mode. The local Jacobian of `(RI,RO,IO)` is zero only at the triple origin. If one partner of a weak component is nonzero, the corresponding product supplies a derivative; if both partners are nonzero, there are two such paths. The complete network also retains the phase path, the positive `softplus` amplitude, the ordinary FFN route and the previous carrier.

### One Tria step from L1 to L2

For one token and hidden channel, the first layer initializes the depth carrier:

```math
C_1=\mathcal N(T_1).
```

The second layer left-composes its local operator with the previous carrier:

```math
C_2=\mathcal N(T_2C_1).
```

The general depth recurrence is

```math
C_l=\mathcal N(T_lC_{l-1}).
```

For every non-final block, the nine entries of `C_l` are flattened and reduced by that layer's learned slot distribution:

```math
p_l=
\sum_{j=1}^{9}
\mathrm{softmax}(\lambda_l)_j
\mathrm{vec}(C_l)_j.
```

The next block applies this value only to its weight-dependent real term:

```math
X_{l+1}=W_{\mathrm{real},l+1}u_{l+1},
```

```math
R_{l+1}=X_{l+1}\odot\left(\mathbf 1+\gamma_{l+1}p_l\right)+b_{l+1}.
```

Thus Tria changes the per-neuron gain of `Wx` while leaving `b` untouched. The gate starts exactly at identity because its raw coefficient is initialized to zero. The last block has no selector because there is no following Paraplex layer; its carrier is the finished depth carrier `D_t` for that token.

### Boundary-to-boundary temporal carry

Let

```math
D_t=C_{L,t}
```

be the final depth-composed carrier produced at the end of the block stack for token `t`. Temporal Tria composes these finished matrices outside the network depth.

Inside a segment with no reset,

```math
A_t=\mathcal N(D_tA_{t-1}).
```

At a document reset, accumulation starts from the local depth carrier:

```math
A_t=\mathcal N(D_t).
```

The active chunked training path and token-by-token inference use a streaming endpoint recurrence. A fixed boundary is scheduled after at most `W=tria_temporal_window` tokens; an explicit `<CARRY>` token may fire earlier. A hard boundary is suppressed when the next token starts a new document, because there is no same-document token to seed.

Suppose a valid boundary fires at token `b_k`. The endpoint

```math
A_{b_k}
=
\mathcal N\!\left(
D_{b_k}D_{b_k-1}\cdots D_{s_k}
\right)
```

contains the depth carriers accumulated since the current segment start `s_k`, with local normalization after each implemented multiplication.

For the first token of the next segment, `t_0=b_k+1`, the boundary endpoint is injected into Tria layer 1:

```math
C_{1,t_0}^{\mathrm{seed}}
=
\mathcal N\!\left(T_{1,t_0}A_{b_k}\right).
```

The remaining layers then run normally:

```math
C_{l,t_0}^{\mathrm{seed}}
=
\mathcal N\!\left(T_{l,t_0}C_{l-1,t_0}^{\mathrm{seed}}\right),
\qquad
D_{t_0}^{\mathrm{seed}}=C_{L,t_0}^{\mathrm{seed}}.
```

After the seed has been consumed, the temporal accumulator is restarted from that newly finished depth carrier:

```math
A_{t_0}=\mathcal N\!\left(D_{t_0}^{\mathrm{seed}}\right).
```

For later tokens before the next boundary,

```math
A_t=\mathcal N(D_tA_{t-1}),
\qquad t_0<t\le b_{k+1}.
```

This reset is intentional. `A_{b_k}` already participates inside `D_{t_0}^{seed}` through the first-layer composition; multiplying the old endpoint into the temporal recurrence again would count it twice.

The complete boundary route is therefore

```text
last-block carriers D_s ... D_b
          │
          └── streaming temporal composition ──► boundary endpoint A_b
                                                   │
                                                   ├── final-CA key/value
                                                   └── T_1,next @ A_b
                                                         │
                                                         ├── compose through all layers
                                                         └── restart temporal state from D_next^seed
```

The full-sequence PyTorch reference also contains an associative segmented scan with doubling offsets. It implements the same matrix order and reset semantics, but the current chunked training path uses the endpoint recurrence above.

### Final Tria readout

A fired temporal endpoint has shape `[H,3,3]`. The reader does not first allocate one `k`-dimensional representation per hidden neuron. It scores the nine raw slots, pools them over the hidden population, and only then applies the shared value projection.

Let `a_{b,h}=vec(A_{b,h})`. A learned query and an independent slot-key projection define one normalized score direction `s` in the nine-slot space. The population weights are

```math
\rho_{b,h}
=
\mathrm{softmax}_h\!\left(
\langle s,a_{b,h}\rangle
\right).
```

The pooled boundary vector is

```math
\bar a_b=\sum_h\rho_{b,h}a_{b,h},
\qquad
z_b=\mathrm{Up}\!\left(W_{\mathrm{reader}}\bar a_b+b_{\mathrm{reader}}\right).
```

Only fired boundaries become keys and values. The final cross-attention uses a shared projection for queries and keys and a separate value projection:

```math
q_t=W_{qk}h_t,
\qquad
k_b=W_{qk}z_b,
\qquad
v_b=W_vz_b.
```

For boundaries allowed by causality and document masking,

```math
\xi_{t,b}
=
\mathrm{softmax}_{b\le t}
\left(
\frac{q_tk_b^{\top}}{\sqrt{d}}
\right),
```

```math
h'_t=h_t+\eta\sum_{b\le t}\xi_{t,b}v_b,
\qquad
\eta=\eta_{\max}\tanh(\widehat\eta).
```

A boundary is readable at its own token position and at later allowed positions. If no boundary has fired, the path is an exact identity. Tria has no separate language-model head and no auxiliary target; it is trained only through its effect on the ordinary next-token loss.

## Runtime state and VRAM

The counts below describe inference state for the actual `loomchat.py` path. The prompt is prefetched by repeated calls to `Model.step`, so the model never keeps a full `[B,T,H,3,3]` Tria tensor or a time-growing DepthAttn history in chat VRAM. Training activations, optimizer states, gradients, CUDA workspaces and autograd saves are separate costs.

### DepthAttn

With the reference `shared` readout, DepthAttn owns

```math
2N^2+N^2+2LN=3N^2+2LN
```

parameters: one shared K/V projection, one shared output projection and one query per sublayer. For `N=768` and `L=10`, this is `1,784,832` parameters, about `1.58%` of the `112,999,621`-parameter reference model. Raw bf16 storage is about `3.4 MiB`.

DepthAttn attends over network depth, not over token time. During one incremental token step it builds two temporary tensors

```math
K_{\mathrm{depth}},V_{\mathrm{depth}}
\in
\mathbb{R}^{B\times 1\times 2L\times N},
```

so the scratch size is

```math
4BLN
```

bf16 elements. For `B=1`, `L=10`, `N=768`, both tensors together occupy `60 KiB`. This amount is fixed with respect to chat length: the tensors are rebuilt for the current token and discarded after the step.

A full-sequence training forward may materialize the same depth states for a token span `T`, giving `4BTLN` elements inside that forward. That is an activation cost of the training/full-sequence path, not a persistent autoregressive cache.

### Tria

In incremental inference Tria carries no time axis. At the largest point of one token step, the live operator state is exactly three carrier-sized tensors:

```math
C_{L-1},\quad C_L,\quad A_T
\in
\mathbb{R}^{B\times H\times 3\times 3},
```

where `C_{L-1}` is the previous depth carrier, `C_L` is the current layer result, and `A_T` is the temporal accumulator retained at the tail of the network. The raw bf16 working set is therefore

```math
3\cdot 9BH
```

elements. For `B=1`, `H=3072`, this is `162 KiB`. It does not grow with the number of chat tokens.

The model also retains one phase trace `[B,H]` per Paraplex layer. For ten layers at `H=3072`, all bf16 phase traces together occupy `60 KiB`. The pending-fire flag is negligible. Thus the fixed Paraplex/Tria recurrent state is about `222 KiB` per batch element, excluding temporary R/I/O values inside the current layer.

The `[B,T,H,3,3]` tensors used by chunked training are training activations. They are not stored across the autoregressive conversation and must not be counted as a chat-length-dependent Tria cache.

### Chat caches and a 10k-token context

The ordinary GQA cache stores K and V for every layer:

```math
2BLT N_{\mathrm{kv}}d_h
```

bf16 elements. With `B=1`, `L=10`, `N_kv=2`, and `d_h=96`, this is `7.5 KiB` per cached token:

| Configured context | Ordinary GQA K/V |
| ---: | ---: |
| `1,024` tokens | `7.5 MiB` |
| `10,000` tokens | `73.24 MiB` |

The incremental final Tria cross-attention cache currently preallocates two dense buffers

```math
K_{\mathrm{CA}},V_{\mathrm{CA}}
\in
\mathbb{R}^{B\times \mathrm{seq\_len}\times N}.
```

Only fired boundary rows become valid, but the allocated storage follows the configured `seq_len`. At `N=768`, bf16 storage is:

| Configured context | Final-CA K/V allocation | Fixed-grid valid rows at `W=128` |
| ---: | ---: | ---: |
| `1,024` tokens | `3.0 MiB` | at most `7` internal fires |
| `10,000` tokens | `29.30 MiB` | at most `78` internal fires |

Therefore a model configured for a `10,000`-token chat allocates approximately

```math
73.24\ \mathrm{MiB}+29.30\ \mathrm{MiB}=102.54\ \mathrm{MiB}
```

for the two length-dependent K/V cache families at `B=1` and bf16. Adding fixed DepthAttn scratch, the three live Tria carriers and ten phase traces changes this by less than `0.3 MiB`. Model weights, allocator overhead, attention workspaces and logits are not included.

The current chat implementation has no rolling-cache wraparound. A 10k conversation therefore requires a checkpoint/configuration whose `seq_len` is at least `10,000`; the published `seq_len=1024` configuration cannot hold that conversation without rebuilding or changing the context policy.

## CUDA kernels and replay

LoomFormer does not train the Paraplex and Tria paths as a long chain of generic PyTorch elementwise operations. The repository contains real CUDA sources under `kernels/`: device code lives in `*_kernel.cuh`, standalone translation units for PTX inspection live in `*_kernel.cu`, and ATen/PyBind launchers live in `*_launcher.cu`.

The current tree contains 16 kernel groups built into six extensions:

| Extension | Fused work |
| --- | --- |
| `loomformer_beta_space` | Compact `W_imag` projection over the open `Q/K/C/U/D` sectors and its backward pass. |
| `loomformer_paraplex` | Phase recurrence, trace handling, amplitude, phase mixing, PvPowLU output, anchor update and parameter reductions. |
| `loomformer_phase_sin` | Standalone bounded phase map and its custom gradient modes. |
| `loomformer_pvpowlu` | Standalone PvPowLU forward and backward. |
| `loomformer_depth_attn_online` | Online-softmax DepthAttn forward and backward over the fixed depth history. |
| `loomformer_tria_carry` | Eleven Tria groups: initialization, seeded initialization, gated variants, depth steps, depth replay, slot mixing, population pooling, temporal carry, endpoint-only temporal carry and sparse final cross-attention. |

The fused Paraplex path computes `beta_space` and then performs phase construction, temporal phase trace, amplitude, `P=R+AI`, PvPowLU and the required backward terms without materializing each algebraic subexpression as a separate Python-level tensor. Tria kernels keep the nine entries of a local `3x3` operator, its matrix product and max-absolute normalization in registers or local kernel state; intermediate local matrices are not persisted as separate model states.

### Recompute and depth replay

The non-initial Tria depth steps use a custom replay tape during chunked training. A plain autograd implementation would save, for every layer, both the previous carrier

```math
C_{L-1}\in\mathbb{R}^{B\times T\times H\times3\times3}
```

and its normalization scale. In replay mode the forward pass saves the local `R/I/O` values already required by the Paraplex path, plus the selector weights for a gated step, but drops `C_{L-1}` and the saved scale for that step.

During backward, the missing previous carrier is recovered in one of two ways:

- for a live FP32 current carrier, the invertible local Tria factor permits an analytic reverse step;
- for bf16/fp16, or when that current carrier is no longer live, the dedicated `depth_replay` kernel reconstructs the required carrier by replaying the recorded Tria factors from the segment seed.

The replay changes storage, not the forward equation. Gradients are still computed for `R`, `I`, `O`, the previous carrier and the nine-slot selector. The trade-off is additional backward arithmetic in exchange for not retaining a full carrier-sized activation at every depth step.

### Endpoint-only temporal carry

Chunked training usually needs the final temporal state of a segment, not every normalized temporal prefix. The `temporal_carry_endpoint` kernel therefore returns only

```math
A_T\in\mathbb{R}^{B\times H\times3\times3}
```

and a small FP32 endpoint copy used by backward. It does not store a second full trajectory of temporal accumulators. Backward walks the input depth carriers in reverse and reconstructs each preceding normalized accumulator analytically from the invertible local factor and the current accumulator. This is the temporal counterpart of replay: the input depth carriers remain available, while the additional `[B,T,H,3,3]` prefix history is avoided.

Other fused Tria kernels cover the nine-slot selector, population pooling and sparse final cross-attention, so those paths do not require constructing dense Python-side operator graphs. Every extension has a PyTorch fallback when compilation or a supported CUDA dtype/device is unavailable; the fast path is selected automatically when its runtime checks pass.

Kernel extensions are built lazily with `torch.utils.cpp_extension.load` and Ninja. Source and included-header hashes are recorded in `kernels/.hashes.json`; unchanged modules are loaded from `kernels/build/` instead of being rebuilt. `TORCH_CUDA_ARCH_LIST` may be supplied explicitly, `KERNELS_VERBOSE=1` exposes the extension build log, and `KERNELS_DUMP_PTX=1` emits standalone PTX dumps for inspection after a changed build.

## Depth and connectivity

The reference checkpoint has ten named Transformer blocks. I also use an informal operational count when reasoning about the graph, but it is not a model-depth standard and it is not backend-invariant.

An earlier README draft reported `~124` operational layers by assigning `ceil(log2(128))=7` stages to the temporal prefix. That number mixed the associative full-sequence reference scan with the active chunked execution path. In the current code:

- the full-sequence PyTorch reference can resolve temporal carry with doubling offsets;
- chunked training computes one streaming endpoint per segment;
- token-by-token inference performs one temporal matrix recurrence per generated token;
- a fired boundary adds one seed composition at the first Tria layer of the next segment.

The temporal dependency depth is therefore execution-dependent: up to the segment length in the streaming path, rather than universally seven stages. I no longer present `124` as a single exact depth of the current implementation.

The static per-token block stack still contains, under my informal convention, `60` block transformations, `10` Tria depth builds/compositions and `27` inter-layer selector/reduction/gate operations before temporal unrolling. Final aggregation, cross-attention, embeddings and the LM head add their own transformations when the corresponding boundary path is active.

The same issue applies to a single exact connection count. Causal attention edges depend on document masks and sequence length; temporal edges depend on explicit and fixed boundaries; sparse final cross-attention depends on which boundaries are valid. I therefore describe the connectivity by its axes instead of publishing an unverified scalar: token attention, depth-history attention, same-channel Tria depth recurrence, temporal matrix recurrence, boundary refeed and sparse boundary-to-token attention.

## Training, SFT and inference

The repository contains complete paths for:

- tokenizer training and raw-corpus tokenization;
- prepared `.bin` token streams and streaming TXT/JSONL/Parquet/Arrow datasets;
- single-device and self-launched multi-GPU DDP pretraining;
- gradient accumulation, activation checkpointing and asynchronous evaluation;
- full sequential evaluation with loss, bits/token and bits/byte reporting;
- smart resume of weights, step, schedule and data position;
- AdamW or ATOM optimization;
- supervised fine-tuning with packed examples, loss masks and tool-call templates;
- donor checkpoint inspection and structural transplantation;
- portable `.aio` packaging and interactive terminal chat;
- AOTInductor export for a self-contained inference package.

When temporal Tria is enabled, a full input passed to `Model.forward` is processed internally as temporal chunks. Token-by-token `step` carries the analogous attention KV caches, Paraplex phase traces, Tria endpoint and sparse final-cross-attention keys. The flat batched path is used when temporal Tria is disabled or ablated; it is not the active refeed path.

The reference PyTorch implementations define the semantics. Optional fused CUDA kernels accelerate phase-space projection, PvPowLU, DepthAttn, Tria depth composition, slot reduction, temporal carry and final sparse cross-attention.

## Repository layout

```text
loomformer.py   model, data pipeline, pretraining, evaluation and AOT export
tria.py         Tria operators, depth/temporal carry, readers and final cross-attention
loomsft.py      supervised fine-tuning
loomcloner.py   donor inspection and checkpoint transplantation
loompack.py     portable .aio pack / inspect / extract utility
loomchat.py     interactive terminal chat for .aio packages
```

## Quick start

### Install

Create an environment and install the Python dependencies used by the selected workflow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pyyaml tokenizers pyarrow safetensors transformers jinja2
```

ATOM and the custom C++/CUDA extensions are optional execution paths. They require their corresponding repository modules and a working CUDA build toolchain.

### Smoke tests

```bash
python loomformer.py --smoke-test
python loomsft.py --smoke-test
python tria.py
```

### Train a tokenizer

```bash
python loomformer.py \
  --config cfg/model.yaml \
  --train-tokenizer ./datasets/raw \
  --vocab 32000 \
  --tokenizer-out tokenizer.json
```

### Pretrain

```bash
ATOM_META_SCALE=8 python -u loomformer.py \
  --train \
  --config cfg/model.yaml \
  --checkpoint ./loomformer.pt
```

If the dataset is not specified inside the YAML:

```bash
ATOM_META_SCALE=8 python -u loomformer.py \
  --train \
  --config cfg/model.yaml \
  --dataset ./datasets/train \
  --checkpoint ./loomformer.pt
```

Run on all visible CUDA devices:

```bash
python loomformer.py \
  --train \
  --device cudas \
  --config cfg/model.yaml \
  --checkpoint ./loomformer.pt
```

The built-in detached mode avoids a separate `nohup` wrapper:

```bash
ATOM_META_SCALE=8 python loomformer.py \
  --train \
  --config cfg/model.yaml \
  --checkpoint ./loomformer.pt \
  --quiet
```

### Evaluate

```bash
python loomformer.py \
  --eval \
  --checkpoint ./loomformer.pt \
  --dataset ./datasets/val \
  --eval-data-cache ram
```

### Minimal checkpoint inference

```bash
python loomformer.py \
  --infer \
  --checkpoint ./loomformer.pt \
  --prompt "The carrier matrix" \
  --max-new 128
```

### Supervised fine-tuning

```bash
python loomsft.py \
  --config cfg/sft.yaml \
  --sft-dataset ./datasets/sft \
  --init-checkpoint ./loomformer.pt \
  --checkpoint ./loomformer-sft.pt
```

### Package and chat

```bash
python loompack.py pack ./loomformer-sft.pt \
  --tokenizer tokenizer.json \
  --template chat_template.jinja \
  --quant bf16 \
  -o loomformer.aio

python loomchat.py loomformer.aio --device cuda:0
```

### Inspect a donor model

```bash
python loomcloner.py \
  --scan ./donor-model \
  --out ./cfg/donor.yaml
```

Cloning is structural transplantation, not exact architectural conversion. Compatible tensors are remapped; LoomFormer-specific Paraplex, DepthAttn and Tria parameters keep their own initialization unless a mapping explicitly defines another destination.

## Configuration

The model is configured from YAML. The main groups are:

| Area | Representative fields |
| --- | --- |
| Shape | `model_dim`, `n_q_heads`, `n_kv_heads`, `gqa_group_size`, `hidden`, `layers` |
| Attention | `attn_impl`, `attn_sdpa_compute_dtype`, `attn_sdpa_value_fusion`, `rope_*` |
| DepthAttn | `depth_attn_readout`, `depth_attn_qkv_rms`, `residual_branch_rms_cap` |
| Paraplex | `phase_sectors`, `activation`, `powlu_m`, `phase_grad_mode`, `phase_grad_floor`, `paraplex_gate_proj` |
| Tria | `tria_carry_enabled`, `tria_temporal_enabled`, `tria_temporal_window`, `tria_carrier_alpha`, calibration fields |
| Data | `dataset_format`, `text_field`, `seq_len`, `batch_size`, `prefetch_batches` |
| Training | `steps`, `lr`, `optimizer`, `weight_decay`, `grad_clip`, `grad_accum_steps`, `warmup_steps` |
| Runtime | `device`, `amp_dtype`, `grad_checkpointing`, `graph`, `save_graph`, CUDA fast-path flags |

Important shape invariants are checked when the configuration is applied:

- `model_dim` must be divisible by the query-head count;
- GQA head counts and group size must agree;
- the Paraplex hidden width must be divisible by the query-head count;
- temporal Tria requires a positive window and carrier coefficient;
- the configured input length cannot exceed `seq_len`.

The startup Tria calibration path can select a temporal window and carrier coefficient from a candidate population using condition number, effective rank and population-pass thresholds.

## Requirements

Core requirements:

- Python 3.10 or newer;
- PyTorch;
- NumPy;
- PyYAML;
- Hugging Face `tokenizers`.

Workflow-dependent packages:

- PyArrow for Arrow and Parquet datasets;
- Jinja2 for chat templates;
- Safetensors and Transformers for donor workflows;
- a C++/CUDA toolchain for fused kernels;
- ATOM when `optimizer: atom` is selected.

CPU execution is supported by the reference paths. CUDA is required for practical training of the reference-sized models and for the fused custom operators.

## Design constraints

LoomFormer deliberately keeps causal attention. Its full token-mixing path therefore remains quadratic in sequence length. Tria is not advertised as a replacement for attention; it adds an operator route across depth and selected temporal boundaries.

The architecture carries more state than a minimal decoder Transformer: KV caches, a depth history, per-layer phase traces, depth carriers, temporal carriers and sparse boundary keys. Parity must be checked between the active chunked forward transitions and token-by-token `step`; the separate flat path is used when temporal refeed is inactive.

The local Tria matrix is small, but it exists for every token and Paraplex hidden channel. Fused paths avoid several large intermediate materializations, yet Tria still adds compute and memory traffic. Its benefit has to be evaluated against that cost by ablation, held-out loss, generation behaviour and profiling.

The model's initial loss should not be compared step-for-step with a vanilla pre-norm Transformer without controlling for normalization placement, DepthAttn residuals, Paraplex activation, initialization and tokenizer. The reference run converged, but that single result does not establish scaling behaviour for every configuration.

No single operational-depth or connection-count scalar is treated as an invariant in this revision. The current execution graph depends on the temporal backend and on fired boundaries. The conventional architecture description remains: 10 decoder blocks with two depth reads, causal GQA, Paraplex FFN and optional Tria carry.

## Status

LoomFormer currently trains, evaluates, fine-tunes, packages and generates text. The 113M run converged on the stated corpus and the full evaluation metrics are reported above. SFT and LoomChat are functional; LoomChat is still relatively VRAM-heavy because the current implementation prioritizes architectural equivalence and inspectability over a minimal inference runtime.

The project is experimental. Configuration names, checkpoint migrations, CUDA kernels and package metadata are not yet a stable public API.

## Citation

Until there is a paper or archived release, cite the repository and checkpoint collection:

```bibtex
@software{loomformer_paraplex,
  author = {srose69},
  title  = {LoomFormer-Paraplex},
  year   = {2026},
  url    = {https://github.com/srose69/LoomFormer-Paraplex}
}
```

---

<a name="ru"></a>

## Содержание

- [Что такое LoomFormer?](#что-такое-loomformer)
- [Референсный чекпойнт](#референсный-чекпойнт)
- [Архитектура в одном маршруте](#архитектура-в-одном-маршруте)
- [Как это работает](#как-это-работает)
  - [Каузальное GQA](#каузальное-gqa)
  - [DepthAttn](#depthattn-1)
  - [Paraplex](#paraplex-1)
  - [Один нейрон от L1H1 к L2H1](#один-нейрон-от-l1h1-к-l2h1)
  - [Tria](#tria-1)
  - [Один шаг Tria от L1 к L2](#один-шаг-tria-от-l1-к-l2)
  - [Temporal carry от границы до границы](#temporal-carry-от-границы-до-границы)
  - [Финальное чтение Tria](#финальное-чтение-tria)
- [Runtime-состояние и VRAM](#runtime-состояние-и-vram)
- [CUDA-кернели и replay](#cuda-кернели-и-replay)
- [Глубина и связность](#глубина-и-связность)
- [Обучение, SFT и инференс](#обучение-sft-и-инференс)
- [Структура репозитория](#структура-репозитория)
- [Быстрый старт](#быстрый-старт)
- [Конфигурация](#конфигурация)
- [Требования](#требования)
- [Ограничения конструкции](#ограничения-конструкции)
- [Статус](#статус)
- [Цитирование](#цитирование)

---

## Что такое LoomFormer?

LoomFormer — decoder-only авторегрессионная языковая модель. В буквальном смысле это всё ещё Transformer: токены смешиваются каузальным self-attention, генерация идёт по одному токену, а инференс использует KV-cache. Архитектурный эксперимент начинается после этого.

Я построил LoomFormer вокруг трёх связанных изменений:

1. **Paraplex** заменяет обычную скалярную FFN-активацию real-путём, ограниченным фазовым путём и фазово-обусловленной выходной нелинейностью.
2. **DepthAttn** заменяет фиксированный источник residual на обучаемое softmax-чтение состояний, уже полученных раньше по глубине сети.
3. **Tria** превращает внутренние координаты Paraplex в оператор `3×3`, композирует его по слоям и времени и возвращает выбранные сводки в дальнейшее вычисление.

Реализация полностью работает на вещественных тензорах и формирует три внутренние координаты каждого FFN-нейрона:

- `R` — real/preactivation-координата;
- `I` — ограниченная фазовая координата;
- `O` — активированный выход Paraplex до понижающей FFN-проекции.

Термины **псевдокомплексный** и **псевдопаравекторный** описывают структуру эффективного веса Paraplex. Реализация работает с вещественными тензорами, но каждый hidden-нейрон параметризован парой real- и imaginary-весов:

```math
W_{\mathrm{eff}}=\left(W_{\mathrm{real}},W_{\mathrm{imag}}
\right).
```

Это псевдокомплексная часть конструкции. `W_real` формирует real-координату, а `W_imag` формирует фазовый аргумент из того же FFN-входа вместе с attention- и depth-контекстом:

```math
R=W_{\mathrm{real}}u+b_{\mathrm{real}},
```

```math
\beta
=
W_{\mathrm{imag}}
\begin{bmatrix}
Q\\
K_{\mathrm{ctx}}\\
C\\
u\\
D
\end{bmatrix}
+b_{\mathrm{imag}}.
```

Полный сектор `U` внутри `W_imag` является второй проекцией того же model-stream входа `u`; остальные сектора обусловливают эту проекцию query, attended key, attention context и depth-context. Поэтому обе части веса образуют единую псевдокомплексную параметризацию, а не две независимые ветви. Для такого сопряжения не требуются complex dtype и замкнутое комплексное умножение.

Сам `W_imag` имеет псевдопаравекторную структуру. Его собственный `U`-сектор играет роль скалярной части, а контекстные сектора образуют векторную часть:

```math
W_{\mathrm{imag}}=\left(W^{I,U},\left[W^{I,Q},W^{I,K},W^{I,C},W^{I,D}\right]\right).
```

Полная вложенная структура имеет вид

```math
W_{\mathrm{eff}}=\left(W_{\mathrm{real}},\left(W^{I,U},\mathbf W^{I,\mathrm{ctx}}\right)\right).
```

Я называю её псевдопаравекторной из-за организации «скаляр плюс вектор»; при этом реализация не заявляет полный Clifford product или полный базис клиффордовой алгебры.

Название **LoomFormer** относится к тому, как эти пути переплетаются: обычное каузальное attention, история по глубине, фазовый trace и операторный carry участвуют в одном forward. Суффикс `-former` оставлен потому, что каузальное attention никуда не исчезло; Tria не заменяет attention.

## Референсный чекпойнт

Опубликованные чекпойнты находятся на Hugging Face:

**[Репозиторий чекпойнтов на Hugging Face](https://huggingface.co/srs6901/LoomFormer-Paraplex/)**

Референсный запуск, описанный в этом README, использовал:

| Свойство | Значение |
| --- | ---: |
| Параметры | `112,999,621` |
| Блоки | `10` |
| Ширина модели | `768` |
| Query heads | `8 × 96` |
| GQA | group size `4`, следовательно `2` KV-heads |
| Hidden-width FFN | `3072` |
| Длина последовательности | `1024` |
| Temporal window Tria | `128` токенов / `8` окон на последовательность |
| Точность | `bf16` |
| Embeddings | untied |
| Датасет | FSS1STR |
| Железо обучения | одна NVIDIA L40S |
| Наблюдаемая скорость | примерно `27.3k tok/s` |

На шаге `150,000` мной был выполнен полный последовательный eval на `45,040,704` токенах:

| Метрика | Значение |
| --- | ---: |
| Full evaluation loss | `3.2584` nats/token |
| Bits per token | `4.7009` |
| Bits per byte | `1.3288` |

Номинальный бюджет на `120,000` шагов составлял `4,608,000,000` токенов, или примерно `40.8` обучающих токена на параметр. Исходный датасет содержит около `4,684,230,241` токена, или `41.5` токена данных на параметр. Опубликованный запуск был продолжен после номинального бюджета до шага `150,000`.

Эти числа относятся к одному датасету, токенизатору, конфигурации и железу. Я не выдаю их за межмодельный benchmark.

## Архитектура в одном маршруте

Для блока `l` и токена `t` основной путь выглядит так:

```text
h_l,t
  │
  ├── causal GQA ───────────────► attention output, q, attended-k, v-context
  │
  ├── DepthAttn по depth-history ─► attention skip
  │
  └── LayerNorm(skip + attention)
                    │
                    ▼
                  u_l,t
                    │
                    ├── второй DepthAttn ─► FFN skip и depth-context d_l,t
                    │
                    ├── Paraplex(u, q, attended-k, context, d, Tria gate)
                    │         └── R_l,t,h , I_l,t,h , O_l,t,h
                    │
                    ├── depth-композиция Tria ─► C_l,t,h и p_l,t,h
                    │
                    └── LayerNorm(FFN skip + W2 O)
                              │
                              ▼
                            h_l+1,t
```

После последнего блока каждый токен имеет законченный depth-carrier `D_t` формы `[H,3,3]`. Вне стека блоков эти carrier-матрицы композируются по токенам до фиксированной границы окна или явной границы `<CARRY>`. Сработавший endpoint используется двумя путями:

```text
D_s, D_s+1, ... , D_b
          │
          └── temporal composition ─► A_b
                                       ├── key/value для sparse final cross-attention
                                       └── seed для первого Tria-слоя следующего сегмента
```

Refeed-seed потребляется один раз в начале следующего сегмента. Затем temporal accumulator перезапускается от нового depth-composed токена, поэтому boundary carrier не умножается в тракт дважды.

## Как это работает

### Каузальное GQA

Attention-путь — обычное каузальное grouped-query attention с YaRN-scaled rotary position embeddings. Несколько query-heads могут разделять один key/value-head.

Для одной query-head, без batch- и head-индексов:

```math
A_t = \mathrm{softmax}\!\left(\frac{Q_tK_{\le t}^{\top}}{\sqrt{d_h}}\right),
\qquad
V^{\mathrm{ctx}}_t = A_tV_{\le t},
\qquad
K^{\mathrm{ctx}}_t = A_tK_{\le t}.
```

LoomFormer сохраняет и attention context `Vctx`, и attended key context `Kctx`. Обычный attention-output проецируется обратно в residual stream, а `Q`, `Kctx` и `Vctx` одновременно передаются в Paraplex.

Реализация causal attention поддерживает flat, chunked и token-by-token режимы. Packed-строки SFT используют явную block-diagonal mask, поэтому примеры внутри одной строки не могут видеть друг друга через границу.

### DepthAttn

Обычный residual-блок всегда прибавляет непосредственно предыдущее hidden-state. LoomFormer хранит keys и values состояний, уже произведённых раньше по глубине сети.

Для sublayer `s` обучаемый запрос `q_s` читает эту историю:

```math
\pi_{s,j} = \mathrm{softmax}_{j}
\left(\frac{\langle q_s,k_j\rangle}{\sqrt{d_h}}\right),
\qquad
D_s = \sum_{j\le s}\pi_{s,j}v_j,
\qquad
\mathrm{skip}_s = W^{\mathrm{depth}}_{o,s}D_s.
```

Softmax идёт по оси **глубины**, а не по токенам. В каждом блоке выполняются два чтения:

- первое даёт skip вокруг causal attention;
- второе даёт FFN-skip и depth context, который поступает в Paraplex.

Readout-проекция может быть общей для всех sublayers или отдельной для каждой. Для depth Q/K/V можно включить фиксацию RMS, а residual-ветви можно ограничить RMS-cap без усиления тихих ветвей.

### Paraplex

Для token representation `u` и hidden-нейрона `h` обычная повышающая FFN-проекция сначала формирует зависящий от весов член

```math
X_{t,h}=(W_{\mathrm{real}}u_t)_h.
```

Если в слой не входит сигнал Tria,

```math
R_{t,h}=X_{t,h}+b_h.
```

При наличии Tria-gate масштабируется только `X`, но не bias:

```math
R_{t,h}=X_{t,h}\left(1+\gamma_l p_{t,h}\right)+b_h,
\qquad
\gamma_l=0.25\tanh(\widehat\gamma_l).
```

В векторной форме

```math
R_t=
\mathrm{diag}\!\left(\mathbf 1+\gamma_l p_t\right)
W_{\mathrm{real},l}u_t+b_l.
```

Хранимая матрица `W_real` не переписывается механизмом Tria. Carrier создаёт зависимый от токена диагональный gain на выходных строках проекции. Поскольку каждая carrier-матрица max-absolute-нормализована, а selector девяти слотов является softmax-смесью, `|p_{t,h}|≤1`; множитель остаётся между `0.75` и `1.25`. Bias сохраняется как несмасштабированное начало координат, а не умножается вместе с входозависимым ответом.

Для самого gate:

```math
\frac{\partial R_{t,h}}{\partial X_{t,h}}=1+\gamma_l p_{t,h},
\qquad
\frac{\partial R_{t,h}}{\partial b_h}=1.
```

Carrier меняет чувствительность к `Wx`, но не масштабирует производную по bias.

Фазовый аргумент является структурированной проекцией пяти источников:

```math
\beta_{t,h}=
W^{\mathrm{imag}}_h
\begin{bmatrix}
Q_t\\
K^{\mathrm{ctx}}_t\\
V^{\mathrm{ctx}}_t\\
u_t\\
d_t
\end{bmatrix}
+b^{\mathrm{imag}}_h.
```

Фазовые веса ограничены секторами. В режиме `head` группа hidden-нейронов читает Q/Kctx/Vctx/Depth-каналы своего query-head и полный model-state `u`. В режиме `open` выбранные контекстные сектора могут пересекать границы heads. Компактный параметр `w1_imag` хранит только живые секторные веса; dense-матрица для GEMM является временным результатом scatter, а не второй обучаемой матрицей.

Текущая фаза ограничивается насыщаемой синусоидальной картой:

```math
I^{\mathrm{base}}_{t,h}
=
\sin\!\left(
\frac{\pi}{2}
\frac{\beta_{t,h}}{\sqrt{1+\beta_{t,h}^{2}}}
\right).
```

Базовая фаза того же нейрона на предыдущем токене может входить через обучаемый trace-коэффициент:

```math
z_{t,h}=I^{\mathrm{base}}_{t,h}+w^{\mathrm{trace}}_h I^{\mathrm{base}}_{t-1,h},
\qquad
I_{t,h}=\frac{z_{t,h}}{\sqrt{1+z_{t,h}^{2}}}.
```

Границы документов сбрасывают trace. При full-sequence training вычисление остаётся параллельным: позиция `t` читает сдвинутый `Ibase` позиции `t-1`, а не уже обогащённый `I_t`.

Показанные фазовые формулы точно описывают forward. Backward настраивается отдельно: `phase_grad_mode: floor` ограничивает снизу cosine-множитель локальной производной, а `secant` вне малой окрестности использует секущую к EMA-якорю радиуса фазы.

#### PvPowLU

В референсном Paraplex/PvPowLU-пути положительная амплитуда строится как

```math
A_{t,h}=\mathrm{softplus}(g_{t,h})>0.
```

По умолчанию gate самореферентный: `g=R`. Для transplant донорских моделей существует опциональный независимый `gate_proj`.

Real- и phase-координаты смешиваются до выходной активации:

```math
P_{t,h}=R_{t,h}+A_{t,h}I_{t,h}.
```

При параметре степени `m` PvPowLU использует положительный PowLU-gate

```math
G(A)=A^{\frac{m}{\sqrt{A}+1}}\sigma(A),
```

а выходная координата Paraplex равна

```math
\begin{aligned}
O_{t,h}
&=P_{t,h}G(A_{t,h}) \\
&=R_{t,h}G(A_{t,h})+I_{t,h}A_{t,h}G(A_{t,h}).
\end{aligned}
```

FFN-ветвь, возвращаемая в model stream:

```math
F_t=W_2O_t.
```

Это не обычный GLU. Gate управляет и внешним nonlinear gain, и количеством фазы, добавленным к real-координате до применения этого gain.

#### Эффективный псевдокомплексный вес

Для каждого Paraplex-нейрона `W_real` и `W_imag` участвуют в одном forward-преобразовании. Их общую по входу часть можно записать как

```math
W^{(u)}_{\mathrm{eff},h}=\left(W^{R}_{h},W^{I,U}_{h}\right).
```

где

```math
R_h=W^{R}_{h}u+b^{R}_{h},
```

```math
\begin{aligned}
\beta_h={}&W^{I,U}_{h}u+W^{I,Q}_{h}Q+W^{I,K}_{h}K_{\mathrm{ctx}} \\
&+W^{I,C}_{h}C+W^{I,D}_{h}D+b^{I}_{h}.
\end{aligned}
```

Фазовая карта превращает `β_h` в `I_h`, после чего PvPowLU объединяет `R_h` и `I_h` в выход нейрона. Таким образом, `W_real` является real-частью эффективного веса, а `W_imag` — его псевдопаравекторной imaginary-частью. Сектор `U` даёт imaginary-проекцию того же `u`, а остальные сектора задают её контекстно-зависимые векторные координаты.

Для самореферентного режима по умолчанию `A(R)=softplus(R)` и `A'(R)=σ(R)`. Прямые производные Paraplex равны

```math
\frac{\partial O}{\partial R}
=
\left(1+I\sigma(R)\right)G(A)
+
\left(R+AI\right)G'(A)\sigma(R),
```

```math
\frac{\partial O}{\partial I}=A\,G(A).
```

Поэтому update строки real-проекции содержит фазозависимый коэффициент

```math
\frac{\partial \mathcal L}{\partial W_{\mathrm{real},h}}
=
\frac{\partial \mathcal L}{\partial R_h}
\left(1+\gamma_l p_h\right)u^{\top},
```

а фазовая проекция получает градиенты через `I(β)` и через все использования `R`, `I`, `O` в Tria. Обученное значение `W_real` тем самым коадаптируется с `w_imag`, хотя это разные parameter tensors. В этом точном смысле матрица `W`, позднее используемая в `Wx+b`, была обучена под влиянием фазового пути.

Есть и межслойный путь. `w_imag,l` меняет `I_l`, затем `O_l`, затем `W_{2,l}O_l`, то есть следующий residual stream. Real-проекция следующего блока умножает уже изменённый вход:

```math
w^{\mathrm{imag}}_l
\longrightarrow I_l
\longrightarrow O_l
\longrightarrow W_{2,l}O_l
\longrightarrow u_{l+1}
\longrightarrow W_{\mathrm{real},l+1}u_{l+1}.
```

Код также поддерживает GELU и ungated single-input PowLU. Формулы выше относятся к конфигурации Paraplex/PvPowLU.

### Один нейрон от L1H1 к L2H1

Рассмотрим один токен `t` и один согласованный hidden-канал `h=1`. Ниже индексы `t` и `h` опущены.

В первый слой не входит Tria-gate:

```math
R_1=w^{\mathrm{real}}_1u_1+b_1.
```

Его фаза и выход:

```math
\beta_1=w^{\mathrm{imag}}_1
[Q_1,K^{\mathrm{ctx}}_1,V^{\mathrm{ctx}}_1,u_1,d_1]+b^{\mathrm{imag}}_1,
```

```math
I_1=\mathrm{phase}(\beta_1,I^{\mathrm{base}}_{1,t-1}),
\qquad
A_1=\mathrm{softplus}(R_1),
```

```math
P_1=R_1+A_1I_1,
\qquad
O_1=P_1G(A_1).
```

Обычный FFN-путь является dense по каналам:

```math
F_1=W_{2,1}O_1,
\qquad
h_2=\mathrm{LN}_{\mathrm{ffn},1}
\left(S^{\mathrm{ffn}}_1+F_1\right).
```

Следовательно, фаза hidden-нейрона `H1` может влиять на множество координат model-width через `W2`. Второй блок строит собственный attention-output и depth-skip из `h2`:

```math
u_2=\mathrm{LN}_{\mathrm{attn},2}
\left(S^{\mathrm{attn}}_2+\mathrm{Attn}_2(h_2)\right).
```

Параллельно `R1`, `I1`, `O1` создают первый carrier Tria `C1`. Первый слой имеет одно обучаемое распределение по девяти слотам carrier:

```math
w^{\mathrm{slot}}_1=\mathrm{softmax}(\lambda_1),
\qquad
p_1=\left\langle w^{\mathrm{slot}}_1,\mathrm{vec}(C_1)\right\rangle.
```

Тот же hidden-index второго слоя получает этот скаляр через identity-anchored gate:

```math
X_2=w^{\mathrm{real}}_2u_2,
\qquad
R_2=X_2\left(1+\gamma_2p_1\right)+b_2.
```

Затем слой строит собственную фазу и выход:

```math
\beta_2=w^{\mathrm{imag}}_2
[Q_2,K^{\mathrm{ctx}}_2,V^{\mathrm{ctx}}_2,u_2,d_2]+b^{\mathrm{imag}}_2,
```

```math
I_2=\mathrm{phase}(\beta_2,I^{\mathrm{base}}_{2,t-1}),
\qquad
A_2=\mathrm{softplus}(R_2),
```

```math
P_2=R_2+A_2I_2,
\qquad
O_2=P_2G(A_2).
```

Конкретный маршрут `L1H1 → L2H1` содержит два одновременных пути:

```text
phase/output L1H1 ──► dense W2 ──► residual stream ──► attention/depth ──► u2 ──► Wreal,2
          │
          └─────────► C1[H1,3,3] ──► nine-slot selector ──► gain на X2[H1]
```

Непосредственный Tria-gate сохраняет hidden-index. Межканальное смешивание выполняют `W2`, следующий `W_real`, attention, DepthAttn и финальный population pool.

### Tria

Tria получает три координаты Paraplex `R`, `I`, `O` для каждого токена и hidden-канала. Реализация материализует три уникальных попарных отношения:

```math
a=\tanh(RI),
\qquad
b=\tanh(RO),
\qquad
c=\tanh(IO).
```

До ограничения `tanh` произведения удовлетворяют

```math
(RI)(RO)(IO)=R^2I^2O^2\ge 0.
```

Они инвариантны к одновременному перевороту знаков `(R,I,O)→(-R,-I,-O)`. Поэтому Tria реагирует на относительные знаки и масштабы, а не на произвольную глобальную знаковую конвенцию.

Код **не** хранит отдельную сырую `3×3` матрицу внешнего произведения. Он хранит три ограниченных отношения, помещает их в кососимметричный генератор, умножает на фиксированный осевой поворот и затем композирует полученный carrier. Девять слотов selector относятся к девяти элементам carrier после поворота и композиции.

Генератор имеет вид

```math
K(a,b,c)=
\begin{bmatrix}
0 & -c & b\\
c & 0 & -a\\
-b & a & 0
\end{bmatrix},
\qquad K^{\top}=-K.
```

Локальный оператор Tria:

```math
T_{l,t,h}=\left(I_3+\alpha K_{l,t,h}\right)R_{\mathrm{axis}(l)},
```

где `Raxis` — фиксированный поворот на `+90°` вокруг одной координатной оси. Ось циклически меняется по глубине. `α` — небольшой коэффициент carrier, задаваемый конфигурацией или startup calibration.

Каждая композиция матриц нормализуется по максимальному абсолютному элементу:

```math
\mathcal N(M)=
\frac{M}{\max\!\left(\max_{i,j}|M_{ij}|,\varepsilon\right)}.
```

Из конструкции следует полезный инвариант. Если `R=I=O=0`, то `a=b=c=0`, но

```math
T=R_{\mathrm{axis}},
```

а не нулевая матрица. Нулевая локальная RIO-модуляция является базовым режимом переноса. Локальный Jacobian отображения `(RI,RO,IO)` равен нулю только в полном начале координат. Если один партнёр слабой компоненты ненулевой, соответствующее произведение даёт производную; если ненулевы оба партнёра, существуют два таких пути. Полная сеть дополнительно сохраняет фазовый путь, положительную амплитуду `softplus`, обычный FFN-путь и предыдущий carrier.

### Один шаг Tria от L1 к L2

Для одного токена и hidden-канала первый слой инициализирует depth-carrier:

```math
C_1=\mathcal N(T_1).
```

Второй слой слева композирует свой локальный оператор с предыдущим carrier:

```math
C_2=\mathcal N(T_2C_1).
```

Общая depth-рекурсия:

```math
C_l=\mathcal N(T_lC_{l-1}).
```

Для каждого не последнего блока девять элементов `C_l` flatten-ятся и сводятся обучаемым slot-распределением слоя:

```math
p_l=
\sum_{j=1}^{9}
\mathrm{softmax}(\lambda_l)_j
\mathrm{vec}(C_l)_j.
```

Следующий блок применяет это значение только к зависящему от весов real-члену:

```math
X_{l+1}=W_{\mathrm{real},l+1}u_{l+1},
```

```math
R_{l+1}=X_{l+1}\odot\left(\mathbf 1+\gamma_{l+1}p_l\right)+b_{l+1}.
```

Таким образом, Tria меняет per-neuron gain у `Wx`, не затрагивая `b`. Gate стартует строго как identity, поскольку его raw-коэффициент инициализируется нулём. У последнего блока нет selector: после него отсутствует следующий Paraplex-слой, а его carrier становится законченным depth-carrier `D_t` данного токена.

### Temporal carry от границы до границы

Обозначим

```math
D_t=C_{L,t}
```

законченный depth-composed carrier, полученный после стека блоков для токена `t`. Temporal Tria композирует эти готовые матрицы вне глубины сети.

Внутри сегмента без reset:

```math
A_t=\mathcal N(D_tA_{t-1}).
```

На границе документа накопление начинается с локального depth-carrier:

```math
A_t=\mathcal N(D_t).
```

Активный chunked training и token-by-token inference используют streaming endpoint recurrence. Фиксированная граница планируется не позднее чем через `W=tria_temporal_window` токенов; явный токен `<CARRY>` может сработать раньше. Hard boundary подавляется, если следующий токен начинает новый документ, поскольку seed для того же документа отсутствует.

Пусть валидная граница сработала на токене `b_k`. Endpoint

```math
A_{b_k}
=
\mathcal N\!\left(
D_{b_k}D_{b_k-1}\cdots D_{s_k}
\right)
```

содержит depth-carriers, накопленные от начала текущего сегмента `s_k`, с локальной нормализацией после каждой реализованной композиции.

Для первого токена следующего сегмента, `t_0=b_k+1`, boundary endpoint вводится в первый Tria-слой:

```math
C_{1,t_0}^{\mathrm{seed}}
=
\mathcal N\!\left(T_{1,t_0}A_{b_k}\right).
```

Остальные слои выполняют обычную depth-рекурсию:

```math
C_{l,t_0}^{\mathrm{seed}}
=
\mathcal N\!\left(T_{l,t_0}C_{l-1,t_0}^{\mathrm{seed}}\right),
\qquad
D_{t_0}^{\mathrm{seed}}=C_{L,t_0}^{\mathrm{seed}}.
```

После потребления seed temporal accumulator перезапускается от нового законченного depth-carrier:

```math
A_{t_0}=\mathcal N\!\left(D_{t_0}^{\mathrm{seed}}\right).
```

Для следующих токенов до очередной границы:

```math
A_t=\mathcal N(D_tA_{t-1}),
\qquad t_0<t\le b_{k+1}.
```

Reset сделан намеренно. `A_{b_k}` уже входит в `D_{t_0}^{seed}` через композицию первого слоя; повторное умножение старого endpoint в temporal recurrence учло бы его дважды.

Полный boundary-маршрут:

```text
last-block carriers D_s ... D_b
          │
          └── streaming temporal composition ──► boundary endpoint A_b
                                                   │
                                                   ├── key/value final-CA
                                                   └── T_1,next @ A_b
                                                         │
                                                         ├── композиция по всем слоям
                                                         └── restart temporal state от D_next^seed
```

В full-sequence PyTorch reference также имеется ассоциативный segmented scan с удваивающимися offset. Он сохраняет тот же порядок матриц и reset-семантику, но текущий chunked training использует endpoint recurrence выше.

### Финальное чтение Tria

Сработавший temporal endpoint имеет форму `[H,3,3]`. Reader не обязан сначала материализовать отдельное `k`-мерное представление каждого hidden-нейрона. Реализация оценивает девять сырых слотов, pooling-ует их по hidden-population и только после этого применяет общую value-проекцию.

Пусть `a_{b,h}=vec(A_{b,h})`. Обучаемый query и независимая slot-key-проекция задают одно нормированное направление `s` в девятимерном slot-space. Population weights:

```math
\rho_{b,h}
=
\mathrm{softmax}_h\!\left(
\langle s,a_{b,h}\rangle
\right).
```

Pooled boundary vector:

```math
\bar a_b=\sum_h\rho_{b,h}a_{b,h},
\qquad
z_b=\mathrm{Up}\!\left(W_{\mathrm{reader}}\bar a_b+b_{\mathrm{reader}}\right).
```

Key/value становятся только сработавшие границы. Final cross-attention использует общую проекцию для query и key и отдельную проекцию value:

```math
q_t=W_{qk}h_t,
\qquad
k_b=W_{qk}z_b,
\qquad
v_b=W_vz_b.
```

Для границ, разрешённых causal- и document-mask:

```math
\xi_{t,b}
=
\mathrm{softmax}_{b\le t}
\left(
\frac{q_tk_b^{\top}}{\sqrt{d}}
\right),
```

```math
h'_t=h_t+\eta\sum_{b\le t}\xi_{t,b}v_b,
\qquad
\eta=\eta_{\max}\tanh(\widehat\eta).
```

Boundary доступен для чтения на собственной позиции токена и на последующих разрешённых позициях. Если ни одна граница не сработала, путь является точным identity. У Tria нет отдельной LM-head и auxiliary target; она обучается только через влияние на обычный next-token loss.

## Runtime-состояние и VRAM

Расчёты ниже относятся к фактическому inference-path в `loomchat.py`. Prompt прогоняется последовательными вызовами `Model.step`, поэтому модель не хранит в VRAM полный Tria-тензор `[B,T,H,3,3]` и не накапливает DepthAttn-history по времени. Training activations, optimizer states, gradients, CUDA workspaces и сохранения autograd считаются отдельно.

### DepthAttn

В референсном режиме `shared` DepthAttn содержит

```math
2N^2+N^2+2LN=3N^2+2LN
```

параметров: общую K/V-проекцию, общую output-проекцию и по одному query на sublayer. При `N=768`, `L=10` это `1,784,832` параметра, около `1.58%` от `112,999,621` параметров референсной модели. Сырые bf16-веса занимают около `3.4 MiB`.

DepthAttn читает историю по глубине сетки, а не по времени токенов. На одном incremental-шаге создаются два временных тензора

```math
K_{\mathrm{depth}},V_{\mathrm{depth}}
\in
\mathbb{R}^{B\times 1\times 2L\times N},
```

поэтому их суммарный размер равен

```math
4BLN
```

bf16-элементам. При `B=1`, `L=10`, `N=768` оба тензора вместе занимают `60 KiB`. Этот объём не зависит от длины чата: тензоры строятся для текущего токена и освобождаются после шага.

Full-sequence training forward может материализовать такие же depth-состояния сразу для span длины `T`, то есть `4BTLN` элементов. Это activation cost training/full-sequence path, а не постоянный autoregressive cache.

### Tria

В incremental inference у Tria нет временной оси в хранимом состоянии. В наиболее ёмкой точке одного token-step одновременно существуют ровно три carrier-sized тензора:

```math
C_{L-1},\quad C_L,\quad A_T
\in
\mathbb{R}^{B\times H\times 3\times 3},
```

где `C_{L-1}` — carrier предыдущего depth-слоя, `C_L` — результат текущего слоя, а `A_T` — temporal accumulator на хвосте сетки. Сырой bf16 working set равен

```math
3\cdot 9BH
```

элементам. При `B=1`, `H=3072` это `162 KiB`. Размер не растёт с числом токенов в чате.

Дополнительно модель хранит по одному phase trace `[B,H]` на каждый Paraplex-слой. Для десяти слоёв и `H=3072` все bf16 phase traces вместе занимают `60 KiB`; pending-fire flag пренебрежимо мал. Итого фиксированное рекуррентное состояние Paraplex/Tria составляет около `222 KiB` на элемент batch, без временных R/I/O текущего слоя.

Тензоры `[B,T,H,3,3]`, возникающие в chunked training, являются training activations. Они не сохраняются вдоль авторегрессионного диалога и не должны считаться Tria-cache, растущим с длиной чата.

### Кэши чата и контекст на 10k токенов

Обычный GQA-cache хранит K и V каждого слоя:

```math
2BLT N_{\mathrm{kv}}d_h
```

bf16-элементов. При `B=1`, `L=10`, `N_kv=2`, `d_h=96` получается `7.5 KiB` на каждый закэшированный токен:

| Настроенный контекст | Обычный GQA K/V |
| ---: | ---: |
| `1,024` токена | `7.5 MiB` |
| `10,000` токенов | `73.24 MiB` |

Incremental cache финального Tria cross-attention сейчас заранее выделяет два плотных буфера

```math
K_{\mathrm{CA}},V_{\mathrm{CA}}
\in
\mathbb{R}^{B\times \mathrm{seq\_len}\times N}.
```

Валидными становятся только строки сработавших границ, но выделенная память определяется полным `seq_len`. При `N=768` и bf16:

| Настроенный контекст | Выделение final-CA K/V | Валидные fixed-grid строки при `W=128` |
| ---: | ---: | ---: |
| `1,024` токена | `3.0 MiB` | максимум `7` внутренних fires |
| `10,000` токенов | `29.30 MiB` | максимум `78` внутренних fires |

Следовательно, конфигурация для чата на `10,000` токенов при `B=1` и bf16 выделяет примерно

```math
73.24\ \mathrm{MiB}+29.30\ \mathrm{MiB}=102.54\ \mathrm{MiB}
```

на два семейства K/V-кэшей, зависящих от длины контекста. Фиксированные DepthAttn scratch, три живых Tria-carrier и десять phase traces добавляют менее `0.3 MiB`. Веса модели, overhead аллокатора, attention workspaces и logits сюда не входят.

В текущем chat-path нет rolling-cache wraparound. Поэтому диалог на 10k токенов требует checkpoint/config с `seq_len` не меньше `10,000`; опубликованная конфигурация `seq_len=1024` такой диалог без смены context policy не удержит.

## CUDA-кернели и replay

LoomFormer не обучает Paraplex- и Tria-пути как длинную цепочку обычных поэлементных операций PyTorch. В репозитории лежат полноценные CUDA-исходники в `kernels/`: device-код находится в `*_kernel.cuh`, отдельные translation units для просмотра PTX — в `*_kernel.cu`, а ATen/PyBind-launchers — в `*_launcher.cu`.

Текущее дерево содержит 16 групп кернелей, собранных в шесть расширений:

| Расширение | Что выполняется fused-путём |
| --- | --- |
| `loomformer_beta_space` | Компактная проекция `W_imag` по открытым секторам `Q/K/C/U/D` и её backward. |
| `loomformer_paraplex` | Фазовая рекурсия, trace, amplitude, фазовое смешивание, выход PvPowLU, обновление anchor и редукции градиентов параметров. |
| `loomformer_phase_sin` | Отдельная ограниченная фазовая карта и её custom gradient modes. |
| `loomformer_pvpowlu` | Отдельные forward и backward для PvPowLU. |
| `loomformer_depth_attn_online` | Online-softmax forward/backward DepthAttn по фиксированной истории глубины. |
| `loomformer_tria_carry` | Одиннадцать групп Tria: инициализация, seeded initialization, gated-варианты, depth steps, depth replay, slot mixing, population pooling, temporal carry, endpoint-only temporal carry и sparse final cross-attention. |

Fused Paraplex-path сначала вычисляет `beta_space`, затем внутри CUDA выполняет построение фазы, temporal phase trace, amplitude, `P=R+AI`, PvPowLU и необходимые величины backward. Каждое алгебраическое подвыражение не материализуется отдельным Python-level тензором. Tria-кернели держат девять элементов локального оператора `3x3`, матричное произведение и max-absolute normalization в регистрах или локальном состоянии кернеля; промежуточные локальные матрицы не сохраняются как отдельное состояние модели.

### Recompute и depth replay

Во время chunked training все неинициальные depth-шаги Tria используют custom replay tape. Обычный autograd-путь должен был бы сохранять на каждом слое предыдущий carrier

```math
C_{L-1}\in\mathbb{R}^{B\times T\times H\times3\times3}
```

и его normalization scale. В replay-режиме forward сохраняет локальные `R/I/O`, которые уже нужны Paraplex-пути, а для gated-step также selector weights, но не сохраняет `C_{L-1}` и scale данного шага.

На backward отсутствующий предыдущий carrier восстанавливается двумя способами:

- если текущий FP32-carrier ещё доступен, обратимый локальный Tria-фактор позволяет сделать аналитический reverse-step;
- для bf16/fp16 или если текущий carrier уже освобождён отдельный кернел `depth_replay` восстанавливает нужное состояние, повторяя записанные Tria-факторы от seed текущего сегмента.

Replay не меняет forward-уравнение. Градиенты по-прежнему вычисляются для `R`, `I`, `O`, предыдущего carrier и девяти slot-selector weights. Цена экономии памяти — дополнительная арифметика на backward вместо хранения carrier-sized activation для каждого depth-step.

### Endpoint-only temporal carry

В chunked training обычно требуется конечное temporal-состояние сегмента, а не каждый нормализованный temporal prefix. Поэтому `temporal_carry_endpoint` возвращает только

```math
A_T\in\mathbb{R}^{B\times H\times3\times3}
```

и небольшую FP32-копию endpoint для backward. Вторая полная траектория temporal accumulators не сохраняется. Backward проходит входные depth-carriers в обратном порядке и аналитически восстанавливает предыдущий нормализованный accumulator из обратимого локального фактора и текущего accumulator. Это temporal-аналог replay: входные depth-carriers остаются доступны, но дополнительная prefix-history `[B,T,H,3,3]` не создаётся.

Отдельные fused Tria-кернели обслуживают девятислотовый selector, population pooling и sparse final cross-attention, поэтому эти пути также не собираются как плотные Python-side operator graphs. Для каждого расширения существует PyTorch fallback на случай ошибки компиляции, неподдерживаемого dtype или устройства; fast path выбирается автоматически после runtime-проверок.

Расширения собираются лениво через `torch.utils.cpp_extension.load` и Ninja. Хэши исходников и подключённых headers записываются в `kernels/.hashes.json`; неизменившиеся модули загружаются из `kernels/build/` без повторной сборки. Архитектуру CUDA можно явно задать через `TORCH_CUDA_ARCH_LIST`, `KERNELS_VERBOSE=1` показывает build log, а `KERNELS_DUMP_PTX=1` после изменившейся сборки создаёт отдельные PTX-файлы для анализа.

## Глубина и связность

В референсном чекпойнте десять именованных Transformer-блоков. Дополнительно мной используется неформальный operational count для рассуждений о графе, но это не стандарт глубины модели и не backend-invariant величина.

В предыдущей версии README было указано `~124` operational layers: temporal prefix считался как `ceil(log2(128))=7` стадий. Это число смешивало ассоциативный full-sequence reference scan с активным chunked execution path. В текущем коде:

- full-sequence PyTorch reference может вычислять temporal carry удваивающимися offset;
- chunked training вычисляет один streaming endpoint на сегмент;
- token-by-token inference выполняет одну temporal matrix recurrence на сгенерированный токен;
- сработавшая граница добавляет одну seed-композицию в первом Tria-слое следующего сегмента.

Поэтому temporal dependency depth зависит от execution path: в streaming-варианте она достигает длины сегмента, а не всегда равна семи стадиям. Я больше не указываю `124` как единственную точную глубину текущей реализации.

Статический per-token stack до temporal unroll по моей неформальной конвенции всё ещё содержит `60` block-transformations, `10` depth-build/composition Tria и `27` межслойных selector/reduction/gate операций. Final aggregation, cross-attention, embeddings и LM head добавляют собственные преобразования, когда соответствующий boundary path активен.

Та же проблема относится к единственному точному числу связей. Рёбра causal attention зависят от document masks и длины последовательности; temporal edges — от явных и фиксированных границ; sparse final cross-attention — от валидных boundary. Поэтому связность описывается по осям без непроверенного скаляра: token attention, depth-history attention, same-channel depth-рекурсия Tria, temporal matrix recurrence, boundary refeed и sparse boundary-to-token attention.

## Обучение, SFT и инференс

В репозитории реализованы:

- обучение токенизатора и токенизация сырых корпусов;
- подготовленные `.bin` token streams и streaming TXT/JSONL/Parquet/Arrow datasets;
- pretraining на одном устройстве и self-launched multi-GPU DDP;
- gradient accumulation, activation checkpointing и асинхронный eval;
- полный последовательный eval с loss, bits/token и bits/byte;
- smart resume весов, шага, schedule и позиции данных;
- оптимизаторы AdamW и ATOM;
- supervised fine-tuning с packed examples, loss masks и tool-call templates;
- анализ донорских чекпойнтов и структурный transplant;
- переносимая упаковка `.aio` и интерактивный терминальный чат;
- AOTInductor export в автономный inference package.

При включённой temporal Tria полный вход `Model.forward` обрабатывается внутренними temporal chunks. Token-by-token `step` переносит аналогичные attention KV-caches, фазовые traces Paraplex, endpoint Tria и sparse final-cross-attention keys. Flat batched path используется при выключенной или ablated temporal Tria и не является активным refeed-путём.

Reference PyTorch-реализации задают семантику. Опциональные fused CUDA-kernels ускоряют phase-space projection, PvPowLU, DepthAttn, depth-композицию Tria, slot reduction, temporal carry и final sparse cross-attention.

## Структура репозитория

```text
loomformer.py   модель, данные, pretraining, evaluation и AOT export
tria.py         операторы Tria, depth/temporal carry, readers и final cross-attention
loomsft.py      supervised fine-tuning
loomcloner.py   анализ донора и transplant чекпойнта
loompack.py     pack / inspect / extract для переносимого .aio
loomchat.py     интерактивный терминальный чат для .aio
```

## Быстрый старт

### Установка

Создайте окружение и установите Python-зависимости, нужные выбранному workflow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pyyaml tokenizers pyarrow safetensors transformers jinja2
```

ATOM и custom C++/CUDA extensions являются опциональными путями исполнения. Для них нужны соответствующие модули репозитория и рабочий CUDA toolchain.

### Smoke tests

```bash
python loomformer.py --smoke-test
python loomsft.py --smoke-test
python tria.py
```

### Обучение токенизатора

```bash
python loomformer.py \
  --config cfg/model.yaml \
  --train-tokenizer ./datasets/raw \
  --vocab 32000 \
  --tokenizer-out tokenizer.json
```

### Pretraining

```bash
ATOM_META_SCALE=8 python -u loomformer.py \
  --train \
  --config cfg/model.yaml \
  --checkpoint ./loomformer.pt
```

Если датасет не указан внутри YAML:

```bash
ATOM_META_SCALE=8 python -u loomformer.py \
  --train \
  --config cfg/model.yaml \
  --dataset ./datasets/train \
  --checkpoint ./loomformer.pt
```

Запуск на всех видимых CUDA-устройствах:

```bash
python loomformer.py \
  --train \
  --device cudas \
  --config cfg/model.yaml \
  --checkpoint ./loomformer.pt
```

Встроенный detached-режим позволяет не оборачивать команду в отдельный `nohup`:

```bash
ATOM_META_SCALE=8 python loomformer.py \
  --train \
  --config cfg/model.yaml \
  --checkpoint ./loomformer.pt \
  --quiet
```

### Оценка

```bash
python loomformer.py \
  --eval \
  --checkpoint ./loomformer.pt \
  --dataset ./datasets/val \
  --eval-data-cache ram
```

### Минимальный инференс из чекпойнта

```bash
python loomformer.py \
  --infer \
  --checkpoint ./loomformer.pt \
  --prompt "The carrier matrix" \
  --max-new 128
```

### Supervised fine-tuning

```bash
python loomsft.py \
  --config cfg/sft.yaml \
  --sft-dataset ./datasets/sft \
  --init-checkpoint ./loomformer.pt \
  --checkpoint ./loomformer-sft.pt
```

### Упаковка и чат

```bash
python loompack.py pack ./loomformer-sft.pt \
  --tokenizer tokenizer.json \
  --template chat_template.jinja \
  --quant bf16 \
  -o loomformer.aio

python loomchat.py loomformer.aio --device cuda:0
```

### Анализ донорской модели

```bash
python loomcloner.py \
  --scan ./donor-model \
  --out ./cfg/donor.yaml
```

Клонирование является структурным transplant, а не точным превращением одной архитектуры в другую. Совместимые тензоры remap-ятся; специфичные для LoomFormer параметры Paraplex, DepthAttn и Tria сохраняют собственную инициализацию, если mapping явно не задаёт другое назначение.

## Конфигурация

Модель настраивается через YAML. Основные группы:

| Область | Характерные поля |
| --- | --- |
| Форма | `model_dim`, `n_q_heads`, `n_kv_heads`, `gqa_group_size`, `hidden`, `layers` |
| Attention | `attn_impl`, `attn_sdpa_compute_dtype`, `attn_sdpa_value_fusion`, `rope_*` |
| DepthAttn | `depth_attn_readout`, `depth_attn_qkv_rms`, `residual_branch_rms_cap` |
| Paraplex | `phase_sectors`, `activation`, `powlu_m`, `phase_grad_mode`, `phase_grad_floor`, `paraplex_gate_proj` |
| Tria | `tria_carry_enabled`, `tria_temporal_enabled`, `tria_temporal_window`, `tria_carrier_alpha`, calibration fields |
| Данные | `dataset_format`, `text_field`, `seq_len`, `batch_size`, `prefetch_batches` |
| Обучение | `steps`, `lr`, `optimizer`, `weight_decay`, `grad_clip`, `grad_accum_steps`, `warmup_steps` |
| Runtime | `device`, `amp_dtype`, `grad_checkpointing`, `graph`, `save_graph`, CUDA fast-path flags |

Основные shape-инварианты проверяются при применении конфигурации:

- `model_dim` должен делиться на число query-heads;
- число GQA-heads и group size должны быть согласованы;
- hidden-width Paraplex должен делиться на число query-heads;
- temporal Tria требует положительных window и carrier coefficient;
- входная длина не может превышать `seq_len`.

Startup calibration Tria может выбрать temporal window и carrier coefficient из набора кандидатов по порогам condition number, effective rank и population pass.

## Требования

Основные зависимости:

- Python 3.10 или новее;
- PyTorch;
- NumPy;
- PyYAML;
- Hugging Face `tokenizers`.

Зависимости отдельных workflow:

- PyArrow для Arrow- и Parquet-датасетов;
- Jinja2 для chat templates;
- Safetensors и Transformers для donor workflows;
- C++/CUDA toolchain для fused kernels;
- ATOM при выборе `optimizer: atom`.

Reference paths поддерживают CPU. CUDA необходима для практического обучения моделей референсного размера и fused custom operators.

## Ограничения конструкции

LoomFormer намеренно сохраняет causal attention. Поэтому полный token-mixing путь остаётся квадратичным по длине последовательности. Tria не заявляется заменой attention; она добавляет operator-route по глубине и выбранным temporal boundaries.

Архитектура несёт больше состояния, чем минимальный decoder Transformer: KV-cache, depth history, per-layer phase traces, depth-carriers, temporal carriers и sparse boundary keys. Parity нужно проверять между переходами активного chunked forward и token-by-token `step`; отдельный flat path используется, когда temporal refeed неактивен.

Локальная матрица Tria мала, но существует для каждого токена и hidden-канала Paraplex. Fused paths устраняют несколько крупных промежуточных materializations, однако Tria всё равно добавляет вычисления и memory traffic. Её пользу нужно проверять ablation-экспериментами, held-out loss, поведением генерации и профилированием.

Начальный loss LoomFormer нельзя корректно сравнивать шаг-в-шаг с vanilla pre-norm Transformer без контроля normalization placement, residual-маршрута DepthAttn, активации Paraplex, инициализации и токенизатора. Референсный запуск сошёлся, но один запуск не доказывает scaling behaviour всех конфигураций.

В этой ревизии ни одно скалярное число operational depth или connections не считается инвариантом. Текущий execution graph зависит от temporal backend и от реально сработавших границ. Обычное описание архитектуры остаётся таким: 10 decoder-блоков, два depth-read на блок, causal GQA, Paraplex FFN и опциональный Tria carry.

## Статус

LoomFormer обучается, оценивается, дообучается, упаковывается и генерирует текст. 113M-запуск сошёлся на указанном корпусе; результаты полного eval приведены выше. SFT и LoomChat работают. LoomChat пока относительно требователен к VRAM, потому что текущая реализация ставит эквивалентность архитектурных путей и возможность инспекции выше минимального inference-runtime.

Проект остаётся экспериментальным. Имена конфигурации, миграции чекпойнтов, CUDA-kernels и metadata пакета пока не являются стабильным публичным API.

## Цитирование

До появления статьи или архивного релиза можно ссылаться на репозиторий и коллекцию чекпойнтов:

```bibtex
@software{loomformer_paraplex,
  author = {srose69},
  title  = {LoomFormer-Paraplex},
  year   = {2026},
  url    = {https://github.com/srose69/LoomFormer-Paraplex}
}
```
