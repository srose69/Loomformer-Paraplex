# LoomFormer-Paraplex

LoomFormer: a Transformer-based LM built from Paraplex neurons.

Paraplex = the pseudo-complex paravector neuron (scalar+vector, Clifford Cl(0,n)
paravector, behaves like a complex number). 

LoomFormer = the architecture: normal causal GQA attention + Paraplex FFN + DepthAttn (AttnRes-style softmax-over-depth skip). 
"Loom" for the pseudo-complex & pseudo paravector numbers; 
"-former" for the... dunno. for the style.

Tria.. too difficult to explain, but it works and useful! 
Read the code tho

Soon 

Checkpoints here
https://huggingface.co/srs6901/LoomFormer-Paraplex/
 
(it converged! full_eval_loss 3.2584  bits/tok 4.7009  bpb 1.3288  eval_tokens 45040704 on 150k steps from FSS1STR at 27.3k tok/s on single L40S, budget 4,608,000,000 tokens over 120,000 steps (0.98 epochs of 4,684,230,241), 112,999,621 params · 40.8 tok/param  ·  41.5 data-tok/param)

## Depth

**Configuration:** 10 blocks, `d_model=768`, 8 heads × 96 dim, GQA-4, hidden size 3072, `seq_len=1024`, temporal window Tria 128 (8 chunks), bf16. ~113M parameters (untied embeddings).

**If my calculations are correct**, the network has **~124 true layers** (ResNet calculation method — each non-linear parameterized transformation is counted as one layer):

| Axis | Layers |
|---|---|
| Depth: 6 ops/block × 10 blocks (DepthAttn×2, causal attn, LayerNorm×2, Paraplex FFN) | 60 |
| Tria Carrier: 10 compositions of 3×3 matrix with normalization | 10 |
| Tria Gate: selector + identity gate + slot mix, depth feedback across 9 blocks | 27 |
| Temporal: Hillis-Steele-style prefix (streaming in kernels), ⌈log₂128⌉ = 7 stages | 7 |
| Temporal Refeed: seed + gate at 7 chunk boundaries | 14 |
| Tria Aggregation + final cross-attention | 4 |
| Embedding + LM head | 2 |
| **Total** | **~124** |

**Connectivity:** ~543K unique connections across 8 axes (depth DAG 210, causal attention ~526K, temporal scan ~448, final CA ~8K, others). The graph is **dynamic** — refeed boundaries shift with each step, chunks are re-segmented, and the temporal state flows between chunks.

For context: ResNet-101 has 101 layers and ~1K skip connections. GPT-2 small has ~24 true layers. LoomFormer-nano: more depth than ResNet-101, orders of magnitude more connections, at 113M parameters.

SFT and LoomChat works too! (last is vram-heavy for now but manageable even on my GTX1080, workin' on it) 
Enjoy 
