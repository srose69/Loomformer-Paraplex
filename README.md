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
 
(it converged! eval_loss 3.2438 bits/tok 4.6798 bpb 1.3228 on 120k steps from FSS1STR)

SFT too slow for now, workin' on it, 
BUT PT works! Enjoy 
