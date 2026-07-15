# LoomFormer-Paraplex

LoomFormer: a Transformer LM built from Paraplex neurons.

Paraplex = the pseudo-complex paravector neuron (scalar+vector, Clifford Cl(0,n)
paravector, behaves like a complex number). 

LoomFormer = the architecture: normal causal GQA attention + Paraplex FFN + DepthAttn (AttnRes-style softmax-over-depth skip). 
"Loom" for the pseudo-complex & pseudo paravector numbers; 
"-former" for the... dunno. for the style.

Soon 

Checkpoints here
https://huggingface.co/srs6901/LoomFormer-Paraplex/

SFT too slow for now, workin' on it, 
BUT PT works! Enjoy 
