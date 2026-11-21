// Inkling-2's fp32 attention core (cuda_attn.zig `kx`, the default; TF_INK2_ATTN=v01 runs inkling2_attn.cu's v0.1
// core): kortexa-ai's tile arithmetic (kortexa-ai/TensorFold e2614fe, Apache-2.0) in the forms of
// inkling2_kx_fp32.cuh (kx-tile, kx-dec, kx-split + tf_kx_merge, kx-split-walk; ring and paged keys; S-key splits for
// S in 256 .. 4096), the relative bias table (tf_kx_relative), and e2614fe's own tf_ik_attention unchanged as
// tf_ik_attention_e2614fe (the control of tf-inkling2-attn --limb; never run by the engine). A separate image so its
// bf16 / sigmoid helpers do not meet inkling2_attn.cu's.
#include "inkling2_kx_fp32.cuh"
