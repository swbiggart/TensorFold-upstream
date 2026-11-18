# Third-party notices

## Kimi K3 and K3.1-Pro release fixtures and native text adapter

The configuration fixtures in `zig/src/families/kimi_k3/fixtures/` are Moonshot AI's release files, Copyright (c) 2026
Moonshot AI. `k3-config.json` comes from Kimi K3 at revision `f831ab66814297da540d832a5235f8e904f29d06`, under the
[Kimi K3 License](LICENSES/Kimi-K3.txt). `k31-config.json` comes from Kimi K3.1 Pro at revision
`6d0a0202956796543364d666ab19c6ad1a494ee1`, under the [Kimi K3.1-Pro License](LICENSES/Kimi-K3.1-Pro.txt). These are
the models' release licenses, including their additional conditions, not the TensorFold Apache-2.0 license.

The native XTML rendering and argument helpers in `zig/src/server/kimi_prompt.zig`
and `kimi_values.zig` adapt that release's `encoding_k3.py`; native tokenizer
integration follows its `tokenization_kimi.py`. The release-rendered prompt
fixtures in `zig/src/server/fixtures/kimi/golden.json` retain their revision and
source hashes. These adaptations retain the same copyright and license notice.

The GLM-OCR HIP operators in `zig/kernels/hip/glm_ocr/pointwise.hip`
adapt reduction ordering and vectorized LayerNorm statistics from PyTorch
revision `8514f05`, `aten/src/ATen/native/cuda/Reduce.cuh` and
`aten/src/ATen/native/cuda/layer_norm_kernel.cu`. These are maintained
TensorFold adaptations under the BSD-3-Clause license, Copyright (c) 2016-
Facebook, Inc. and the other holders listed in
[`LICENSES/PyTorch-BSD-3.txt`](LICENSES/PyTorch-BSD-3.txt).

The optional native attention adapter in `zig/src/hip/attention_aotriton.zig`
uses the AOTriton 0.11.1 version-1 forward ABI from upstream revision
`d34f3b6c824df77d5c5788a2e7555b2398be4b79`, under the MIT license.
The declarations follow the original headers; this adapter is maintained
by TensorFold. The AOTriton library and architecture images are external
runtime dependencies. See [`LICENSES/AOTriton-MIT.txt`](LICENSES/AOTriton-MIT.txt).

TensorFold uses [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm), MIT License, Copyright © 2023 Apple Inc.
The native engine installs and links neither. It reads checkpoints in their formats, and its Metal kernels under
`zig/kernels/metal/` repeat the arithmetic of MLX's kernels and mlx-lm's model math, many of them bit for bit.
The Python engine, frozen at 0.6.6, installs MLX and mlx-lm as dependencies.

## Nemotron 3.5 Parse on HIP

The reference tool `tools/zig/nemotron_parse_reference.py` adapts `run_transformers_inference.py` from NVIDIA's
[nvidia/Nemotron-3.5-Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse) at revision
`4035ffb1d8f8bafcdb867dfaf6f7514f101464f9`, and keeps that file's SPDX header. The native family follows the same
release's code for its math, rounding and resizing; each file is written for TensorFold, and the file named is the
reference:

- `zig/src/families/nemotron_parse/processor.zig`: `NemotronParseImageProcessor` in `hf_nemotron_parse_processor.py`
  (the aspect-preserving resize, the NaViT white padding, the normalization and the patch order).
- `zig/src/families/nemotron_parse/tower.zig` and `decoder.zig`: `RadioWithNeck`, `_VisionTransformer`,
  `NemotronParseDecoder`, `_DecoderLayer`, `_GQAAttention` and `_ScaledEmbedding` in `hf_nemotron_parse_modeling.py`.
- `zig/kernels/hip/ops/vl_position.hpp`: `_normalized_2d_position_pages` in `hf_prefix_geometry.py`.

These release files are Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, under the NVIDIA OpenMDW License 1.1; see
[the license text](LICENSES/NVIDIA-OpenMDW-1.1.txt).

## LightOnOCR-3 release fixtures

The configuration fixtures for LightOnOCR-3 are LightOn's release files, unchanged, under the Apache License 2.0; see
[the license text](LICENSES/Apache-2.0.txt). `zig/src/families/lighton_ocr/fixtures/config.json` and
`processor_config.json` come from `lightonai/LightOnOCR-3-1B` at revision `23d544085c886bbdace647bcfab91203299417e2`.
`zig/src/families/qwen3_5/fixtures/lighton-ocr-0.8b.json` is the `config.json` of `lightonai/LightOnOCR-3-0.8B` at
revision `36d8636925327c671bd8e479c517a004f7566bd9`, and `lighton-ocr-4b.json` the `config.json` of
`lightonai/LightOnOCR-3-4B` at revision `b71010f095dc3735de7ee6b9969bff8f8f50b39a`.

## MLX and mlx-lm adaptations

In the Python engine, frozen at 0.6.6, the DeltaNet implementations in
`src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py` and
`lane_tree.py` adapt mlx-lm's `qwen3_5` and `gated_delta` model math and kernels under its MIT License.
The native engine carries that adaptation. `zig/kernels/metal/qwen3_5/gdn_pre.metal`, `gdn_chain.metal` and
`gdn_post.metal` are generated from the Python engine's dense Qwen kernels, whose recurrence step is `lane_tree.py`'s.
`zig/kernels/metal/core/deltanet.metal` and Flash Next's `zig/kernels/metal/flashnext/q4_gdn_*.metal` run the same
model math.

## Vulkan affine projections

`zig/kernels/vulkan/qmm.comp` adapts the GLSL shader from
[mlx-omarchy](https://github.com/joshuaswarren/omarchy-mlx), MIT License,
Copyright © 2026 Joshua Warren and mlx-omarchy contributors.
Its arithmetic is unchanged; the native Vulkan path does not link MLX.
See [the license text](LICENSES/mlx-omarchy-MIT.txt).

## Qwen Flash Next

The n-gram ID helpers translate Hugging Face transformers' `models/qwen4_exp/modeling_qwen4_exp.py` with renamed
identifiers. The native engine's are in Zig and Metal, in `zig/src/families/flashnext/config.zig` and the
`fz_ple_ids` kernel of `zig/src/families/flashnext/replay.zig`. The Python engine's, frozen at 0.6.6, are in MLX and
NumPy, in `src/tensorfold/families/qwen4_exp/model.py` and `cuda/ngram.py`.
Copyright 2026 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0.
See [the license text](LICENSES/Apache-2.0.txt). The same helpers appear in mlx-vlm's
`models/qwen4_exp/language.py`, MIT License, Copyright © 2025 Prince Canuma.

The native engine's image and video input for Flash Next follows transformers' `get_rope_index` positions in
`models/qwen4_exp/modeling_qwen4_exp.py` (`zig/src/families/flashnext/vis_prompt.zig`) and its Qwen3-VL video
processor and processor in `models/qwen3_vl/` (frame sampling, `smart_resize`, pair times and the prompt layout;
`zig/src/core/qwen_vl/video.zig`, `zig/src/server/images.zig`). Copyright 2025 and 2026 The Qwen Team and The
HuggingFace Inc. team, Apache License 2.0; see [the license text](LICENSES/Apache-2.0.txt).

## GLM-5.3-Flash on Apple Silicon

The native GLM-5.3-Flash engine (`zig/src/families/glm/`) runs the Python `glm5_next` family's decode kernels.
`tools/zig/gen_glm_kernels.py` generates `zig/kernels/metal/glm/` from that family's Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/` at the `v0.6.6` tag), which @chadhurley25075-png wrote in
[#9](https://github.com/ashhart/TensorFold/pull/9) ([credits](https://github.com/ashhart/TensorFold/blob/main/docs/credits.md)).
`zig/kernels/metal/glm_kda_prompt.metal` extends the generated KDA kernel to prompt chunks.
In the Python engine, frozen at 0.6.6, the MLX engine of `glm5_next`
(`src/tensorfold/families/glm5_next/`: the forward pass in `model.py`, `kda.py`,
`mla.py` and `mlp.py`, the draft head in `mtp.py`, `runtime.py`) and its Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/`) are written for TensorFold. What they follow or port:

- The forward pass follows, op for op on its prefill path, the GLM-5.3-Flash (`glm5_next`)
  implementation added to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by PR #2030 (by Lazarus-931; MIT License,
  Copyright (c) 2025 Prince Canuma), as vendored by [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Nothing is
  imported from either at runtime. The native forward in `zig/src/families/glm/` follows the Python one.
- `kernels/glm/flash/v1/kda.py` is ported from mlx-vlm PR #2105 ("glm5_next: fuse the KDA decode chain into one
  Metal kernel", by avlp12; `mlx_vlm/models/glm5_next/fused_kda.py`; closed without merging, MIT License,
  Copyright (c) 2025 Prince Canuma): the whole KDA decode step in one Metal kernel. TensorFold runs a window of
  rows in order inside the launch, folds the 4-bit `f_b` / `g_b` projections in with MLX's one-row `qmv_quad`
  arithmetic, and keeps its own rounding points. Its precision rules (precise exp, uncontracted sums of squares)
  are also used in `fused.py`, `moe.py` and `hc.py`. The native `zig/kernels/metal/glm/kda_rows*.metal` are
  generated from it.
- `kernels/glm/flash/v1/sparse_attention.py` is mlx-vlm's `indexed_sparse_attention` kernel
  (`mlx_vlm/models/sparse_attention.py`) as extended by mlx-vlm PR #2245 ("Fix GLM-5.3 cached decode batch
  invariance", by raullenchai; closed without merging, MIT License, Copyright (c) 2025 Prince Canuma), adapted to
  TensorFold's single latent cache. The native `zig/kernels/metal/glm/sparse_attention*.metal` are generated from it.
- mlx-vlm PR #2107 (the sparse indexer's incremental decode and a stale-pool fix, by avlp12) needed no code:
  TensorFold's cache already pools once per completed block. Its stale-pool case is pinned by the Python engine's
  `tests/test_glm5_ported_kernels.py`.
- The hyper-connection kernel `_HC_SPLIT` in `kernels/glm/flash/v1/kernels.py`, and the sinkhorn and collapse in
  `hc.py`, repeat the `hc_sinkhorn_collapse` kernel of mlx-vlm's `mlx_vlm/models/deepseek_v4/hyper_connection.py`
  (MIT License, Copyright (c) 2026 Apple Inc.), with its output type set to the input's. The native
  `zig/kernels/metal/glm/hc_expand_10.metal` is generated from `hc.py`, and `zig/kernels/metal/core/hc_boundary.metal`
  runs the same split, Sinkhorn and collapse.
- The 4-bit matvec `_QMV_ROWS` in `kernels.py` is Flash Next's `qmv_rows` with MLX's group-64 scale indexing, and
  the expert kernels (`_EXPERT_GROUP`, `_EXPERT_QMV`) follow Flash Next's `expert_group` / `grouped_gateup`. The
  row kernels in `kernels.py`, `moe.py` and `hc.py` repeat the arithmetic and partitions of MLX 0.32's own kernels (MIT
  License, Copyright © 2023 Apple Inc.): `qmv_fast`, `qmv_quad` and `gather_qmv_fast` (`quantized.h`), `GEMVKernel`
  and `GEMVTKernel` (`gemv.h`) and the `rms_norm` kernels, one row per grid slice with the tiling MLX picks for one
  row, so each row keeps MLX's one-row bits. The native `zig/kernels/metal/glm/qmv*.metal`, `gemv*.metal`,
  `moe_*.metal` and `router.metal` are generated from these.

## Gemma 4 on Apple Silicon

The native Gemma 4 engine (`zig/src/families/gemma4/`) ports the Python engine's dense Gemma 4 decode, which @gilby
wrote in [#360](https://github.com/ashhart/TensorFold/pull/360) on @cshintov's Gemma kernels from
[#10](https://github.com/ashhart/TensorFold/pull/10) ([credits](https://github.com/ashhart/TensorFold/blob/main/docs/credits.md)).
Its reference is gilby's `gemma4-dense` branch at `a6ff795e`: `RowDecode` in `src/tensorfold/kernels/gemma/v1/decode.py`
and the `_EMIT`, `qkv_prep`, `attn_tail` and `dense_tail` kernels and the attention partials and merge in `glue.py` and
`attention.py`. `zig/kernels/metal/gemma4/glue.metal` repeats that arithmetic, and each kernel names its source.

Where that reference calls MLX or mlx-lm, `glue.metal` repeats their kernels' arithmetic (MLX and mlx-lm, MIT License,
Copyright © 2023 Apple Inc.; see [the license text](LICENSES/MIT.txt)): `gemma_rms_looped` is MLX's `rms_looped` from
`mlx/backend/metal/kernels/rms_norm.metal`, the kernel `mx.fast.rms_norm` runs for rows past 4,096, with its thread
counts, reads and rounding points; `gemma_embed` is MLX's affine dequantize followed by a bf16 multiply;
`gemma_geglu` and `gemma_softcap` are `geglu` and `logit_softcap` from mlx-lm's `models/gemma4_text.py` as MLX's
bf16 ops; and `gemma_rope_freqs` is mlx-lm's `ProportionalRoPE` frequencies, computed as MLX's `pow` computes them.

The image tower (`zig/src/families/gemma4/vision.zig`, `zig/kernels/metal/gemma4/vision.metal`) follows
`Gemma4VisionModel` and `Gemma4MultimodalEmbedder` in transformers' `models/gemma4/modeling_gemma4.py`, and
`zig/src/families/gemma4/image.zig` follows `Gemma4ImageProcessorPil` in `image_processing_pil_gemma4.py`
(`get_aspect_ratio_preserving_size`, the patch layout and the rescale). Copyright 2026 the HuggingFace Team, Apache
License 2.0; see [the license text](LICENSES/Apache-2.0.txt). The engine runs none of their code.

The drafter (`zig/src/families/gemma4/dflash.zig`) implements z-lab's DFlash for Gemma 4 31B from the MLX model listed
under [vendored code](#vendored-code-and-weights) (z-lab/dflash, MIT License, Copyright © 2026 Z Lab); the native
engine vendors no copy.

The 12B (`gemma4_unified`) and its MTP assistant (`zig/src/families/gemma4/assistant.zig`, `assistant_weights.zig`)
follow @gilby's `gemma4_unified` family at `6b0ead7a` ([gilby/TensorFold](https://github.com/gilby/TensorFold),
`src/tensorfold/families/gemma4_unified/model.py` and `assistant.py`), which follows `Gemma4UnifiedAssistantForCausalLM`
in transformers. The 12B's image embedder (`zig/src/families/gemma4/vision_unified.zig`) follows
`Gemma4UnifiedVisionEmbedder` in `models/gemma4_unified/modeling_gemma4_unified.py`, and `image.zig`'s merged patches
`patches_merge` in `image_processing_gemma4_unified.py` (transformers, Copyright 2026 the HuggingFace Team, Apache
License 2.0; see [the license text](LICENSES/Apache-2.0.txt)). The engine runs none of their code.

Other widths (`zig/src/families/gemma4/widths.zig` and the stacks in `weights.zig`) follow @gilby's oQ per-layer widths
at `76c83c37` on the `gemma4-dense` branch: each module's MLX affine width from the config, and a q|k|v or gate|up stack
whose members differ split into runs of equal widths (`src/tensorfold/kernels/gemma/v1/matmul.py`). The engine runs none of
that code.

## CUDA

The Python engine's CUDA backends, frozen at 0.6.6, use [PyTorch](https://github.com/pytorch/pytorch), BSD-3-Clause, and
[Triton](https://github.com/triton-lang/triton), MIT, supplied by NVIDIA's container rather than bundled.
Dense Qwen implements mlx-lm's model math. CUDA DFlash2 implementations port z-lab's architecture under
the MIT License, Copyright © 2026 Z Lab.
The native CUDA build links neither PyTorch nor Triton. It runs its own kernels (`zig/kernels/cuda/`), and it loads
Triton-compiled cubins (`zig/src/cuda/triton.zig`) only from a captured set named by `TENSORFOLD_CUDA_KERNELS` or
installed beside the executable.

Flash Next implements transformers' model math under the attribution above. In the Python engine, its new CUDA
DeltaNet kernel follows flash-linear-attention's numerics, MIT; its NCCL wrapper follows vLLM's stream convention,
Apache-2.0, without copying either implementation.

The earlier #927 Drama backbone included a Gated DeltaNet prefill adapted from Flash Linear Attention.
The merged Drama family uses #939/#956's operators instead. The staged BF16 GEMV candidate in
`zig/kernels/cuda/drama3_staged_gemv.cu` and the direct-I/O stager in `zig/src/families/drama3/cuda/stager.zig`
come from plotarmordev's #927 implementation. Their qualification against the merged model is separate from
#927's earlier native baseline receipts.

The Python engine's GLM CUDA engine implements transformers' `models/glm5_next/modular_glm5_next.py` math,
Apache-2.0, without including that source. Its draft inputs and thinking-off rendering follow the public GLM recipe
from Mia-AiLab without including recipe code.
The native CUDA GLM family (`zig/src/families/glm/cuda_*.zig`, its kernels in `zig/kernels/cuda/glm_*.cu`) implements
the same math. Its image tower, on Metal (`zig/src/families/glm/vision.zig`) and on CUDA (`cuda_vision.zig`,
`zig/kernels/cuda/glm_vision.cu`), and the PyTorch reference that checks it (`tools/glm_vision_ref.py`) follow
mlx-vlm's `glm5_next` (`vision.py`, and `processing.py` for the reference's patches; MIT License, Copyright (c) 2025
Prince Canuma) without importing it.

In the Python engine, frozen at 0.6.6, GLM EXL3 (`families/glm5_next/cuda/exl3.py`, `exl3.cu`, `exl3_mm.py`), the
shared EXL3 module (`src/tensorfold/cuda/exl3/`) and the EXL3 loaders of Qwen3.8-27B and Qwen3.8 Flash Next
(`families/qwen3_5/cuda/exl3_load.py`, `families/qwen4_exp/cuda/exl3.py`; the module and the loaders are @vcruz305's,
from [#42](https://github.com/ashhart/TensorFold/pull/42)) read
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s EXL3 format: its trellis layout and bitstream, its
"3inst", "mcg" and "mul1" codebooks, its half-integer bit widths and its tensor-core fragment order. Flash
Next's packs also carry ExLlamaV3's n-gram row codec, read as its `ngram_dequant` reads it. ExLlamaV3 uses the
MIT License, Copyright © 2025 Turboderp. TensorFold's decoders and kernels are separate implementations,
checked bit for bit against ExLlamaV3's dequantization.

The native `zig/src/core/exl3_format.zig` reference decoder, the `zig/src/core/exl3_projection*.zig` decoder and
`zig/kernels/metal/core/exl3_projection.metal` separately implement the same format and codebooks
under the Turboderp attribution above. So do the native checkpoint scan (`zig/src/core/exl3_load.zig`,
`exl3_metal_load.zig`), the CUDA kernels (`zig/kernels/cuda/exl3_*.cu`, `exl3_decode.cuh`, `exl3_candidate.cuh`) and
their Zig side (`zig/src/cuda/exl3*.zig`), and GLM's reader of its EXL3 experts (`zig/src/families/glm/cuda_exl3.zig`).
No captured kernel binaries or model weights are bundled.

The EXLR8 reader, decoder, pull and kernels (`zig/src/core/exlr8*.zig`, `zig/src/cli/pull_exlr8*.zig`,
`zig/src/cuda/exlr8.zig`, `zig/kernels/metal/core/exlr8*.metal`, `zig/kernels/cuda/exlr8*.cu`) read kindlingai's exlr8
v2 format as the Loader section of the [glm-5.3-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4)
model card lays it out. The format's packing, manifests and error tables are kindlingai's, offered under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); its codebook, Hadamard rotation and tile order are
ExLlamaV3's, under the Turboderp attribution above. TensorFold's implementations are its own, checked bit for bit
against the format's reference loader (MIT License, Copyright (c) 2025 Turboderp), which is not included.
`zig/tests/fixtures/exlr8/oracle.json` holds that loader's digests and a few sampled values of the release; no
weights are bundled.

In the Python engine, frozen at 0.6.6, Flash Next's optional int8 and int4 KV caches
(`families/qwen4_exp/cuda/kvcache.py`, by @vcruz305 in [#47](https://github.com/ashhart/TensorFold/pull/47)) follow
the cache quantization scheme of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) `-cq 8` and `-cq 4`
(MIT License, Copyright (c) 2025 Turboderp, text below): groups of 32, one fp16 absmax scale per group, the group
rotated by a 32-point Hadamard, midpoint-grid codes, `compand_a == 0`. 8-bit stores each code as a signed int8
(`q - 128`). 4-bit stores two unsigned codes per byte, low nibble first (the same bits as ExLlamaV3's little-endian
packing, a uint8 tensor rather than their uint32 words). Their dequantizer folds another `1/sqrt(32)` into the scale
and applies the unnormalized butterfly on the way out; this cache applies the normalized H32 to the query and to the
merged output instead, and leaves the stored codes rotated. Scales match their quantizer bit for bit. Reconstructed
values agree within fp16/bf16 rounding (under 0.01 on random groups), not bit for bit. The quantizer and the attention
dequant are written for TensorFold and checked against an independent reference of that arithmetic. The native engine
has no quantized KV cache and refuses `--kv-dtype`.

The two-rank CUDA RoCE gather (`zig/src/cuda/roce.zig`, `zig/kernels/cuda/roce_gather.cu`) adapts
the one-shot all-gather protocol of [b12x](https://github.com/local-inference-lab/b12x)'s RoCEnante
(`b12x/comm/roce/`, PRs #295 and #315): a device-resident epoch, sequence flags after each payload, two slots by
sequence parity and a timeout that fails the transport. Copyright 2026 Luke Alonso and the b12x contributors,
Apache License 2.0 (see [the license text](LICENSES/Apache-2.0.txt)); b12x ships no NOTICE file.

The Zig tensor-parallel fabric's RoCE path (`zig/src/fabric/tp/mailbox.cu`, `roce.zig`, `roce_verbs.zig`) adapts
the same one-shot all-gather protocol of b12x's RoCEnante (`b12x/comm/roce/`, PRs #295 and #315) for any family that
runs tensor parallelism over it, under the same copyright and Apache License 2.0 terms.

## Cloudflare Clef

`zig/src/server/clef_encode.zig` and `clef_answer.zig`, the host head in `zig/src/core/joint_schema/`
(`reference.zig`, `reference_attn.zig`), the device heads in `zig/src/hip/joint_schema/` and
`zig/src/cuda/joint_schema/` with their kernels in `zig/kernels/hip/joint_schema/joint_schema.hip` and
`zig/kernels/cuda/joint_schema.cu` port Cloudflare's Clef release code, `joint_schema_model.py` in
[Cloudflare/clef](https://huggingface.co/Cloudflare/clef) and
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash): the record rendering and encoding
(`render`, `question_options`, `encode_record`), the verbatim system prompt, the joint schema head
(`JointSchemaHead`, `EvidenceRoutingLayer`) and the SystemOne answer shape (`systemone_answer`). The encoder's first
port is gilby's. Copyright Cloudflare, Inc., Apache License 2.0; see [the license text](LICENSES/Apache-2.0.txt).
`tools/clef_head_golden.py`, `tools/clef_tokens.py` and `tools/clef_head_gate.py` run that file:
`zig/src/core/joint_schema/fixtures/clef_tiny.json` holds a generated head's outputs (no Cloudflare weights), and
`zig/src/server/fixtures/clef_tokens.json` the Clef-Flash tokenizer's ids for synthetic records.

## Vendored code and weights

In the Python engine, frozen at 0.6.6, `src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is the unmodified
`dflash/model_mlx.py` from
[z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab.
The native engine vendors no copy. Its DFlash2 drafter, `zig/src/families/qwen27/dflash/`, implements z-lab's
architecture under that attribution.

In the Python engine, frozen at 0.6.6, `src/tensorfold/families/deepseek_v4/vendor/encoding_dsv4.py` is the
unmodified `encoding/encoding_dsv4.py` of
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (revision 60d8d70), and
`tests/fixtures/deepseek_v4/` holds two of its test cases, MIT License, Copyright (c) 2023 DeepSeek.
The MTP layer that engine drafts with comes from that checkpoint's last shard (MIT), converted by
`families/deepseek_v4/convert.py`. The native engine has no DeepSeek-V4 family and ships none of these files.

`zig/kernels/cuda/ggml_iq2xxs_tables.cuh` holds [ggml](https://github.com/ggml-org/ggml)'s IQ2_XXS grid and sign
tables as `ggml-common.h` defines them, MIT License, Copyright (c) 2023-2026 The ggml authors (the license text is
below); `tools/zig/bolt_ggml.py` reads them from that file. The IQ2_XXS and Q8_0 kernels that use them for Nemotron 3.5
Bolt's GGML checkpoints (`zig/kernels/cuda/ggml.cu`, `zig/src/cuda/ggml.zig`) are TensorFold's own.

TensorFold ships no model weights. The `z-lab/Qwen3.8-27B-DFlash2` model card states Apache-2.0.
The optional `incoai/GLM-5.3-Flash-DFlash2` model card states CC BY-NC-ND 4.0, for non-commercial use
without derivatives; only the Python engine reads that drafter. Each checkpoint keeps its own license.

## Structured output: libtfgrammar

`lib/libtfgrammar` in the release archives is `zig/src/grammar/tf_grammar.cc` compiled with the C++ sources of
[xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8, Apache License 2.0, Copyright (c) 2024 by XGrammar Contributors
(its `NOTICE`: "XGrammar Copyright (c) 2024 by XGrammar Contributors"), and the headers it bundles:
[dlpack](https://github.com/dmlc/dlpack), Apache License 2.0, and [picojson](https://github.com/kazuho/picojson),
BSD 2-Clause License, Copyright 2009-2010 Cybozu Labs, Inc. and 2011-2014 Kazuho Oku. Nothing of theirs is in this
repository; the build reads a checkout given with `-Dxgrammar=DIR`, and each archive carries xgrammar's and dlpack's
license and xgrammar's notice in `LICENSES/` (`xgrammar-Apache-2.0.txt`, `xgrammar-NOTICE.txt`,
`dlpack-Apache-2.0.txt`, and picojson's as `picojson-BSD-2-Clause.txt`), also copied into `lib/` beside the
library. The Inkling-2-Small TP4 release assets carry an aarch64 build of it in `lib/libtfgrammar.so`. picojson's license:

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
   disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Cortheon

`--cortheon` (`zig/src/server/cortheon/`) adapts [Cortheon](https://github.com/ashhart/Cortheon)'s task-classification word lists,
its stage sequences and their purposes, its rules for when a task needs outside sources, and its instruction-filter patterns,
rewritten in Zig. MIT License, Copyright (c) 2026 Cortheon contributors; the license text is below.

## Model release and library code these files follow

These follow a model's release code or a library's model code for their math, rounding or resizing. Each is written for TensorFold; the file named is the reference.

- `tools/lfm3/ref64.py` and `zig/kernels/xpu/lfm3.cl`: `modeling_lfm3.py` in [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B). Copyright Liquid AI, LFM Open License v1.0; see [the license text](LICENSES/LFM-Open-1.0.txt).
- `zig/kernels/metal/minimax_m3/glue.metal`: the rounding points of `modeling_minimax_m3_vl.py` in [transformers](https://github.com/huggingface/transformers).
- `zig/src/vision/qwen_cuda.zig`: `Qwen3_5VisionModel` in transformers.
- `zig/src/core/picture/patches_test.zig`: recorded image-processor output data from `Lfm3VlImageProcessor` and `Qwen2VLImageProcessor` in transformers; the processing implementation is described below.
- `zig/src/families/flashnext/rotary_config.zig`: `_compute_yarn_parameters` in transformers, for the 64 rotary dims.

transformers is Copyright The HuggingFace Team, Apache License 2.0; see [the license text](LICENSES/Apache-2.0.txt).

## Image patch preparation

`zig/src/core/picture/patches.zig` implements the image budgets and geometry described by the
[Qwen2-VL model card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct),
[Qwen2.5-VL model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and
[LFM2-VL model card](https://huggingface.co/LiquidAI/LFM2-VL-450M).
Its discrete sizing, patch ordering, normalization and cache identities are pinned by TensorFold's own tests and
black-box processing outputs. The implementation was rewritten from those behavior sources; the recorded comparison
data in `patches_test.zig` retains its provenance above.

## MIT License text

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Flash Next CUDA image integration

This notice covers the Flash Next image input on CUDA that @shantanugoel added to the Python engine in
[#146](https://github.com/ashhart/TensorFold/pull/146), under `src/tensorfold/vision/` and
`src/tensorfold/families/qwen4_exp/cuda/`; that engine is frozen at 0.6.6. The native engine ships none of this code.
The multimodal rotary and image-feature integration is adapted from MiaAI-Lab's
[Flash Next vision patch 0008](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/a3aa89835022c55ca8e55008c37785954834e04f/patches/0008-flash-next-vision.patch),
MIT License, Copyright (c) 2026 MiaAI-Lab. The license is included in `LICENSES/MiaAI-Lab-MIT.txt`.
The port preserves the v0.5 CUDA execution APIs and adds an offline EXL3 vision adapter.

## Pictures in the native engine

`zig/src/core/picture/` reads PNG, JPEG and WebP pictures to the RGB pixels Pillow loads them as (Pillow 12.3 for
TensorFold 0.6.6's PNG and JPEG, Pillow 13.0.0 for WebP), and the families' processors resize them. No libjpeg-turbo,
libwebp, Pillow or PyTorch code is included.

- JPEG pictures decode in libjpeg-turbo 3.1.4.1's TurboJPEG library (`libturbojpeg`), the release Pillow 12.3
  bundles, loaded at run time and called with Pillow's decoder settings. TensorFold neither includes nor ships it;
  libjpeg-turbo is under the IJG, BSD-3-Clause and zlib licenses.
- WebP pictures decode in libwebp 1.6.0's decoder library (`libwebpdecoder`), the release Pillow 13.0.0 bundles,
  loaded at run time. TensorFold neither includes nor ships it; libwebp is under the BSD-3-Clause license, Copyright
  (c) 2010, Google Inc.
- PNG pictures decode from the PNG specification on Zig's zlib inflater, and each layout becomes 8-bit samples as
  Pillow's PNG modes hold it. The bicubic resizes repeat, bit for bit, the fixed-point arithmetic of Pillow's
  `Image.resize` (MIT-CMU License, Copyright © 1997-2011 by Secret Labs AB, Copyright © 1995-2011 by Fredrik Lundh
  and contributors, Copyright © 2010 by Jeffrey 'Alex' Clark and contributors; the license is in
  `LICENSES/Pillow-MIT-CMU.txt`) and of PyTorch's antialiased uint8 bicubic (BSD-3-Clause, Copyright (c) 2016-
  Facebook, Inc. and the other holders listed in `LICENSES/PyTorch-BSD-3.txt`).
- `zig/src/families/lfm2/image.zig` ports transformers' `Lfm2VlImageProcessor` and `Lfm2VlProcessor` (smart resize,
  tile grid, placeholder tokens). Apache License 2.0, Copyright 2025 the HuggingFace Inc. team; see
  [the license text](LICENSES/Apache-2.0.txt).

## Intel GPU engine (XPU)

The Intel GPU kernels and host code (`zig/kernels/xpu/`, `zig/src/xpu/`, the `xpu_*` files of the families) are written for
TensorFold. What they follow or copy:

- The Level Zero loader and the Intel compute runtime are loaded at run time and are not bundled; `ocloc` compiles the kernels
  at build time and is not distributed.

## Fish Audio Drama 3 on CUDA

`zig/src/families/drama3/`, `zig/kernels/cuda/drama3*.cu` and `drama3*.cuh` implement the model math of
[SGLang-Omni](https://github.com/sgl-project/sglang-omni)'s `sglang_omni/models/fishaudio_drama3` (revision 416a70c0),
Apache-2.0, without including its source. The codec they decode with is Fish Audio's modded DAC from
[Fish Speech](https://github.com/fishaudio/fish-speech) (Fish Audio Community License Agreement v1.0), itself built on
[descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) (MIT License, Copyright (c) 2023-present,
Descript); `codec.zig` and `drama3_codec.cu` are separate implementations of its arithmetic and include no Fish Speech
or descript-audio-codec source. The samplers draw from the Philox stream of SGLang-Omni's Triton sampling kernels.
`drama3_fp8.cu` is TensorFold's `fp8_experts.cu` with per-row FP8 activation scales, the dynamic activation scheme
SGLang serves the checkpoint's FP8 experts with. The resampler in `wav.zig` follows torchaudio's `functional.resample`
(BSD-2-Clause) without including its source.

TensorFold ships no Drama 3 weights and no codec weights. [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3)
and its `codec.pth` are under the Fish Audio Community License Agreement v1.0: research and non-commercial use,
commercial use below its revenue threshold, and no Hosted Service without a separate license from Fish Audio. Users
obtain the checkpoint and accept that license themselves.

## DeepSeek-V4.1 shared checkpoint forms

The DeepSeek-V4.1 config and checkpoint planning combine @jayleaton’s chain (#727) and
@akol1’s #825 admission. The reference is DeepSeek AI’s MIT licensed inference at revision
2cba9e42aa026125f3ed06c6d98c1db82f7ca027, Copyright (c) 2023 DeepSeek.
EXL3 scanning reuses the common checkpoint scanner and retains its ExLlamaV3 attribution.

The shared Engram host code and token-map builder combine @jayleaton’s #668 and @akol1’s
#824, #825 and #826. Normalization, PCG64 and bucket metadata follow the pinned DeepSeek MIT reference
above. nfkd_data.zig is generated Unicode decomposition data from #824; it is kept verbatim.

## DeepSeek-V4.1 CUDA sources

The CUDA sources and loading combine @jayleaton’s #671, #674 and #727 contributions. Source comments retain
the original TensorFold Python-kernel blob receipts. Generic EXL3 variants extend the one
common EXL3 operators and their per-slot epilogues; tile decoding is shared.
The ExLlamaV3 MIT attribution, Copyright (c) 2025 Turboderp, applies as described above.
No external reference, captured cubin, model shard or CUDA compiler is bundled by this patch.

## Shared EXL3 resolver setup

The shared EXL3 resolver split and CUDA lane option forwarding retain work from PR #728 by @ajensenwaud,
with the common EXL3 and DeepSeek contributor credits preserved.

## MiaAI-Lab Flash Next prompt expert staging

`zig/kernels/cuda/nvfp4_experts_tile.cu` adapts the K32 shared-memory pipeline in patch0014 of
[MiaAI-Lab's Flash Next recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark),
recipe commit e1f564eb915bb5272474b6c1c527aec3b9ed3de6. It credits TensorFold Contributors for the expert
arithmetic and staging design. Copyright (c) 2026 MiaAI-Lab and Copyright (c) 2026 TensorFold contributors.
The complete original MIT notices and permission text are retained in [Mia-FlashNext-MIT.txt](LICENSES/Mia-FlashNext-MIT.txt).

## Ling-3.0-tiny and Ling-3.1-flash

Ling-3.0-tiny and Ling-3.1-flash are MIT licensed by inclusionAI. The host family package
(`zig/src/families/ling/`) is a port of @jschmied's ling-tiny branch (98026d35). TensorFold ships no Ling weights.

## MOSS-Image seeded noise

`zig/src/families/moss_image/noise.zig` is an altered port, to Zig, of the code x86 PyTorch runs for `torch.randn` on a
seeded CPU generator, so that it makes the same latents bit for bit:

- PyTorch's mt19937 engine and `normal_fill` (BSD-3-Clause, Copyright (c) 2016- Facebook, Inc. and the other holders
  listed in `LICENSES/PyTorch-BSD-3.txt`).
- `log256_ps` and `sincos256_ps` from `avx_mathfun.h`, which PyTorch includes (zlib license, Copyright (C) 2012 Giovanni
  Garberoglio, based on Julien Pommier's `sse_mathfun.h`, Copyright (C) 2007; the license is in
  `LICENSES/avx_mathfun-zlib.txt`).

## DeepSeek chat encoder output fixture

`zig/src/server/fixtures/deepseek_v41_encoder/cases.jsonl.gz` contains 328
request/output cases supplied from DeepSeek's encoder, under the MIT license,
Copyright (c) 2026 DeepSeek. The unmodified license is beside the fixture as
[`LICENSE`](zig/src/server/fixtures/deepseek_v41_encoder/LICENSE).
This is test data, not an encoder implementation. TensorFold's renderer was
written from these outputs and its existing code without consulting the encoder's source.

## Thinking-off checkpoint template fixtures

The fixtures in `zig/src/server/fixtures/thinking/` are unmodified release files, used only as host-test inputs:
- `glm53.jinja` and `glm53-tokenizer_config.json` come from
  [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) at revision `da920bb0b9f4`
  (NVIDIA's quantization of zai-org's GLM-5.3-Flash), under the MIT license its model card states. See
  [`LICENSES/MIT.txt`](LICENSES/MIT.txt).
- `lfm2.jinja` and `lfm2-tokenizer_config.json` come from
  [LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) at revision
  `667408d494bde191c2cb9dd4f4549827816b9275`, under the LFM Open License v1.0. See
  [`LICENSES/LFM-Open-1.0.txt`](LICENSES/LFM-Open-1.0.txt).

## Inkling-2-Small

The Inkling-2 family (`zig/src/families/inkling2/`, `zig/kernels/cuda/inkling2_*.cu`, `tools/zig/inkling2_*.py`), its
text layer (`zig/src/server/inkling2_reply.zig`, `chat_inkling2.zig`, `inkling2_text_test.zig`), the chat-template
additions made for it in `zig/src/core/template/` are written for TensorFold: Copyright 2026 Scott Biggart, Apache
License 2.0 (see `NOTICE`). They are part of a modified fork of TensorFold at commit `223da7659`. The recipe that
serves the model (`recipe/inkling2-tp4/`) and the release scripts (`scripts/release/`) are in the inkling2-sparks
repository (<https://github.com/swbiggart/inkling2-sparks>), under the same copyright and license; where the credits
below name a path under those folders, it is a path in that repository.

The outside code these files include is listed first, with its license and copyright. Then come the sources whose
numerics or behaviour they reimplement, the upstream TensorFold code they reuse, the projects credited as references
only, the third-party parts of the release binaries, and the model data.

### Code included from outside projects

**Transformers** ([huggingface/transformers](https://github.com/huggingface/transformers),
`src/transformers/models/inkling2/` and `src/transformers/conversion_mapping.py` at commit `109a7d19`), Apache
License 2.0 (see [the license text](LICENSES/Apache-2.0.txt)). `models/inkling2/` is Copyright 2026 the HuggingFace
Team; `conversion_mapping.py` is Copyright (C) 2025 the HuggingFace Inc. team. Transformers ships no NOTICE file.
These parts are adapted from it, modified for TensorFold, and carry that copyright and license where they appear:

- `tools/zig/inkling2_ref.py` writes out the `inkling2_text` weight renames and the gate/up de-interleave (the
  `Interleave` converter) of `conversion_mapping.py` (`LAYER_RENAMES`, `deinterleave`) and restates the decoder loop
  of `Inkling2TextModel.forward` (`masks`, `run_layer`, `finish`). Its `LazyExperts.forward` is
  `Inkling2Experts.forward`, changed to fetch each expert's weights from the checkpoint on first use.
- `tools/zig/inkling2_op_ref.py`: `tau_of` and `dyn_conv` split the attention's log scaling and
  `Inkling2DynamicShortConvolution.forward` into ops.
- `tools/zig/inkling2_moe_check.py`: `exact_sums` is `Inkling2Experts.forward` with fp64 GEMMs.
- `zig/src/core/template/template.zig` (`aliasReasoning`) is `_alias_reasoning_fields` from Transformers'
  `utils/chat_template_utils.py` (Copyright 2024 The HuggingFace Team, Apache License 2.0), rewritten in Zig. As
  `apply_chat_template` does, it copies a message's first non-null `reasoning_content`, `thinking` or `reasoning` into
  `reasoning_content`, and into `thinking` when the message has none. It applies to every chat template, not only
  Inkling-2's.

The other oracle tools (`inkling2_ref_check.py`, `inkling2_attn_check.py`, `inkling2_f64.py`,
`inkling2_margins.py`, and `inkling2_dspark_ref.py`) import Transformers' Inkling-2 modules, masks, cache and
tokenizer at run time. Where they recompute part of the math themselves (the attention in `inkling2_f64.py` and
`inkling2_margins.py`, `Inkling2NormedEmbedding` in `inkling2_dspark_ref.py`), they copy no Transformers code. Transformers itself is not included.

**TokenSpeed** ([lightseekorg/tokenspeed](https://github.com/lightseekorg/tokenspeed)), Copyright (c) 2026 LightSeek
Foundation, MIT License (see [the license text](LICENSES/TokenSpeed-MIT.txt)). TokenSpeed ships no NOTICE file.

- `tools/zig/inkling2_ref.py` ports TokenSpeed's Torch reference for the block-scaled experts
  (`tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/block_scaled.py` at commit `f27ae32d`, the same file as at
  `04ab2e08`: `_fp4_round`, `reference_block_quantize`, `_decode_values`, `reference_capped_swiglu` and the slot fold
  of `torch_block_scaled_moe_routes`). It keeps TokenSpeed's copyright and permission notice above `_ts_fp4_round`.
- The DSpark drafter (`tools/zig/inkling2_dspark_ref.py`, `zig/kernels/cuda/inkling2_dspark.cu`,
  `zig/src/families/inkling2/cuda_dspark.zig`) ports the drafter math of TokenSpeed's native
  Inkling-2 draft (`python/tokenspeed/runtime/models/inkling2_mtp.py`, `execution/drafter/inkling2.py` and
  `execution/drafter/dflash.py`, including `Inkling2MTPHead.refresh_fold` and the bf16 `RotaryEmbedding` cache),
  under the same copyright and license.

**kortexa-ai's TensorFold fork** ([kortexa-ai/TensorFold](https://github.com/kortexa-ai/TensorFold), by francip, commit
`e2614fe`), Apache License 2.0 (see [the license text](LICENSES/Apache-2.0.txt)); its `NOTICE` is TensorFold's own,
reproduced in `NOTICE`. The Inkling-2 attention core is adapted from its `tf_ik_attention` kernel. Each file below keeps
the Apache-2.0 header, names that source and commit, and says that it was modified:

- `zig/kernels/cuda/inkling2_kx_common.cuh` is `zig/kernels/cuda/inkling2_common.cuh` at `e2614fe`, with that header
  added and nothing else changed.
- `zig/kernels/cuda/inkling2_kx_attention.cuh` is `zig/kernels/cuda/inkling2_attention.cuh` at `e2614fe`, with its
  include renamed to `inkling2_kx_common.cuh` and the kernel renamed `tf_ik_attention_e2614fe`. The engine never runs
  it; `tf-inkling2-attn` runs it as a control.
- `zig/kernels/cuda/inkling2_kx_fp32.cuh`, the attention the engine runs, keeps the per-row arithmetic of
  `inkling2_attention.cuh` and `inkling2_attention_decode.cuh` (Q·K on `mma.sync` with fp32 accumulation, fp32 scores
  and online softmax over 64-key tiles, P·V as three bf16 parts of fp32 P, the multiplier form of the attention sink,
  one bf16 rounding at the output) and the relative-bias arithmetic of `inkling2_heads.cuh`. Modified: one tile body
  for every form, with ring and paged keys, the position read on the device for CUDA graphs and scalar kernel
  arguments; keys outside a block's range loaded as zero; subnormal q, K, V elements and P parts flushed per element;
  a row with a non-finite input recomputed alone (never its whole block); NaN outputs stored as one canonical value;
  exception counters; column splits; fixed S-key splits on the global layers and on the sliding-window layers with an
  ordered fp32 merge, and a walk over the splits for long prompt chunks; the query heads that share a key/value head
  packed into one tile; the relative-bias table in one launch.

### Numerics and behaviour followed

These files reimplement another project's math or behaviour. They copy none of its code, and the project is not
included.

- **Transformers** (the copyright and license above). The model math, its rounding points and the tensor layouts
  follow `modeling_inkling2.py` and `configuration_inkling2.py`, and each file names the module it follows:
  `zig/kernels/cuda/inkling2_attn.cu`, `inkling2_conv.cu`, `inkling2_head.cu`, `inkling2_moe.cu` and
  `inkling2_res.cu` (attention and its L2 clips, the short convolutions, the head and softcap, the router and
  experts, the block residuals), the router order in `inkling2_ck.cu`, and `zig/src/families/inkling2/config.zig`
  (the configuration defaults), `roles.zig`, `plan.zig`, `cuda_attn.zig`, `cuda_conv.zig`, `cuda_forward.zig`,
  `cuda_head.zig`, `cuda_moe.zig`, `cuda_ops.zig`, `cuda_res.zig` and `cuda_weights.zig`. The gate/up row interleave
  of the split experts (`plan.zig`) reads the rows as the `Interleave` converter in `conversion_mapping.py` does.
- **TokenSpeed** (the copyright and license above). The checkpoint-mode kernels compute the numerics of its Torch
  reference, one rounding per Torch op: `zig/kernels/cuda/inkling2_ck.cu`, `zig/src/families/inkling2/cuda_moe.zig`
  (checkpoint mode) and `quant.zig`. The slot order of the fold (`tokenspeed_slot_order` in
  `tools/zig/inkling2_ref.py`, after `torch_block_scaled_topk`) and the fp32 unembed output of checkpoint mode
  (`cuda_head.zig`) follow TokenSpeed. The gate/up interleave in `plan.zig` also matches TokenSpeed's
  `gate_up[:, 0::2]` read. The quantization manifest parser in `quant.zig` follows the checks of TokenSpeed's
  `Inkling2QuantConfig` (`python/tokenspeed/runtime/layers/quantization/inkling2.py`): the group keys it accepts and
  the FP32, BF16, NVFP4 and MXFP8 group settings. The drafter and
  vision tensor names in `roles.zig` follow TokenSpeed's `inkling2_mtp.py` and `inkling2_vision.py`.
  `tools/zig/inkling2_ref_check.py --tokenspeed TS_DIR` runs TokenSpeed's own `block_scaled.py` from a checkout the
  user supplies, unmodified (its unused imports stubbed); it is not copied or shipped.
- **kortexa-ai's TensorFold fork** (the copyright and license above). The fp32 attention reference and checks follow
  the arithmetic of `tf_ik_attention` at `e2614fe` and copy none of its code: `tools/zig/inkling2_ref.py`,
  `inkling2_op_ref.py` and `inkling2_margins.py` (`--attn fp32`, in PyTorch); `zig/src/core/relbias_attention_ref.zig`
  (float64 ranges of its fast path and its per-row path, and of the split form); the argument layout of the control
  launch in `zig/src/families/inkling2/cuda_attn.zig` (`InklingAttention`).
- **Four Over Six** ([mit-han-lab/fouroversix](https://github.com/mit-han-lab/fouroversix); J. Cook, J. Guo, G. Xiao,
  Y. Lin, S. Han, "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling",
  [arXiv:2512.02010](https://arxiv.org/abs/2512.02010)). The checkpoint's `scale_selection_rule: fouroversix_mae`
  names this method. For each block of 16, the checkpoint-mode NVFP4 activation quantizer (`quant.zig`,
  `inkling2_ck.cu`, `tools/zig/inkling2_ref.py`) picks the scale with the lower reconstruction MAE, between a
  candidate that maps the block maximum to 6 and one that maps it to 4. We use the method as TokenSpeed implements
  it; no Four Over Six code is copied or linked.
- **PyTorch** ([pytorch/pytorch](https://github.com/pytorch/pytorch)), BSD-3-Clause, Copyright (c) 2016- Facebook,
  Inc. and the other holders listed in [the license text](LICENSES/PyTorch-BSD-3.txt). The oracle and its checks
  (`tools/zig/inkling2_*.py`) run on PyTorch, imported at run time. The CUDA kernels (`inkling2_ck.cu`,
  `inkling2_conv.cu`, `inkling2_moe.cu`, with `cuda_moe.zig` and `cuda_ops.zig`) reproduce the rounding PyTorch's
  CUDA ops were measured to apply: one rounding per op, division by a Python scalar as a multiply by its fp32
  reciprocal, and the summation order of `sum(-1)`. That behaviour was matched by testing against PyTorch's outputs.
- **Inkling-2-Small's reply grammar.** The text layer (`zig/src/server/inkling2_reply.zig`, `chat_inkling2.zig` and
  its test `inkling2_text_test.zig`) reads the reply grammar of the checkpoint: the `response_template` in
  `tokenizer_config.json`, the typed-part marker tokens and `chat_template.jinja`, published by Thinking Machines in
  [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) under the Apache
  License 2.0. Its handling of text outside a part follows Transformers' response-template parser
  (`utils/chat_parsing/response_parser.py`, Copyright 2026 The HuggingFace Team, Apache License 2.0). No code or
  template text is copied from either; the test builds its own small template, and the release files are read only
  at run time.
- **Jinja2** ([pallets/jinja](https://github.com/pallets/jinja)), Copyright 2007 Pallets, BSD 3-Clause License. The
  chat-template `float` filter added for Inkling-2 (`zig/src/core/template/filters.zig`) follows the behaviour of
  Jinja2's `do_float` filter: `float(value)`, or the given default (`0.0` if none) where Python raises `TypeError` or
  `ValueError`.
- **Python's `float()` of a string** (`zig/src/core/template/pyfloat.zig`, used by that filter) follows CPython
  3.12's `float()` on a `str` ([python/cpython](https://github.com/python/cpython), Python Software Foundation
  License Version 2): Unicode whitespace and decimal digits read as ASCII, then Python's float-literal grammar, with
  underscores between digits and `inf`/`infinity`/`nan`. Its table of decimal-digit runs (the zero of each run of
  ten `Nd` characters) is data from the [Unicode Character Database](https://www.unicode.org/ucd/) 15.0.0 as Python
  3.12's `unicodedata` reports it; Copyright © 1991-2026 Unicode, Inc., Unicode License V3
  ([`LICENSES/Unicode-3.0.txt`](LICENSES/Unicode-3.0.txt), from <https://www.unicode.org/license.txt>).

### Upstream TensorFold code reused

The family reuses upstream [TensorFold](https://github.com/ashhart/TensorFold) code under the same license (Copyright
2026 TensorFold contributors, Apache License 2.0; see `LICENSE` and `NOTICE`; code from before 0.6.0, including the
shipped `tools/bench_openai.py` and `tools/bench_concurrent.py`, keeps its MIT notice in `LICENSES/MIT.txt`):

- `mma_fp4` in `zig/kernels/cuda/inkling2_ck.cu` is `zig/kernels/cuda/mma4.cuh`'s, and `mma_mx8` is its MXFP8
  counterpart. `inkling2_head.cu` keeps `dense_rows.cu`'s summation order. The lane backend in
  `zig/src/families/inkling2/cuda_lanes.zig` follows `minimax_m3/cuda_lanes.zig` and `glm/cuda_lanes.zig`.
- The family builds on the N-rank collectives and the multirank RoCE wire with its four-rank relay in
  `zig/src/fabric/tp/`, the GLM family's CUDA rank, shard and boundary code, the MiniMax-M3 family's CUDA skeleton and
  DSpark drafter, and `tf-tp-test`. `zig/src/families/inkling2/cuda_wire.zig` borrows
  `glm/cuda_roce_multi.zig`.
- The recipe's NCCL `NCCL_CUMEM_ENABLE=1` and `NCCL_RUNTIME_CONNECT=1` settings follow
  `docs/recipes/glm-5.3-flash-cuda.md`.
- The recipe's RoCE gathers (`TF_INK2_ROCE_MULTI=1`, on by default, through `cuda_wire.zig` with the shipped
  `bin/tp_mailbox.fatbin`; and the opt-in `TF_ROCE=1`) run upstream's RoCE transports (`zig/src/cuda/roce.zig`, `zig/src/fabric/tp/`) unchanged.
  These adapt [b12x](https://github.com/local-inference-lab/b12x)'s RoCEnante one-shot all-gather and are credited
  in the CUDA section above (Copyright 2026 Luke Alonso and the b12x contributors, Apache License 2.0).

### References credited (nothing included)

These shaped the recipe or the design. None of their code, text or files is included, so their licenses place no
conditions on TensorFold; they are credited by name.

- **MiaAI-Lab's four-Spark DeepSeek recipe**
  ([MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold),
  commit `c744e6c7`), Copyright 2026 MiaAI-Lab, Apache License 2.0. The recipe's NCCL settings over RoCE
  (`NCCL_CROSS_NIC`, subnet-aware routing with a subnet for each PCIe half, no P2P or SHM transport, four channels,
  `NCCL_ALGO=Ring`, `NCCL_IB_MERGE_NICS=0` and, on the switchless ring, a per-rank NCCL graph file) follow its
  `scripts/nodes.sh` (`config.example.sh`, `rank.sh`, `fabric/host/`, `fabric/ring-experimental/`,
  `tools/zig/inkling2_rank.sh`). The ring's measurements used per-rank NCCL graph files generated for the same four
  hosts by its `graph_xml`; those files are not included. The checks in `smoke.py` (chat, thinking, a tool-call round
  trip, streamed equals whole, a long prompt) follow its `tools/validate.py`. Two decode ideas follow its engine
  patches: 0019 (graph-capture agreement) for one CUDA graph a decode step on every rank, and 0037 (NVFP4 checkpoint)
  for running the experts on the FP4 tensor cores. That recipe credits its own sources, among them MiaAI-Lab's
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  recipe and webzone's PCIe-twin rails.
- **MiaAI-Lab's switchless-ring guide**
  ([DeepSeek-v4.1-Flash-DGX-Sparks `docs/switchless-ring.md`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/64b02dea8681fb1d8e3a659ea432a8e9e9ba3995/docs/switchless-ring.md),
  Copyright (C) 2026 zurih, AGPL-3.0-or-later). The ring addressing in `recipe/inkling2-tp4/fabric/ring-experimental/`
  follows it: one /24 per cable, `.10` on the originating rank's port, `.11` on the next rank's, and MTU 9000.
  TensorFold extends it to the second PCIe half.
- **NVIDIA Developer Forums** ([forums.developer.nvidia.com](https://forums.developer.nvidia.com/)). The rule to reboot a host after any fabric cable change (`recipe/inkling2-tp4/README.md`,
  `fabric/README.md`, `fabric/switch/README.md`, `fabric/host/README.md`) and the PCIe slot power check in `rank.sh`
  and `status.sh` follow a report by forum user daikw in
  [ConnectX-7 Inter-Spark Link Capped at ~13 Gbps](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461/4):
  after recabling, a link stayed near 13 Gb/s until the host was rebooted with its cables in place. We saw the same
  on our hosts.
- **z-lab DFlash** ([z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab). The DSpark
  drafter's block design (a block of 8 rows, the anchor's embedding followed by mask embeddings, drafted in one
  non-causal pass over the target's hidden-state taps) is DFlash's. It reached TensorFold through TokenSpeed, whose
  `Inkling2DSpark` drafter extends TokenSpeed's own `DFlash` class. The drafter contains no z-lab code.
- **vLLM's Inkling-2-Small recipe** ([vllm-project/vllm](https://github.com/vllm-project/vllm),
  `recipes/models/thinkingmachines/Inkling-2-Small.yaml`, read at commit `43c1db8c`; Copyright contributors to the
  vLLM project, Apache License 2.0; <https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small>) was a design
  reference for the placement (attention split across the ranks by heads, the experts spread over the group:
  `plan.zig`, `cuda_tp.zig`) and for DSpark at the checkpoint's full width of 7 drafts (`cuda_dspark.zig`).
- **Community Inkling-Small recipes for DGX Spark**:
  [0xSero/Inkling-Small-2-Sparks](https://github.com/0xSero/Inkling-Small-2-Sparks),
  [Reederey87/inkling-small-nvfp4-2x-dgx-spark](https://github.com/Reederey87/inkling-small-nvfp4-2x-dgx-spark) and
  [PixelML/Inkling-Small-NVFP4-Dual-DGX-Spark](https://github.com/PixelML/Inkling-Small-NVFP4-Dual-DGX-Spark). Their
  published speeds and findings (fp32 reductions in decode, a CUDA graph for each batch size, the draft block width)
  were planning reference points.
- **0xSero's inkling2-exl3** ([0xSero/inkling2-exl3](https://github.com/0xSero/inkling2-exl3), MIT License). Ideas
  only: `/health`'s KV capacity and admission report (`server/status_routes.zig` `kv`), and the measuring process
  behind `recipe/inkling2-tp4/bench.sh` and `tools/inkling2_bench.py` (a one-command speed sweep with JSON out and a
  check command, a fixed realistic prompt set measured on the server's clock and at temperature 0.7 too, a speed
  ledger against a bandwidth floor, concurrent needle sessions); and FP8 weight-only projections for the DSpark
  drafter, whose MLP projections run from an FP8 copy of the checkpoint's bf16 drafter weights (theirs: e4m3 with a
  scale per row; ours: MXFP8, e4m3 with a power-of-two scale per 32 values, quantized at start-up by the quantizer in
  `zig/src/cuda/mxfp8.zig` and loaded in `zig/src/families/inkling2/cuda_dspark.zig`). No code was copied; the prompts
  (`tools/inkling2_prompts.json`) and the needle text are our own.
- **BertholomusAI's four-Spark TensorFold recipes**
  ([deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10),
  [mistral-large-4-tensorfold-tp4-4xgb10](https://github.com/bertholomus/mistral-large-4-tensorfold-tp4-4xgb10),
  [glm-5.3-tensorfold-tp4-4xgb10](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10); Apache-2.0) were consulted when planning
  the four-host layout of `recipe/inkling2-tp4/`, and [spark-attic](https://github.com/bertholomus/spark-attic)
  (MIT OR Apache-2.0) for GB10 memory headroom.
- **NVIDIA PTX ISA 8.7/8.8 documentation** ([docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/)) **and LLVM's NVPTX backend** ([NVPTXUsage](https://llvm.org/docs/NVPTXUsage.html)). `mma_mx8` in `inkling2_ck.cu` uses the
  `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32 ... ue8m0` instruction form and its
  scale-operand layout as NVIDIA's PTX ISA documentation gives them ("Block Scaling for mma.sync"); LLVM's list of
  valid NVPTX block-scale combinations confirmed which forms exist.
- **perftest** ([linux-rdma/perftest](https://github.com/linux-rdma/perftest), GPL-2.0 or the OpenIB.org BSD
  license). `recipe/inkling2-tp4/rank.sh` (`bw-server`, `bw-client`) and `status.sh --bw` run the host's installed
  `ib_write_bw` and read its output. The raw-RDMA figures in `recipe/inkling2-tp4/fabric/*/README.md` were measured
  with perftest 24.01. Install it from your distribution (the `perftest` package).
- **MikroTik RouterOS** ([mikrotik.com](https://mikrotik.com)). `recipe/inkling2-tp4/fabric/switch/crs812-fabric.rsc` is a RouterOS configuration script
  written for TensorFold, for a MikroTik CRS812-8DS-2DQ-2DDQ on RouterOS 7.20, checked against the switch's own
  configuration export. It contains no MikroTik code. MikroTik and RouterOS are trademarks of SIA Mikrotīkls;
  TensorFold is not affiliated with or endorsed by MikroTik.

### Release binaries: toolchain and NVIDIA components

- **Zig.** The executables in `bin/` of the release archives (`tensorfold-native`, `tf-inkling2-decode`,
  `tf-inkling2-plan`, `tf-inkling2-fold`, `tf-tp-test`) are built with [Zig](https://ziglang.org) 0.17.0 and include compiled code from
  Zig's standard library and its compiler_rt. MIT License (Expat), Copyright (c) Zig contributors; see
  [the license text](LICENSES/Zig-MIT.txt).
- **NVIDIA CUDA Toolkit.** Those executables and `bin/tp_mailbox.fatbin` contain GPU code compiled by NVIDIA's CUDA
  13.3 compiler (`nvcc`) from TensorFold's own sources. That code includes inline functions from the CUDA headers
  (`cuda_bf16.h`, `cuda_fp16.h`, `cuda_fp8.h` and others), Copyright NVIDIA Corporation, distributed in object form as
  the [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) permits; they remain under its terms,
  not under the Apache License. The release bundles no NVIDIA compiler or CUDA runtime; the CUDA driver
  (`libcuda.so.1`) comes from the host. Two NVIDIA libraries are bundled in `lib/`, below.
- **NVIDIA NCCL** (`lib/libnccl.so.2`, version 2.30.7 for CUDA 13, aarch64) is bundled unmodified, the same bytes as
  NVIDIA's `libnccl2_2.30.7-1+cuda13.3_arm64.deb` (sha256 in `lib/SHA256SUMS` of the install). NCCL is Copyright (c)
  2015-2026 NVIDIA CORPORATION & AFFILIATES, mostly under the Apache License 2.0 with parts (the bundled DOCA GPUNetIO
  code) under the BSD 3-Clause license; its `LICENSE.txt` from [NVIDIA/nccl](https://github.com/NVIDIA/nccl) at tag
  `v2.30.7-1` is [LICENSES/NCCL-LICENSE.txt](LICENSES/NCCL-LICENSE.txt) and is also shipped as
  `lib/LICENSE-NCCL.txt`. NCCL is not covered by this repository's Apache License notice for its own work.
- **NVIDIA cuBLAS / cuBLASLt** (`lib/libcublas.so.13` and `lib/libcublasLt.so.13`, version 13.1.1.3, aarch64) are
  bundled unmodified (the same bytes as NVIDIA's `nvidia-cublas` 13.1.1.3 wheel for aarch64). They are NVIDIA
  proprietary software under the [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/), which lists them
  among the redistributable components (Attachment A); they are not covered by the Apache License. The EULA text as
  NVIDIA ships it in that wheel is [LICENSES/NVIDIA-CUDA-EULA.txt](LICENSES/NVIDIA-CUDA-EULA.txt), also shipped as
  `lib/LICENSE-NVIDIA-CUDA-EULA.txt`; the libraries ship with that text and without modification, as it requires.
  Do not repackage them without the EULA text. The container image takes the same files from `lib/`; it is built on each host and never pushed to
  a registry.

### Model data

`zig/src/families/inkling2/fixtures/config.json` and `hf_quant_config.json` are copied unchanged from
[thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (Thinking Machines) at
revision `aac974122acd33ddb0d3e1f54433a9c5dd2249ce`, licensed under the Apache License 2.0 (see
[the license text](LICENSES/Apache-2.0.txt)); `inventory.tsv` lists that revision's safetensors headers. The release
ships no NOTICE file and no copyright line, so there is no further notice to carry. Its model card also links Thinking
Machines' Acceptable Use Policy for the model.

The release tarball holds data derived from that checkpoint at the same revision, under the same license. These files
are computed by TensorFold's tools and are not files from the original repository:

- `plan/ranges-w4-r*.tsv`: byte ranges and tensor names read from the checkpoint's safetensors headers;
  `plan/inkling2-digests-w4.tsv`: sha256 digests of the checkpoint's bytes (`tf-inkling2-plan`).
- `data/chat_short.ids.json`, `data/tool_call.ids.json` and the prompt ids in `data/long_4000.safetensors`:
  TensorFold's own prompts, rendered with the checkpoint's chat template and tokenizer.
- `data/decode.json` and the other tensors in `data/long_4000.safetensors` (hidden states, routes, logits): outputs of
  the model at that revision, computed by `tools/zig/inkling2_ref.py`.

The model weights are not included in TensorFold or its release assets. The DSpark fold file
(`inkling2-dspark-fold.safetensors`, computed from the unembed and the drafter's `hidden_out`) is derived weights;
each user builds it locally from their own copy of the weights with the release's `tf-inkling2-fold` (the recipe's
`setup.sh` and `weights.sh` do this on each host) or with `inkling2_dspark_ref.py fold`, and release assets must not
include it. "Inkling" and
"Thinking Machines" are named only to identify the source; TensorFold is not affiliated with or endorsed by Thinking
Machines.
