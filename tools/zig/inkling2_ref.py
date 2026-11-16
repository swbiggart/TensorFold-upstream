"""Inkling-2-Small's text model streamed a layer at a time through Transformers' own Inkling2 modules: the oracle.

The checkpoint (366 GiB, NVFP4 and MXFP8 experts) fits neither the GPUs nor host memory, so the reference loads one
decoder layer at a time onto one GPU, runs every prompt through it, and keeps only what crosses layers: each prompt's
hidden rows and block-residual states (the inputs of layers 0, 6, 12, ...). The layer itself is Transformers'
`Inkling2SlidingDecoderLayer` / `Inkling2FullDecoderLayer` with the checkpoint's tensors assigned, called exactly as
`Inkling2TextModel.forward` calls it (same masks from `masking_utils`, same block-state bookkeeping, same final
`final_input`, norm, unembed and softcap). The short convolutions and attention need no state across layers in a
teacher-forced pass; in --decode they live in a `DynamicCache` per prompt as in Transformers' own generation.

USAGE = python tools/zig/inkling2_ref.py MODEL_DIR --out OUT_DIR [--mode full|checkpoint] [--layers N]
            [--prompts a,b] [--all-rows] [--fp32-reduction] [--decode 48 --decode-prompts a,b,c,d]

  python tools/zig/inkling2_ref.py MODEL_DIR --out OUT                  # 48 layers, full mode
  python tools/zig/inkling2_ref.py MODEL_DIR --out OUT --mode checkpoint
  python tools/zig/inkling2_ref.py MODEL_DIR --out OUT --layers 6 --prompts chat_short
  python tools/zig/inkling2_ref.py MODEL_DIR --out OUT --prompts long_4000 --all-rows   # every row (layer feeds)
  python tools/zig/inkling2_ref.py MODEL_DIR --out OUT --fp32-reduction   # bf16 GEMMs reduced in fp32 (tight)
  python tools/zig/inkling2_ref.py MODEL_DIR --out OUT --decode 48 \
      --decode-prompts chat_short,code,tool_call,long_1500 --skip-forward                    # greedy text (gate G0)
  python tools/zig/inkling2_ref.py MODEL_DIR --out OUT --mode checkpoint --decode 768 --structured all \
      --grammar-lib LIBTFGRAMMAR --xgrammar-check --cases-out DIR       # grammar-masked greedy (structured output)

Structured output (--structured): six OpenAI chat requests (structured_specs: json_object; a json_schema with nesting,
enums and arrays; tool_choice required; a named function; thinking on with a schema; a schema forcing a long string)
rendered by the chat template, then greedy under the kind-5 grammar the server compiles for each (st_reply / st_tools:
an optional free thinking part, then the text part or the call(s), each closed by <|end_message|>, then a stop). Each
step's pick is the argmax over the tokens the grammar allows; a case ends when its grammar does (after a stop id, kept
as the last token). Masks come from libtfgrammar (--grammar-lib, the server's library, through its C ABI) or from
Python xgrammar 0.2.8 with its compiler cache off; both use tf_grammar.cc's vocabulary (tokenizer.json's model.vocab
and added tokens, text_config.vocab_size wide) and the server's stop ids. --xgrammar-check compares every mask of the
first with the second, byte for byte. Writes OUT_DIR/<mode>/structured.json and, with --cases-out, one <case>.json per
case (request, grammar text, prompt ids, reply ids and their sha256, the masks' sha256).

Modes (what the experts compute; everything else is the same Transformers code in both):
  full        Stored expert weights decoded to bf16 (NVFP4: e2m1 x e4m3 per 16 x amax/outer_scale_denominator per
              expert; MXFP8: e4m3 x 2^(ue8m0-127) per 32), bf16 activations, Transformers' `Inkling2Experts` eager loop
              (each pick's down projection times its weight, summed in fp32 in expert-id order, one bf16 rounding).
              This is what Transformers itself computes: its forward never quantizes activations (its NVFP4 loader is
              not in the release it supports; its MXFP8 path either dequantizes or uses FP8 GEMM kernels).
  checkpoint  Activations quantized as hf_quant_config.json requires, following TokenSpeed's Torch reference
              (`torch_block_scaled_moe_routes`, ported below): NVFP4 per token (outer amax/1536, an e4m3 scale per 16
              chosen between the 6- and 4-target candidates by reconstruction MAE), MXFP8 per 32 (ue8m0 rounded up);
              fp32 GEMMs on the decoded values; capped SwiGLU in fp32 with the routing weight applied before the bf16
              store and the second quantization; route rows bf16, folded in fp32 slot order, one bf16 rounding.
              The router is Transformers' in both modes (same picks, same weights); in checkpoint mode its picks are
              put in TokenSpeed's slot order (descending sigmoid(logit) + bias, ties to the lower id), the order of
              TokenSpeed's fold. --decode reads each routed expert's raw tensors on first use (--expert-cache).

Accumulation order of the expert sum (what a kernel must follow to reproduce a layer's bits):
  full        per row, fp32, the picks in ascending expert id (Transformers' eager loop over experts), weight applied
              in fp32 to the bf16 down projection, one bf16 rounding of the sum
  checkpoint  per row, fp32, the picks in slot order (route.slot_ids), each route row already bf16, one bf16 rounding

Output: OUT_DIR/<mode>/ (with --layers N: OUT_DIR/<mode>-L<N>/)
  manifest.json      checkpoint revision and shard sha256 (the Hub's LFS ids), sha256 of the json/tokenizer files,
                     Transformers version and commit, torch, GPU, mode, dtypes, layers, prompts (messages, rendered
                     text, token ids), row sampling, sha256 of every output file, wall times
  <prompt>.safetensors, one per prompt:
    ids            int64 [T]            prompt token ids
    rows           int64 [R]            rows kept for the hidden states: all rows when T <= 512 or --all-rows, else
                                        the first and last 64, every 16th row, and rows 1020..1031, 2044..2051,
                                        3068..3075
    embed          bf16 [R, 4608]       normed embedding (layer 0's input)
    h.<L>          bf16 [R, 4608]       decoder layer L's output (layer L+1's input; the input of layers 0, 6, ...
                                        is what enters the block residual)
    route.ids.<L>  int32 [T, 8]         layer L's selected experts, ascending id, every row
    route.w.<L>    f32 [T, 8]           their routing weights (route_scale 8 included), same order
    route.slot_ids.<L> int32 [T, 8]     the same picks in the slot order the experts received them (checkpoint mode:
                                        TokenSpeed's order, which its fold follows; full mode: Transformers' router
                                        order, which its expert-id-order sum does not use)
    route.slot_w.<L>   f32 [T, 8]       their routing weights, slot order
    route.logits.<L> f32 [R, 384]       router logits (before sigmoid and bias) of the kept rows
    final          bf16 [R, 4608]       after final_input and the final norm
    logits.rows    int64 [F]            rows with full logits: the last 8, and every 256th row
    logits.full    f32 [F, 200058]      softcapped logits (unpadded vocabulary)
    topk.ids       int32 [T, 64]        top-64 token ids of every row, descending
    topk.vals      f32 [T, 64]          their softcapped logits
  decode.json        with --decode: greedy continuation ids and text per prompt, and the prefill's top-1 against
                     the teacher-forced pass

Logits follow Transformers: the unembed is a bf16 Linear, so each logit is rounded to bf16 before the fp32 softcap
(the checkpoint's manifest declares an fp32 unembed output, which TokenSpeed keeps). One bf16 step of a logit near 30
is 0.125, so the same row computed in a GEMM of another shape (the decode's single row against the teacher-forced
chunk) can differ by up to that much; hidden rows and routes are bit-equal between those paths.

Transformers must contain models/inkling2 (main from commit 109a7d19, November 2026; the 5.22 release does not).
Adapted and ported parts keep their notices. Transformers (Apache-2.0) is imported at run time, and parts of it are
adapted here, each marked with its notice: the weight renames and gate/up de-interleave (`LAYER_RENAMES`,
`deinterleave`), `LazyExperts.forward`, and the decoder loop (`masks`, `run_layer`, `finish`). The checkpoint-mode
expert numerics are ported from TokenSpeed (MIT, notice above `_ts_fp4_round`).
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

BF = torch.bfloat16
CK = "model.llm."
GLOBAL_STATIC = {  # config fields that Transformers' Inkling2 hardcodes; the checkpoint must agree
    "q_bias": False, "o_bias": False, "pre_norm_pre_affine": False, "final_norm_pre_affine": False,
    "use_value_norm_no_gamma": True, "attn_res_normalize_values": True, "n_shared_experts": 0, "use_gate_bias": True,
}

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ----------------------------------------------------------------------------------------------- checkpoint access


class Source:
    """A safetensors folder read by name, shards opened lazily; whole tensors or slices on the first axis."""

    def __init__(self, folder: str):
        self.folder = folder
        self.where = json.load(open(os.path.join(folder, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def _f(self, name: str):
        shard = self.where[name]
        if shard not in self.files:
            self.files[shard] = safe_open(os.path.join(self.folder, shard), framework="pt", device="cpu")
        return self.files[shard]

    def get(self, name: str, device="cpu") -> torch.Tensor:
        return self._f(name).get_tensor(name).to(device)

    def rows(self, name: str, a: int, b: int, device="cpu") -> torch.Tensor:
        return self._f(name).get_slice(name)[a:b].to(device)

    def names(self, prefix: str) -> list[str]:
        return sorted(n for n in self.where if n.startswith(prefix))


def text_config(folder: str, layers: int = 0):
    from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig

    raw = json.load(open(os.path.join(folder, "config.json")))["text_config"]
    for k, v in GLOBAL_STATIC.items():
        if raw.get(k, v) != v:
            raise SystemExit(f"config {k}={raw[k]}: Transformers' Inkling2 hardcodes {v}")
    if raw["swa_head_dim"] != raw["head_dim"] or raw["swa_num_attention_heads"] != raw["num_attention_heads"]:
        raise SystemExit("sliding and global attention shapes differ")
    n = raw["num_hidden_layers"]
    local = set(raw["local_layer_ids"])
    types = ["hybrid_sliding" if i in local else "hybrid" for i in range(n)]
    if types != ["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(n)]:
        raise SystemExit("local_layer_ids is not every layer but 5, 11, ...")
    keep = ["vocab_size", "unpadded_vocab_size", "hidden_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "swa_num_attention_heads", "swa_num_key_value_heads", "swa_head_dim", "sliding_window_size",
            "d_rel", "rel_extent", "log_scaling_n_floor", "log_scaling_alpha", "rms_norm_eps",
            "qk_l2_clip_threshold", "rel_l2_clip_threshold", "n_layers_per_attn_res_block", "mlp_gate_softcap",
            "mlp_up_softcap", "final_logit_softcapping", "n_routed_experts", "num_experts_per_tok", "route_scale"]
    kw = {k: raw[k] for k in keep}
    kw.update(conv_kernel_size=raw["sconv_kernel_size"], moe_intermediate_size=raw["intermediate_size"],
              max_position_embeddings=raw["model_max_length"], num_hidden_layers=layers or n,
              layer_types=types[: layers or n], eos_token_id=200006)
    cfg = Inkling2TextConfig(**kw)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    return cfg


def keep_fp32(full_name: str) -> bool:
    """Transformers' `_keep_in_fp32_modules_strict` test (a substring search, as `from_pretrained` does)."""
    from transformers.models.inkling2.modeling_inkling2 import Inkling2PreTrainedModel

    return any(p in full_name for p in Inkling2PreTrainedModel._keep_in_fp32_modules_strict)


# checkpoint leaf -> Transformers leaf inside a decoder layer (the "inkling2_text" conversion mapping, written out).
# LAYER_RENAMES and deinterleave: adapted from huggingface/transformers src/transformers/conversion_mapping.py at
# 109a7d19 (its "inkling2_text" WeightRenaming entries and Interleave(dim=1) converter), Copyright (C) 2025 the
# HuggingFace Inc. team, Apache License 2.0; modified for TensorFold (see THIRD_PARTY_NOTICES.md).
LAYER_RENAMES = [
    (r"^attn\.wq_du\.weight$", "self_attn.q_proj.weight"),
    (r"^attn\.wk_dv\.weight$", "self_attn.k_proj.weight"),
    (r"^attn\.wv_dv\.weight$", "self_attn.v_proj.weight"),
    (r"^attn\.wr_du\.weight$", "self_attn.r_proj.weight"),
    (r"^attn\.wo_ud\.weight$", "self_attn.o_proj.weight"),
    (r"^attn\.gate_attn\.weight$", "self_attn.gate_proj.weight"),
    (r"^attn\.attn_sink_proj\.weight$", "self_attn.sink_proj.weight"),
    (r"^attn\.(q_l2_clip|rel_l2_clip)\.scale$", r"self_attn.\1.scale"),
    (r"^attn\.rel_logits_proj\.proj$", "self_attn.rel_logits_proj.proj"),
    (r"^attn\.(k|v)_sconv\.weight$", r"self_attn.\1_sconv.conv1d.weight"),
    (r"^attn\.(k|v)_sconv\.static_w$", r"self_attn.\1_sconv.static_weight"),
    (r"^attn\.(k|v)_sconv\.weight_proj$", r"self_attn.\1_sconv.weight_proj.weight"),
    (r"^attn_sconv\.weight$", "post_attention_convolution.conv1d.weight"),
    (r"^mlp_sconv\.weight$", "post_mlp_convolution.conv1d.weight"),
    (r"^attn_norm\.weight$", "input_layernorm.weight"),
    (r"^(q|k|v)_input_norm\.weight$", r"\1_input_layernorm.weight"),
    (r"^mlp_norm\.weight$", "post_attention_layernorm.weight"),
    (r"^mlp\.gate\.weight$", "mlp.gate.weight"),
    (r"^mlp\.gate\.bias$", "mlp.gate.e_score_correction_bias"),
]


def hf_leaf(leaf: str) -> str:
    for pat, rep in LAYER_RENAMES:
        if re.match(pat, leaf):
            return re.sub(pat, rep, leaf)
    raise KeyError(f"no Transformers name for layer tensor {leaf}")


# ----------------------------------------------------------------------------------------------- expert weights

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_PAIRS = {}


def nvfp4_values(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 [..., n, k/2] (earlier column in the low nibble), e4m3 per 16 [..., n, k/16]: fp32 [..., n, k]."""
    dev = packed.device
    if dev not in _PAIRS:
        t = torch.tensor(_E2M1, dtype=torch.float32)
        _PAIRS[dev] = torch.stack([t.repeat(16), t.repeat_interleave(16)], -1).to(dev)  # byte -> (low, high)
    v = _PAIRS[dev][packed.long()].flatten(-2)
    return (v.unflatten(-1, (-1, 16)) * scale.float()[..., None]).flatten(-2)


def mxfp8_values(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """e4m3 [..., n, k], ue8m0 per 32 [..., n, k/32]: fp32 [..., n, k]."""
    if (scale == 255).any():
        raise ValueError("ue8m0 NaN scale in the checkpoint")
    s = torch.exp2(scale.float() - 127.0)
    return (w.float().unflatten(-1, (-1, 32)) * s[..., None]).flatten(-2)


def expert_format(src: Source, prefix: str) -> str:
    return "nvfp4" if prefix + "w13_weight_amax" in src.where else "mxfp8"


def dequant_rows(src: Source, prefix: str, leaf: str, a: int, b: int, dev) -> torch.Tensor:
    """Experts a..b of `leaf` (w13 or w2) decoded to bf16 [b-a, rows, k] in the checkpoint's row order."""
    w = src.rows(f"{prefix}{leaf}_weight", a, b, dev)
    s = src.rows(f"{prefix}{leaf}_weight_scale", a, b, dev)
    if expert_format(src, prefix) == "nvfp4":
        amax = src.rows(f"{prefix}{leaf}_weight_amax", a, b, dev).float()
        den = src.get(f"{prefix}{leaf}_weight_outer_scale_denominator", dev).float()
        return (nvfp4_values(w, s) * (amax / den[0])[:, None, None]).to(BF)
    return mxfp8_values(w, s).to(BF)


def deinterleave(w13: torch.Tensor) -> torch.Tensor:
    """Checkpoint rows gate0, up0, gate1, up1, ... -> Transformers' [gate; up] (its Interleave(dim=1) converter)."""
    return torch.cat([w13[:, 0::2], w13[:, 1::2]], dim=1)


def dense_experts(src: Source, prefix: str, n: int, dev, chunk=16):
    """Transformers' `gate_up_proj` [E, 2I, H] and `down_proj` [E, H, I] in bf16."""
    gu, dn = [], []
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        gu.append(deinterleave(dequant_rows(src, prefix, "w13", a, b, dev)))
        dn.append(dequant_rows(src, prefix, "w2", a, b, dev))
    return torch.cat(gu), torch.cat(dn)


# ------------------------------------------- checkpoint mode: TokenSpeed's Torch reference for the quantized experts
#
# Ported from TokenSpeed (github.com/lightseekorg/tokenspeed, commit f27ae32d, the same file as at 04ab2e08,
# tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/block_scaled.py), under its MIT licence:
#
#   Copyright (c) 2026 LightSeek Foundation
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#   to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#   the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#   THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#   IN THE SOFTWARE.


def _ts_fp4_round(x: torch.Tensor):
    magnitude = x.abs()
    code = ((magnitude > 0.25).to(torch.int32) + (magnitude >= 0.75).to(torch.int32)
            + (magnitude > 1.25).to(torch.int32) + (magnitude >= 1.75).to(torch.int32)
            + (magnitude > 2.5).to(torch.int32) + (magnitude >= 3.5).to(torch.int32)
            + (magnitude > 5.0).to(torch.int32))
    table = x.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    sign = torch.signbit(x)
    rounded = torch.where(sign, -table[code.long()], table[code.long()])
    return (code | (sign.to(torch.int32) << 3)).to(torch.uint8), rounded


def _ts_reference_block_quantize(x: torch.Tensor, quant_format: str):
    """Rows of bf16 x quantized per 16 (NVFP4, two-candidate scale) or per 32 (MXFP8): (values, scales, row amax)."""
    if x.ndim != 2 or x.dtype != BF or not x.is_contiguous():
        raise ValueError("Block quantization requires a contiguous BF16 matrix")
    group = 16 if quant_format == "nvfp4" else 32
    rows, width = x.shape
    grouped = x.float().view(rows, width // group, group)
    maxima = grouped.abs().amax(dim=-1)
    if quant_format == "mxfp8":
        raw = maxima * (1.0 / 448.0)
        exponent = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
        exponent = torch.where(raw <= 2.0**-127, 0, exponent.clamp(max=254))
        exponent = torch.where(raw.isnan(), 255, exponent)
        scale = torch.exp2(exponent.float() - 127.0)
        scale = torch.where(exponent == 255, float("nan"), scale)
        inverse = torch.where(exponent == 0, 2.0**127, scale.reciprocal())
        inverse = torch.where(exponent == 255, 0.0, inverse)
        values = (grouped * inverse[..., None]).clamp(-448.0, 448.0)
        return values.reshape_as(x).to(torch.float8_e4m3fn), exponent.to(torch.uint8), None
    amax = x.float().abs().amax(dim=1)
    encoder = (torch.full_like(amax, 1536.0) / amax).clamp(max=torch.finfo(torch.float32).max)
    decoder = encoder.reciprocal()
    outer = amax * (1.0 / 1536.0)
    codes, scales, errors = [], [], []
    for target in (6.0, 4.0):
        scale = (maxima * (encoder[:, None] * (1.0 / target))).clamp(max=448.0)
        scale = scale.to(torch.float8_e4m3fn).float()
        inverse = (scale * decoder[:, None]).reciprocal().clamp(max=torch.finfo(torch.float32).max)
        code, rounded = _ts_fp4_round(grouped * inverse[..., None])
        reconstructed = (rounded * scale[..., None]) * outer[:, None, None]
        error = (grouped - reconstructed).abs()
        for _ in range(4):
            error = error.reshape(*error.shape[:-1], error.shape[-1] // 2, 2)
            error = error[..., 0] + error[..., 1]
        codes.append(code)
        scales.append(scale)
        errors.append(error.squeeze(-1))
    choose4 = errors[1] < errors[0]
    code = torch.where(choose4[..., None], codes[1], codes[0]).reshape(rows, width)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    scale = torch.where(choose4, scales[1], scales[0]).to(torch.float8_e4m3fn)
    return packed, scale, amax


def _ts_decode_values(values: torch.Tensor, scales: torch.Tensor, quant_format: str) -> torch.Tensor:
    if quant_format == "nvfp4":
        codes = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2)
        table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=values.device)
        data = table[(codes & 7).long()]
        data = torch.where((codes & 8) != 0, -data, data)
        return data * scales.float().repeat_interleave(16, dim=-1)
    exponent = scales.float()
    scale = torch.exp2(exponent - 127.0)
    scale = torch.where(scales == 255, float("nan"), scale)
    return values.float() * scale.repeat_interleave(32, dim=-1)


def _ts_capped_swiglu(gate_up: torch.Tensor, gamma: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    gate, up = gate_up[:, 0::2].float(), gate_up[:, 1::2].float()
    sigmoid = torch.sigmoid(gate)
    if gate_cap > 0:
        gate = gate_cap * torch.tanh(gate / gate_cap)
    if up_cap > 0:
        up = up_cap * torch.tanh(up / up_cap)
    return (((gate * sigmoid) * up) * gamma[:, None]).to(BF)


def _ts_block_scaled_moe_routes(x, fetch, weights, ids, formats, gate_cap, up_cap):
    """TokenSpeed's `torch_block_scaled_moe_routes` with fold_slots=True and no EP: bf16 [T, H].

    `fetch(e)` gives expert e's (w13, s13, a13, d13, w2, s2, a2, d2): a13/a2 its fp32 amax (0-d, as `a13[expert]`),
    d13/d2 the [1] denominators, so the raw experts can stay on disk. The loop visits the routed experts in ascending
    id where TokenSpeed's visits all of them and skips the unrouted; each expert writes only its own route rows, so
    the result is the same (inkling2_ref_check.py `experts` compares it with TokenSpeed's function bit for bit)."""
    first, first_scale, first_amax = _ts_reference_block_quantize(x, formats[0])
    decoded_x = _ts_decode_values(first, first_scale, formats[0])
    routes, hidden = ids.numel(), x.shape[1]
    output = torch.zeros((routes, hidden), dtype=BF, device=x.device)
    for expert in ids.unique().tolist():
        locations = (ids.flatten() == expert).nonzero().flatten()
        token_ids = locations // ids.shape[1]
        w13, s13, a13, d13, w2, s2, a2, d2 = fetch(expert)
        weight = _ts_decode_values(w13, s13, formats[0])
        gate_up = decoded_x[token_ids] @ weight.T
        if formats[0] == "nvfp4":
            scale = (first_amax[token_ids] * a13) * (1536.0 * d13[0]).reciprocal()
            gate_up = gate_up * scale[:, None]
        activated = _ts_capped_swiglu(gate_up.to(BF), weights.flatten()[locations], gate_cap, up_cap)
        values, scales, amax = _ts_reference_block_quantize(activated.contiguous(), formats[1])
        weight = _ts_decode_values(w2, s2, formats[1])
        down = _ts_decode_values(values, scales, formats[1]) @ weight.T
        if formats[1] == "nvfp4":
            scale = (amax * a2) * (1536.0 * d2[0]).reciprocal()
            down = down * scale[:, None]
        output[locations] = down.to(BF)
    rows = output.view(x.shape[0], ids.shape[1], hidden)
    folded = torch.zeros((x.shape[0], hidden), dtype=torch.float32, device=x.device)
    for slot in range(ids.shape[1]):
        folded += rows[:, slot].float()
    return folded.to(BF)


# ---------------------------------------------------------------------------------- expert modules (drop-in)


class CheckpointExperts(torch.nn.Module):
    """In place of `Inkling2Experts`: the checkpoint's own bytes, activations quantized (TokenSpeed's reference).

    Without `lru` the layer's raw expert tensors are resident on `dev`; with one (--decode), each routed expert's raw
    tensors are read on first use into that shared LRU, as `LazyExperts` does for the bf16 decode."""

    def __init__(self, src: Source, prefix: str, cfg, dev, lru: "ExpertLRU | None" = None):
        super().__init__()
        self.src, self.prefix, self.lru = src, prefix, lru
        self.fmt = expert_format(src, prefix)
        self.names = ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]
        self.den = [None, None]
        if self.fmt == "nvfp4":
            self.names += ["w13_weight_amax", "w2_weight_amax"]
            self.den = [src.get(f"{prefix}{p}_weight_outer_scale_denominator", dev).float() for p in ("w13", "w2")]
        self.res = None if lru is not None else {n: src.get(prefix + n, dev) for n in self.names}
        self.caps = (cfg.mlp_gate_softcap, cfg.mlp_up_softcap)

    def fetch(self, e: int, dev):
        if self.res is not None:
            t = {n: v[e] for n, v in self.res.items()}
        else:
            key = (self.prefix, e)
            t = self.lru.get(key)
            if t is None:
                t = self.lru.put(key, {n: self.src.rows(self.prefix + n, e, e + 1, self.lru.dev)[0]
                                       for n in self.names})
            t = {n: v.to(dev) for n, v in t.items()}
        a = [t[f"{p}_weight_amax"].float() if self.fmt == "nvfp4" else None for p in ("w13", "w2")]
        d = [x.to(dev) if x is not None else None for x in self.den]
        return t["w13_weight"], t["w13_weight_scale"], a[0], d[0], t["w2_weight"], t["w2_weight_scale"], a[1], d[1]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        dev = hidden_states.device
        return _ts_block_scaled_moe_routes(hidden_states.contiguous(), lambda e: self.fetch(e, dev),
                                           top_k_weights.float().contiguous(), top_k_index.contiguous(),
                                           (self.fmt, self.fmt), *self.caps)


def tokenspeed_slot_order(gate):
    """Checkpoint mode: the router's picks put in TokenSpeed's slot order, which its fp32 fold follows.

    Transformers' router returns `topk(sorted=False)`, an order CUDA does not define; TokenSpeed's router
    (`torch_block_scaled_topk`) returns the picks by descending sigmoid(logit) + bias, ties to the lower id. Same
    picks and weights, reordered; the full mode's eager loop sums in expert-id order and does not depend on it."""
    inner = gate.forward

    def forward(hidden_states):
        logits, w, idx = inner(hidden_states)
        by_id = idx.argsort(-1)
        idx, w = idx.gather(-1, by_id), w.gather(-1, by_id)
        score = (torch.sigmoid(logits) + gate.e_score_correction_bias).gather(-1, idx)
        order = score.argsort(dim=-1, descending=True, stable=True)
        return logits, w.gather(-1, order), idx.gather(-1, order)

    gate.forward = forward
    return gate


# LazyExperts.forward: adapted from huggingface/transformers src/transformers/models/inkling2/modeling_inkling2.py
# (Inkling2Experts.forward) at 109a7d19, Copyright 2026 the HuggingFace Team, Apache License 2.0; modified for
# TensorFold: each expert's weights are fetched on first use from an LRU, dequantized from the checkpoint, and the
# gate is applied inline (see THIRD_PARTY_NOTICES.md).
class LazyExperts(torch.nn.Module):
    """In place of `Inkling2Experts` for decoding: its forward line for line, each expert's bf16 weights fetched on
    first use (the same decode as `dense_experts`) into a shared LRU on `cache_dev`."""

    def __init__(self, src: Source, prefix: str, cfg, act_fn, lru: "ExpertLRU"):
        super().__init__()
        self.src, self.prefix, self.lru = src, prefix, lru
        self.num_experts = cfg.n_routed_experts
        self.act_fn = act_fn

    def weights(self, e: int):
        key = (self.prefix, e)
        hit = self.lru.get(key)
        if hit is None:
            gu = deinterleave(dequant_rows(self.src, self.prefix, "w13", e, e + 1, self.lru.dev))[0]
            dn = dequant_rows(self.src, self.prefix, "w2", e, e + 1, self.lru.dev)[0]
            hit = self.lru.put(key, (gu, dn))
        return hit

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            gu, dn = self.weights(int(expert_idx))
            gu, dn = gu.to(hidden_states.device), dn.to(hidden_states.device)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, gu).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate, up)
            current_hidden_states = F.linear(current_hidden_states, dn)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states.to(hidden_states.dtype)


class ExpertLRU:
    def __init__(self, dev, capacity: int):
        self.dev, self.capacity, self.d = dev, capacity, OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        v = self.d.get(key)
        if v is not None:
            self.d.move_to_end(key)
            self.hits += 1
        return v

    def put(self, key, v):
        self.misses += 1
        self.d[key] = v
        while len(self.d) > self.capacity:
            self.d.popitem(last=False)
        return v


# ----------------------------------------------------------------------------------------------- the model


class Model:
    """The checkpoint's text model as Transformers modules, built a layer at a time."""

    def __init__(self, folder: str, cfg, dev, mode: str):
        from transformers.models.inkling2 import modeling_inkling2 as M

        self.M, self.src, self.cfg, self.dev, self.mode = M, Source(folder), cfg, dev, mode

    def _assign(self, module, sd: dict, prefix: str):
        """Checkpoint tensors in their Transformers dtype (bf16, or fp32 where `from_pretrained` keeps fp32)."""
        out = {}
        for k, v in sd.items():
            want = torch.float32 if keep_fp32(prefix + k) else BF
            out[k] = v.to(self.dev).to(want)
        module.load_state_dict(out, strict=True, assign=True)
        for n, t in list(module.named_parameters()) + list(module.named_buffers()):
            if t.device.type == "meta":
                raise RuntimeError(f"{prefix}{n} was not loaded")
        return module.eval()

    def embed(self):
        M, cfg, src = self.M, self.cfg, self.src
        with torch.device("meta"):
            m = M.Inkling2NormedEmbedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id, cfg.rms_norm_eps)
        sd = {"weight": src.get(CK + "embed.weight"), "embed_norm.weight": src.get(CK + "embed_norm.weight")}
        return self._assign(m, sd, "model.embed_tokens.")

    def head(self):
        """(final_input, norm, lm_head, softcap) of `Inkling2TextModel` / `Inkling2ForCausalLM`."""
        M, cfg, src = self.M, self.cfg, self.src
        with torch.device("meta"):
            fi = M.Inkling2BlockAttentionResidual(cfg)
            norm = M.Inkling2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            lm = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        fi = self._assign(fi, {"query": src.get(CK + "attn_res.output_query")}, "model.final_input.")
        norm = self._assign(norm, {"weight": src.get(CK + "norm.weight")}, "model.norm.")
        lm = self._assign(lm, {"weight": src.get(CK + "unembed.weight")}, "lm_head.")
        return fi, norm, lm, M.Inkling2Softcap(cfg.final_logit_softcapping)

    def layer(self, i: int, experts: str = "dense", lru: ExpertLRU | None = None):
        """Decoder layer i; full mode: experts "dense" (all 384 decoded) or "lazy" (decoded on use); checkpoint mode:
        the checkpoint's bytes, resident or ("lazy") read on use."""
        M, cfg, src = self.M, self.cfg, self.src
        sliding = cfg.layer_types[i] == "hybrid_sliding"
        with torch.device("meta"):
            layer = (M.Inkling2SlidingDecoderLayer if sliding else M.Inkling2FullDecoderLayer)(cfg, i)
        p = f"{CK}layers.{i}."
        sd = {}
        for name in src.names(p):
            leaf = name[len(p):]
            if leaf.startswith("mlp.experts."):
                continue
            sd[hf_leaf(leaf)] = src.get(name)
        if i > 0:
            sd["attention_input.query"] = src.get(f"{CK}attn_res.attn_queries.{i - 1}")
        sd["mlp_input.query"] = src.get(f"{CK}attn_res.mlp_queries.{i}")
        ep = p + "mlp.experts."
        if self.mode == "checkpoint":
            layer.mlp.experts = CheckpointExperts(src, ep, cfg, self.dev, lru if experts == "lazy" else None)
        elif experts == "lazy":
            layer.mlp.experts = LazyExperts(src, ep, cfg, M.Inkling2SoftcappedSwiGLU(cfg), lru)
        else:
            sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"] = dense_experts(
                src, ep, cfg.n_routed_experts, self.dev)
        layer = self._assign(layer, sd, f"model.layers.{i}.")
        if self.mode == "checkpoint":
            tokenspeed_slot_order(layer.mlp.gate)
        return layer


# masks, run_layer and finish: adapted from huggingface/transformers
# src/transformers/models/inkling2/modeling_inkling2.py (Inkling2TextModel.forward) at 109a7d19, Copyright 2026 the
# HuggingFace Team, Apache License 2.0; modified for TensorFold: the loop split into per-layer steps over saved
# prompt states (see THIRD_PARTY_NOTICES.md).
def masks(cfg, embeds, position_ids, cache=None):
    from transformers.masking_utils import (create_causal_mask, create_recurrent_attention_mask,
                                            create_sliding_window_causal_mask)

    kw = dict(config=cfg, inputs_embeds=embeds, attention_mask=None, past_key_values=cache, position_ids=position_ids)
    m = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw),
         "linear_attention": create_recurrent_attention_mask(**kw)}
    if embeds.shape[1] > 1 and (m["full_attention"] is None or m["sliding_attention"] is None):
        raise RuntimeError("eager attention needs explicit masks")
    return m


def run_layer(layer, st: dict, cache=None):
    """One `Inkling2TextModel.forward` loop step for one prompt state {h, blocks, masks, pos}."""
    new = layer(st["h"], st["blocks"], attention_mask=st["masks"][layer.attention_type],
                conv_mask=st["masks"]["linear_attention"], past_key_values=cache, position_ids=st["pos"])
    if layer.is_block_boundary:
        st["blocks"] = torch.cat([st["blocks"].to(st["h"].device), st["h"].unsqueeze(-2)], dim=-2)
    st["h"] = new


def finish(head, st: dict) -> torch.Tensor:
    """Final hidden rows [1, T, H] bf16 (final_input over the block states, then the norm)."""
    fi, norm, _, _ = head
    h = fi(st["blocks"], st["h"])[0]
    return norm(h).to(BF)


def logits_of(head, cfg, hn: torch.Tensor) -> torch.Tensor:
    _, _, lm, cap = head
    return cap(lm(hn).float())[..., : cfg.unpadded_vocab_size]


# ----------------------------------------------------------------------------------------------- prompts


def inventory_text(n_items: int) -> str:
    colors = ["red", "blue", "green", "amber", "violet", "grey", "white", "black", "teal", "orange"]
    things = ["ceramic mug", "steel bolt case", "paper notebook", "glass jar", "copper wire spool", "wool scarf",
              "oak picture frame", "rubber gasket set", "brass hinge pair", "linen tablecloth", "plastic crate"]
    lines = []
    for k in range(1, n_items + 1):
        lines.append(f"Item {k}: a {colors[(k * 7) % len(colors)]} {things[(k * 5) % len(things)]}, "
                     f"weight {100 + (k * 37) % 900} grams, stored on shelf {'ABCDEFGH'[(k * 3) % 8]}{1 + (k * 11) % 9}.")
    return "\n".join(lines)


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["city"]}}}]


def prompt_specs(tok) -> list[dict]:
    """About 8 conversations; the two long ones are sized by tokens (about 1,500 and 4,000)."""
    specs = [
        {"name": "chat_short", "effort": "none",
         "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]},
        {"name": "thinking", "effort": "high",
         "messages": [{"role": "user", "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
                                                   "than the ball. How much does the ball cost? Think it through."}]},
        {"name": "tool_call", "effort": "low", "tools": TOOLS,
         "messages": [{"role": "user", "content": "What's the weather like in Paris right now, in celsius?"}]},
        {"name": "code", "effort": "none",
         "messages": [{"role": "system", "content": "You are a careful Python programmer."},
                      {"role": "user", "content": "Write a Python function fib(n) that returns the n-th Fibonacci "
                                                  "number iteratively, with a docstring."}]},
        {"name": "multi_turn", "effort": "minimal",
         "messages": [{"role": "user", "content": "Name three primary colors."},
                      {"role": "assistant", "content": "Red, yellow and blue."},
                      {"role": "user", "content": "Which of them is warmest?"}]},
        {"name": "tool_result", "effort": "low", "tools": TOOLS,
         "messages": [{"role": "user", "content": "Is it raining in Oslo?"},
                      {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                       "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]},
                      {"role": "tool", "tool_call_id": "call_1",
                       "content": "{\"city\":\"Oslo\",\"condition\":\"light rain\",\"temp_c\":6}"}]},
    ]
    for name, target, ask in (("long_1500", 1500, 9), ("long_4000", 4000, 14)):
        n = 10
        while True:
            content = (f"Below is a warehouse inventory.\n\n{inventory_text(n)}\n\n"
                       f"Question: on which shelf is item {ask} stored, and what is it? Answer briefly.")
            spec = {"name": name, "effort": "none", "messages": [{"role": "user", "content": content}]}
            if len(render(tok, spec)[1]) >= target:
                break
            n += 2
        specs.append(spec)
    return specs


def render(tok, spec: dict):
    kw = {"reasoning_effort": spec["effort"]}
    if spec.get("tools"):
        kw["tools"] = spec["tools"]
    text = tok.apply_chat_template(spec["messages"], tokenize=False, add_generation_prompt=True, **kw)
    ids = tok.apply_chat_template(spec["messages"], tokenize=True, add_generation_prompt=True, return_dict=False, **kw)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return text, list(ids)


def sample_rows(T: int, every: bool = False) -> list[int]:
    if every or T <= 512:
        return list(range(T))
    rows = set(range(64)) | set(range(T - 64, T)) | set(range(0, T, 16))
    for a, b in ((1020, 1032), (2044, 2052), (3068, 3076)):
        rows |= set(r for r in range(a, b) if r < T)
    return sorted(rows)


def logit_rows(T: int) -> list[int]:
    return sorted(set(range(max(0, T - 8), T)) | set(range(0, T, 256)))


# ----------------------------------------------------------------------------------------------- structured output

# Inkling-2's markers (tokenizer.json added tokens) and the server's stop ids (config.json eos 200006 and the extra
# stops of inkling2_reply.zig), passed to the grammar library as its stop tokens.
MESSAGE_MODEL, CONTENT_TEXT, CONTENT_THINKING, END_MESSAGE, INVOKE_TOOL_JSON = 200001, 200004, 200008, 200010, 200049
STOPS = [200006, 199999, 200000, 200002, 200003]
ADDED = range(199998, 200058)  # all 60 added tokens; ids 200058 and up have no text and are never allowed
EXCLUDES = ["<|message_", "<|content_", "<|end", "<|begin_of_text|>", "<|unused"]  # prefixes of all 60 added tokens


def st_token(i: int) -> dict:
    return {"type": "token", "token": i}


def st_body(schema: dict) -> dict:
    """A JSON body: no added token's text in a string or key, as one token or spelled out."""
    return {"type": "json_schema", "json_schema": schema, "max_whitespace_cnt": 32, "excludes": EXCLUDES}


def st_think() -> dict:
    """<|message_model|><|content_thinking|> free text <|end_message|>: any token but a marker."""
    return {"type": "sequence", "elements": [st_token(MESSAGE_MODEL), st_token(CONTENT_THINKING),
                                             {"type": "any_tokens", "exclude_tokens": list(ADDED)},
                                             st_token(END_MESSAGE)]}


def st_reply(schema: dict) -> dict:
    """response_format: [thinking part] <|message_model|><|content_text|> BODY <|end_message|> (then a stop)."""
    return {"type": "structural_tag", "format": {"type": "sequence", "elements": [
        {"type": "optional", "content": st_think()}, st_token(MESSAGE_MODEL), st_token(CONTENT_TEXT), st_body(schema),
        st_token(END_MESSAGE)]}}


def st_call(fn: dict) -> dict:
    """<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>; the arguments'
    properties in sorted key order, as the chat template writes them."""
    params = dict(fn.get("parameters") or {"type": "object"})
    if isinstance(params.get("properties"), dict):
        params["properties"] = {k: params["properties"][k] for k in sorted(params["properties"])}
    schema = {"type": "object", "properties": {"name": {"const": fn["name"]}, "args": params},
              "required": ["name", "args"], "additionalProperties": False}
    return {"type": "sequence", "elements": [st_token(MESSAGE_MODEL), {"type": "const_string", "value": fn["name"]},
                                             st_token(INVOKE_TOOL_JSON), st_body(schema), st_token(END_MESSAGE)]}


def st_tools(fns: list, parallel: bool) -> dict:
    """tool_choice required (a choice of every tool) or a named function (one): [thinking part] then the call(s)."""
    one = st_call(fns[0]) if len(fns) == 1 else {"type": "or", "elements": [st_call(f) for f in fns]}
    calls = {"type": "plus", "content": one} if parallel else one
    return {"type": "structural_tag", "format": {"type": "sequence", "elements": [
        {"type": "optional", "content": st_think()}, calls]}}


def grammar_of(req: dict) -> dict:
    """The kind-5 grammar of an OpenAI chat request's response_format / tool_choice (the cases below use these)."""
    rf, tc = req.get("response_format"), req.get("tool_choice")
    if rf:
        return st_reply({"type": "object"} if rf["type"] == "json_object" else rf["json_schema"]["schema"])
    fns = [t["function"] for t in req["tools"]]
    if isinstance(tc, dict):
        fns = [f for f in fns if f["name"] == tc["function"]["name"]]
    return st_tools(fns, req.get("parallel_tool_calls", True))


SO_TOOLS = TOOLS + [{"type": "function", "function": {
    "name": "get_time", "description": "Current local time in a time zone.",
    "parameters": {"type": "object", "properties": {"zone": {"type": "string", "description": "IANA zone name"}},
                   "required": ["zone"]}}}]
SO_FILM = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
        "genre": {"type": "string", "enum": ["drama", "comedy", "science fiction", "thriller"]},
        "director": {"type": "object", "properties": {"name": {"type": "string"}, "born": {"type": "integer"}},
                     "required": ["name", "born"], "additionalProperties": False},
        "cast": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
    },
    "required": ["title", "year", "genre", "director", "cast"],
    "additionalProperties": False,
}


def structured_specs() -> list[dict]:
    """Grammar-masked cases: an OpenAI chat request each (what a server gate sends; temperature 0) and its prompt.
    so_long_string forces its length with a pattern: xgrammar 0.2.8 drops minLength / maxLength from a string when the
    body has excludes (it warns), and a pattern string is not subject to the excludes, so its class leaves out '<'."""
    def schema(name, s):
        return {"type": "json_schema", "json_schema": {"name": name, "schema": s, "strict": True}}

    reqs = [
        ("so_json_object", {"reasoning_effort": "none", "response_format": {"type": "json_object"}, "messages": [
            {"role": "user", "content": "Describe the planet Mars as a JSON object: its name, its position from the "
                                        "sun and two of its moons."}]}),
        ("so_json_schema", {"reasoning_effort": "none", "response_format": schema("film", SO_FILM), "messages": [
            {"role": "user", "content": "Describe the film Blade Runner (1982)."}]}),
        ("so_tool_required", {"reasoning_effort": "low", "tools": SO_TOOLS, "tool_choice": "required", "messages": [
            {"role": "user", "content": "I'm calling a colleague in Tokyo. What time is it there now?"}]}),
        ("so_tool_named", {"reasoning_effort": "low", "tools": SO_TOOLS, "parallel_tool_calls": False,
                           "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "messages": [
            {"role": "user", "content": "Should I take an umbrella in Oslo today?"}]}),
        ("so_thinking_schema", {"response_format": schema("temperature", {
            "type": "object", "properties": {"celsius": {"type": "number"}, "feels": {"type": "string", "enum": [
                "cold", "mild", "warm", "hot"]}}, "required": ["celsius", "feels"], "additionalProperties": False}),
            "messages": [{"role": "user", "content": "Convert 72 degrees Fahrenheit to Celsius and say how it feels."}]}),
        ("so_long_string", {"reasoning_effort": "none", "response_format": schema("story", {
            "type": "object", "properties": {"story": {"type": "string", "pattern": "^[A-Za-z0-9 ,.;:'!?-]{400,}$"}},
            "required": ["story"],
            "additionalProperties": False}), "messages": [
            {"role": "user", "content": "Write a very short story about a lighthouse keeper."}]}),
    ]
    out = []
    for name, req in reqs:
        req = {"model": "model", "temperature": 0, **req}
        g = json.dumps(grammar_of(req), separators=(",", ":"))
        out.append({"name": name, "effort": req.get("reasoning_effort"), "messages": req["messages"],
                    "tools": served_tools(req), "request": req, "grammar": g})
    return out


def served_tools(req: dict):
    """The tools the server renders into the prompt: a named tool_choice offers only that tool (the server's
    tool_specs.active, every family); otherwise all of them."""
    tools, tc = req.get("tools"), req.get("tool_choice")
    if tools and isinstance(tc, dict) and tc.get("type") == "function":
        tools = [t for t in tools if t["function"]["name"] == tc["function"]["name"]]
    return tools


def plain_request(req: dict) -> dict:
    """A case's request without its constraint: no response_format, tool_choice "auto" over the tools the server
    renders for the case (so the prompt is the same tokens), for the unmasked greedy reply beside the masked one."""
    out = {k: v for k, v in req.items() if k not in ("response_format", "tool_choice", "parallel_tool_calls")}
    if req.get("tools"):
        out["tools"] = served_tools(req)
    return out


def tokenizer_vocab(folder: str, vocab: int):
    """tokenizer.json's text and the encoded vocabulary libtfgrammar builds (tf_grammar.cc tfg_open): model.vocab, then
    the added tokens' content, `vocab` wide ("" where no token is)."""
    raw = open(os.path.join(folder, "tokenizer.json"), "rb").read()
    tj = json.loads(raw)
    enc = [""] * vocab
    for s, i in tj["model"]["vocab"].items():
        if 0 <= i < vocab:
            enc[i] = s
    for t in tj.get("added_tokens", []):
        if 0 <= t["id"] < vocab:
            enc[t["id"]] = t["content"]
    return raw, enc


class TfgMasker:
    """Masks from libtfgrammar (the server's library) through its C ABI: kind 5, the stop ids above."""

    def __init__(self, lib: str, folder: str, vocab: int):
        import ctypes
        self.ct = ct = ctypes
        L = ct.CDLL(lib)
        vp, i32p = ct.c_void_p, ct.POINTER(ct.c_int32)
        L.tfg_open.restype = vp
        L.tfg_open.argtypes = [ct.c_char_p, ct.c_size_t, ct.c_int, i32p, ct.c_int, ct.c_char_p, ct.c_size_t]
        L.tfg_words.argtypes = [vp]
        L.tfg_compile.restype = vp
        L.tfg_compile.argtypes = [vp, ct.c_int, ct.c_char_p, ct.c_size_t, ct.c_char_p, ct.c_size_t]
        L.tfg_matcher.restype = vp
        L.tfg_matcher.argtypes = [vp]
        L.tfg_accept.argtypes = [vp, ct.c_int32]
        L.tfg_terminated.argtypes = [vp]
        L.tfg_fill.argtypes = [vp, i32p, ct.c_int]
        self.L = L
        raw, _ = tokenizer_vocab(folder, vocab)
        err = ct.create_string_buffer(512)
        stops = (ct.c_int32 * len(STOPS))(*STOPS)
        self.c = L.tfg_open(raw, len(raw), vocab, stops, len(STOPS), err, 512)
        if not self.c:
            raise RuntimeError(f"tfg_open: {err.value.decode()}")
        self.words = L.tfg_words(self.c)

    def matcher(self, text: str):
        err = self.ct.create_string_buffer(512)
        b = text.encode()
        g = self.L.tfg_compile(self.c, 5, b, len(b), err, 512)
        if not g:
            raise RuntimeError(f"tfg_compile: {err.value.decode()}")
        return _TfgMatcher(self, self.L.tfg_matcher(g))


class _TfgMatcher:
    def __init__(self, owner, m):
        self.o, self.m = owner, m
        self.buf = (owner.ct.c_int32 * owner.words)()

    def fill(self) -> torch.Tensor:
        if self.o.L.tfg_fill(self.m, self.buf, self.o.words) != 0:
            raise RuntimeError("tfg_fill failed")
        return torch.frombuffer(bytearray(self.buf), dtype=torch.int32)

    def accept(self, t: int) -> bool:
        return self.o.L.tfg_accept(self.m, t) == 1

    def terminated(self) -> bool:
        return self.o.L.tfg_terminated(self.m) == 1


class XgrMasker:
    """The same masks from Python xgrammar (pin 0.2.8): the TokenizerInfo built as tfg_open builds it, and the
    compiler's cache off (v0.2.8's rule cache confuses grammars that differ only in a token id)."""

    def __init__(self, folder: str, vocab: int):
        import xgrammar as xgr
        self.xgr = xgr
        raw, enc = tokenizer_vocab(folder, vocab)
        meta = xgr.TokenizerInfo._detect_metadata_from_hf(raw.decode())  # DetectMetadataFromHF, as tfg_open
        meta = json.loads(meta) if isinstance(meta, str) else meta
        info = xgr.TokenizerInfo(enc, xgr.VocabType(meta["vocab_type"]), vocab_size=vocab, stop_token_ids=STOPS,
                                 add_prefix_space=meta["add_prefix_space"])
        self.compiler = xgr.GrammarCompiler(info, max_threads=8, cache_enabled=False)
        from importlib.metadata import version
        self.vocab, self.version = vocab, version("xgrammar")

    def matcher(self, text: str):
        return _XgrMatcher(self, self.xgr.GrammarMatcher(self.compiler.compile_structural_tag(text)))


class _XgrMatcher:
    def __init__(self, owner, m):
        self.o, self.m = owner, m
        self.buf = owner.xgr.allocate_token_bitmask(1, owner.vocab)

    def fill(self) -> torch.Tensor:
        self.m.fill_next_token_bitmask(self.buf, 0)
        return self.buf[0].clone()

    def accept(self, t: int) -> bool:
        return bool(self.m.accept_token(t))

    def terminated(self) -> bool:
        return bool(self.m.is_terminated())


def allowed_of(words: torch.Tensor, n: int) -> torch.Tensor:
    """Bool [n]: token t is allowed when bit t % 32 of word t // 32 is set."""
    bits = (words.to(torch.int64).unsqueeze(1) >> torch.arange(32)) & 1
    return bits.reshape(-1)[:n].bool()


class Guide:
    """One case's grammar: masks from the main masker, every mask byte-compared with the check masker when given."""

    def __init__(self, text: str, main, check=None):
        self.m = main.matcher(text)
        self.c = check.matcher(text) if check else None
        self.sha = hashlib.sha256()
        self.steps = self.differ = self.changed = 0

    def mask(self) -> torch.Tensor:
        w = self.m.fill()
        if self.c is not None and not torch.equal(w, self.c.fill()):
            self.differ += 1
        self.sha.update(w.numpy().tobytes())
        self.steps += 1
        return w

    def accept(self, t: int):
        if not self.m.accept(t) or (self.c is not None and not self.c.accept(t)):
            raise RuntimeError(f"grammar refused token {t}")

    def terminated(self) -> bool:
        return self.m.terminated()


# ----------------------------------------------------------------------------------------------- provenance


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def provenance(folder: str) -> dict:
    import transformers

    meta = os.path.join(folder, ".cache", "huggingface", "download")
    shards, revision = {}, None
    for f in sorted(os.listdir(folder)):
        mp = os.path.join(meta, f + ".metadata")
        if os.path.exists(mp):
            lines = open(mp).read().split("\n")
            revision = revision or lines[0].strip()
            if f.endswith(".safetensors"):
                shards[f] = {"sha256_hub": lines[1].strip(), "bytes": os.path.getsize(os.path.join(folder, f))}
    small = {f: sha256(os.path.join(folder, f)) for f in
             ("config.json", "hf_quant_config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "model.safetensors.index.json") if os.path.exists(os.path.join(folder, f))}
    tf_commit = None
    try:
        from importlib.metadata import distribution
        du = distribution("transformers").read_text("direct_url.json")
        tf_commit = json.loads(du).get("vcs_info", {}).get("commit_id") if du else None
    except Exception:
        pass
    return {"checkpoint_revision": revision, "shards": shards, "files_sha256": small,
            "transformers": transformers.__version__, "transformers_commit": tf_commit, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


# ----------------------------------------------------------------------------------------------- passes


def teacher_forced(model: Model, specs: list, out: str, layers: int, every: bool = False):
    cfg, dev = model.cfg, model.dev
    emb = model.embed()
    states, rec = [], []
    with torch.no_grad():
        for s in specs:
            ids = torch.tensor([s["ids"]], device=dev)
            e = emb(ids)
            pos = torch.arange(ids.shape[1], device=dev)[None]
            rows = torch.tensor(sample_rows(ids.shape[1], every), device=dev)
            states.append({"h": e, "blocks": e.new_zeros(1, ids.shape[1], 0, e.shape[-1]),
                           "masks": masks(cfg, e, pos), "pos": pos, "rows": rows})
            rec.append({"ids": ids[0].cpu(), "rows": rows.cpu(), "embed": e[0, rows].cpu()})
        del emb
        torch.cuda.empty_cache()
        times = []
        for i in range(layers):
            t0 = time.time()
            layer = model.layer(i)
            t1 = time.time()
            for st, r in zip(states, rec):
                got = {}
                hook = layer.mlp.gate.register_forward_hook(lambda m, a, o: got.update(o=o))
                run_layer(layer, st)
                hook.remove()
                logits, w, idx = got["o"]
                r[f"route.slot_ids.{i}"] = idx.to(torch.int32).cpu()
                r[f"route.slot_w.{i}"] = w.float().cpu()
                order = idx.argsort(-1)
                r[f"route.ids.{i}"] = idx.gather(-1, order).to(torch.int32).cpu()
                r[f"route.w.{i}"] = w.gather(-1, order).float().cpu()
                r[f"route.logits.{i}"] = logits[st["rows"]].float().cpu()
                r[f"h.{i}"] = st["h"][0, st["rows"]].cpu()
            del layer
            torch.cuda.synchronize()
            times.append((i, round(t1 - t0, 2), round(time.time() - t1, 2)))
            print(f"layer {i}: load {t1 - t0:.1f} s, run {time.time() - t1:.1f} s", flush=True)
        head = model.head()
        for st, r in zip(states, rec):
            hn = finish(head, st)
            r["final"] = hn[0, st["rows"]].cpu()
            T = hn.shape[1]
            ti, tv = [], []
            for a in range(0, T, 512):
                lg = logits_of(head, cfg, hn[0, a:a + 512])
                v, k = lg.topk(64, dim=-1)
                ti.append(k.to(torch.int32).cpu())
                tv.append(v.cpu())
            r["topk.ids"], r["topk.vals"] = torch.cat(ti), torch.cat(tv)
            lr = torch.tensor(logit_rows(T))
            r["logits.rows"] = lr
            r["logits.full"] = logits_of(head, cfg, hn[0, lr.to(dev)]).cpu()
    files = {}
    for s, r in zip(specs, rec):
        path = os.path.join(out, s["name"] + ".safetensors")
        save_file({k: v.contiguous() for k, v in r.items()}, path)
        files[os.path.basename(path)] = sha256(path)
    return files, times


def greedy_decode(model: Model, specs: list, steps: int, lru_dev, lru_cap: int, tf_dir: str | None,
                  record: list | None = None, guides: list | None = None, limits: list | None = None,
                  stops: set | None = None):
    """Greedy continuation, all prompts in lockstep: dense layer weights resident, experts fetched per pick.
    `record` (a list per prompt) receives each step's last-row logits. `guides` (a Guide or None per prompt): the pick
    is the argmax over the tokens its grammar allows (the others -inf), and the prompt ends when the grammar does.
    `limits` (per prompt, else `steps`): a prompt's own token limit. `stops`: the ids that end an unguided prompt
    (default 200006 alone). A prompt that has ended is not run further (each prompt has its own cache; the others'
    results do not change)."""
    from transformers.cache_utils import DynamicCache

    cfg, dev = model.cfg, model.dev
    guides = guides or [None] * len(specs)
    lru = ExpertLRU(lru_dev, lru_cap)
    t0 = time.time()
    emb = model.embed()
    layers = [model.layer(i, experts="lazy", lru=lru) for i in range(cfg.num_hidden_layers)]
    head = model.head()
    print(f"resident layers built in {time.time() - t0:.1f} s", flush=True)
    caches = [DynamicCache(config=cfg) for _ in specs]
    outs = [[] for _ in specs]
    report = {}
    eos = stops or {200006}
    limits = limits or [steps] * len(specs)
    ended = [False] * len(specs)
    with torch.no_grad():
        cur = [torch.tensor([s["ids"]], device=dev) for s in specs]
        for step in range(steps + 1):
            live = [k for k in range(len(specs)) if not ended[k]]
            if not live:
                break
            ts = time.time()
            states = {}
            for k in live:
                c, ids = caches[k], cur[k]
                e = emb(ids)
                past = c.get_seq_length()
                pos = (torch.arange(ids.shape[1], device=dev) + past)[None]
                states[k] = {"h": e, "blocks": e.new_zeros(1, ids.shape[1], 0, e.shape[-1]),
                             "masks": masks(cfg, e, pos, c), "pos": pos}
            for layer in layers:
                for k in live:
                    run_layer(layer, states[k], caches[k])
            for k in live:
                st, s, g = states[k], specs[k], guides[k]
                lg = logits_of(head, cfg, finish(head, st)[0, -1:])[0]
                if g is not None:
                    ok = allowed_of(g.mask(), lg.shape[0]).to(lg.device)
                    free = int(lg.argmax())
                    lg = lg.masked_fill(~ok, float("-inf"))
                tok = int(lg.argmax())
                if g is not None:
                    if not bool(ok[tok]):
                        raise RuntimeError(f"{s['name']}: the grammar allows no token at step {step}")
                    g.changed += free != tok
                    g.accept(tok)
                if record is not None:
                    record[k].append(lg.cpu())
                if step == 0:
                    report[s["name"]] = {"prefill_top1": tok, "prefill_top5": lg.topk(5).indices.tolist()}
                    if tf_dir and g is None:
                        from safetensors.torch import load_file
                        ref = load_file(os.path.join(tf_dir, s["name"] + ".safetensors"))
                        report[s["name"]]["tf_top1"] = int(ref["topk.ids"][-1, 0])
                        full = ref["logits.full"][-1].to(dev)
                        report[s["name"]]["tf_max_abs_logit_diff"] = float((full - lg).abs().max())
                if step < limits[k]:
                    outs[k].append(tok)
                ended[k] = (g.terminated() if g is not None else tok in eos) or step >= limits[k]
                cur[k] = torch.tensor([[tok]], device=dev)
            print(f"step {step}: {len(live)} prompts, {time.time() - ts:.1f} s, expert hits {lru.hits} "
                  f"misses {lru.misses}", flush=True)
    return outs, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["full", "checkpoint"], default="full")
    ap.add_argument("--layers", type=int, default=0, help="run only the first N layers, then the final head (0: all)")
    ap.add_argument("--prompts", default="", help="comma-separated prompt names (default: all)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-forward", action="store_true", help="only --decode")
    ap.add_argument("--decode", type=int, default=0, help="greedy-continue --decode-prompts this many tokens")
    ap.add_argument("--decode-prompts", default="chat_short,code,tool_call,long_1500")
    ap.add_argument("--expert-device", default="cuda:1", help="--decode's expert cache")
    ap.add_argument("--expert-cache", type=int, default=1000, help="--decode's expert cache size (experts)")
    ap.add_argument("--all-rows", action="store_true", help="keep every row's hidden states (long prompts too)")
    ap.add_argument("--fp32-reduction", action="store_true",
                    help="bf16 matmuls reduce in fp32 (allow_bf16_reduced_precision_reduction off; torch's default "
                    "lets cuBLAS reduce partly in bf16, about 2e-3 relative noise a projection)")
    ap.add_argument("--structured", default="", help="grammar-masked greedy (with --decode, --skip-forward): "
                    "comma-separated structured_specs names, or 'all'")
    ap.add_argument("--grammar-lib", default="", help="--structured's masks from this libtfgrammar (default: Python "
                    "xgrammar)")
    ap.add_argument("--xgrammar-check", action="store_true", help="--structured: byte-compare every mask with Python "
                    "xgrammar's (cache off) and fail on any difference")
    ap.add_argument("--cases-out", default="", help="--structured: also write one <case>.json per case here")
    ap.add_argument("--plain-steps", type=int, default=160, help="--structured: each case's prompt also greedy "
                    "without its grammar, at most this many tokens (stops: the server's), as `plain_ids` (0: not)")
    a = ap.parse_args()
    if a.structured:
        a.fp32_reduction = True  # the masked oracle's picks meet near-ties: no bf16 partial sums in its matmuls
    if a.fp32_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    from transformers import AutoTokenizer

    dev = torch.device(a.device)
    cfg = text_config(a.model_dir, a.layers)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    specs = prompt_specs(tok)
    for s in specs:
        s["text"], s["ids"] = render(tok, s)
    if a.prompts:
        want = a.prompts.split(",")
        specs = [s for s in specs if s["name"] in want]
    out = os.path.join(a.out, a.mode if not a.layers else f"{a.mode}-L{a.layers}")
    os.makedirs(out, exist_ok=True)
    man_path = os.path.join(out, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    man.update(provenance(a.model_dir))
    man.update({
        "mode": a.mode, "layers": cfg.num_hidden_layers, "device": a.device,
        "dtypes": {"activations": "bf16", "norms_router_softmax": "fp32", "kept_fp32": "Transformers' "
                   "_keep_in_fp32_modules_strict (router weight and bias, convolutions of sliding k/v and "
                   "post-attention/post-MLP, block-residual queries)",
                   "experts": "bf16 decoded weights" if a.mode == "full" else "checkpoint bytes, activations "
                   "quantized (TokenSpeed reference)", "logits": "fp32 softcapped"},
        "experts_implementation": "eager", "attn_implementation": "eager",
        "expert_sum_order": "fp32 per row over the picks in ascending expert id, one bf16 rounding" if a.mode == "full"
        else "fp32 per row over the bf16 route rows in slot order (route.slot_ids: TokenSpeed's order, descending "
        "sigmoid(logit) + bias, ties to the lower id), one bf16 rounding",
        "prompts": [{k: s[k] for k in ("name", "effort", "messages", "text", "ids")} | {"tokens": len(s["ids"])}
                    for s in specs],
        "row_sampling": "all rows" if a.all_rows else
        "all rows when T <= 512; else first/last 64, every 16th, 1020..1031, 2044..2051, 3068..3075",
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "logit_rows": "last 8 rows and every 256th row",
    })
    model = Model(a.model_dir, cfg, dev, a.mode)
    if a.structured:
        return structured(a, model, tok, cfg, out, man, man_path)
    if not a.skip_forward:
        t0 = time.time()
        files, times = teacher_forced(model, specs, out, cfg.num_hidden_layers, a.all_rows)
        man["files_out_sha256"] = files
        man["forward_wall_s"] = round(time.time() - t0, 1)
        man["layer_times_s"] = times
        print(f"teacher-forced pass: {man['forward_wall_s']} s", flush=True)
        json.dump(man, open(man_path, "w"), indent=1)
    if a.decode:
        want = a.decode_prompts.split(",")
        dspecs = [s for s in specs if s["name"] in want]
        t0 = time.time()
        outs, report = greedy_decode(model, dspecs, a.decode, torch.device(a.expert_device), a.expert_cache,
                                     out if os.path.exists(os.path.join(out, dspecs[0]["name"] + ".safetensors"))
                                     else None)
        res = {}
        for s, o in zip(dspecs, outs):
            res[s["name"]] = {"ids": o, "text": tok.decode(o, skip_special_tokens=False), **report[s["name"]]}
            print(f"--- {s['name']}\n{res[s['name']]['text']}", flush=True)
        res["_wall_s"] = round(time.time() - t0, 1)
        with open(os.path.join(out, "decode.json"), "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        man["decode_wall_s"] = res["_wall_s"]
        json.dump(man, open(man_path, "w"), indent=1)


def ids_sha(ids: list) -> str:
    """sha256 of the ids as little-endian int32, first 16 hex digits."""
    return hashlib.sha256(torch.tensor(ids, dtype=torch.int32).numpy().astype("<i4").tobytes()).hexdigest()[:16]


def structured(a, model: Model, tok, cfg, out: str, man: dict, man_path: str):
    """--structured: each case's grammar-masked greedy reply (checkpoint mode: the server's numerics' oracle)."""
    specs = structured_specs()
    if a.structured != "all":
        want = a.structured.split(",")
        specs = [s for s in specs if s["name"] in want]
    for s in specs:
        s["text"], s["ids"] = render(tok, s)
    vocab = cfg.vocab_size
    main_m = TfgMasker(a.grammar_lib, a.model_dir, vocab) if a.grammar_lib else XgrMasker(a.model_dir, vocab)
    check = XgrMasker(a.model_dir, vocab) if a.xgrammar_check and a.grammar_lib else None
    source = "libtfgrammar (C ABI, kind 5)" if a.grammar_lib else f"Python xgrammar {main_m.version} (cache off)"
    t0 = time.time()
    guides = [Guide(s["grammar"], main_m, check) for s in specs]
    print(f"{len(specs)} grammars compiled in {time.time() - t0:.1f} s; masks from {source}"
          f"{f', checked against Python xgrammar {check.version}' if check else ''}", flush=True)
    t0 = time.time()
    n = len(specs)
    plain = [dict(s, name=s["name"] + "+plain") for s in specs] if a.plain_steps > 0 else []
    outs, report = greedy_decode(model, specs + plain, a.decode, torch.device(a.expert_device), a.expert_cache, None,
                                 guides=guides + [None] * len(plain),
                                 limits=[a.decode] * n + [a.plain_steps] * len(plain), stops=set(STOPS))
    plain_outs = outs[n:] if plain else [None] * n
    outs = outs[:n]
    wall = round(time.time() - t0, 1)
    res, fails = {}, 0
    for s, o, g, po in zip(specs, outs, guides, plain_outs):
        r = {"ids": o, "ids_sha": ids_sha(o), "tokens": len(o), "ended": g.terminated(),
             "text": tok.decode(o, skip_special_tokens=False), "mask_steps": g.steps,
             "mask_sha": g.sha.hexdigest()[:16], "masked_picks": g.changed,
             "mask_check": None if check is None else ("equal" if g.differ == 0 else f"{g.differ} differ"),
             "prefill_top1": report[s["name"]]["prefill_top1"]}
        if po is not None:
            r.update({"plain_request": plain_request(s["request"]), "plain_ids": po, "plain_ids_sha": ids_sha(po),
                      "plain_tokens": len(po), "plain_text": tok.decode(po, skip_special_tokens=False),
                      "first_masked_pick": next((i for i, (x, y) in enumerate(zip(o, po)) if x != y), None)})
        fails += g.differ + (not g.terminated())
        res[s["name"]] = r
        print(f"--- {s['name']}: {len(o)} tokens, ended {r['ended']}, ids {r['ids_sha']}, masks {r['mask_sha']} "
              f"({g.changed} picks changed by the mask), check {r['mask_check']}\n{r['text']}", flush=True)
        if a.cases_out:
            os.makedirs(a.cases_out, exist_ok=True)
            case = {"name": s["name"], "mode": a.mode, "request": s["request"], "grammar_kind": 5,
                    "grammar": s["grammar"], "grammar_sha": hashlib.sha256(s["grammar"].encode()).hexdigest()[:16],
                    "stops": STOPS, "vocab": vocab, "prompt_ids": s["ids"], "prompt_text": s["text"],
                    "prompt_sha": ids_sha(s["ids"]), **{k: r[k] for k in r if k != "prefill_top1"},
                    "mask_source": source, "fp32_reduction": not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    "checkpoint_revision": man.get("checkpoint_revision"),
                    "files_sha256": {k: man.get("files_sha256", {}).get(k) for k in ("tokenizer.json",
                                                                                     "chat_template.jinja")}}
            with open(os.path.join(a.cases_out, s["name"] + ".json"), "w") as f:
                json.dump(case, f, indent=1, ensure_ascii=False)
                f.write("\n")
    res["_wall_s"], res["_mask_source"] = wall, source
    with open(os.path.join(out, "structured.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    man["structured_wall_s"] = wall
    man["structured_prompts"] = [{k: s[k] for k in ("name", "request", "grammar", "text", "ids")} for s in specs]
    json.dump(man, open(man_path, "w"), indent=1)
    print(f"structured: {len(specs)} cases in {wall} s, {fails} problems", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
