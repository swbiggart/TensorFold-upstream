"""Checks of inkling2_ref.py against code it does not share: Transformers' loader and model, TokenSpeed's functions.

USAGE = python tools/zig/inkling2_ref_check.py MODEL_DIR --work DIR CHECK [options]

Every check writes DIR/check-<name>.json: its explicit acceptance items (each pass or fail, with the numbers), the
information around them, and exits 1 when any item fails. Forward comparisons are bit-exact; the only tolerances
are the cached decode's logits (--decode-tol) and the full-against-checkpoint expert bound (--quant-bound), both
reported apart from the exact items.

truncated --layers N [--prompts a,b] [--subset] [--decode-check K]
    Streaming and naming. inkling2_ref.py runs the first N layers (then final_input, norm, unembed, softcap) on the
    prompts. A truncated checkpoint is written in the release's names (`model.llm.*`), experts decoded to bf16 in the
    release's interleaved gate/up row order, with an `inkling2_text` config of N layers; unmodified
    `Inkling2ForCausalLM.from_pretrained` loads it through Transformers' conversion mapping (renames, the attn_queries
    index offset, gate/up de-interleave, fp32 modules) onto the two GPUs. Accepted: nothing missing, unexpected or
    mismatched; parameters equal in name, dtype and bits; every layer output, router pick and weight, the final
    state, logits and top-64 bit-equal. --subset keeps, per layer, only the experts the streamed run routed a row to
    (padded with experts whose selection bias is -1e9, so never picked; ids mapped back), so N=12 fits two 96 GB
    cards. The expert decode here is the oracle's own: the `weights` and `layers` checks cover it.
    --decode-check K also greedy-decodes K <= 7 tokens with the cache (as --decode does) and compares each step's
    logits with a teacher-forced pass over prompt + continuation: argmax equal, max abs <= --decode-tol (0.25: the
    unembed rounds each logit to bf16 before the softcap, one bf16 step below 64 is at most 0.25, and the decode's
    single row is another GEMM shape than the teacher-forced chunk).

weights --tokenspeed TS_DIR [--layers 0,20,39,40,47] [--experts 0,1,97,191,290,383]
    The experts as stored, decoded by the oracle (`nvfp4_values`, `mxfp8_values`, `dequant_rows`) and, independently,
    by TokenSpeed's `_decode_values` (its block_scaled.py from TS_DIR, run unmodified) and Transformers'
    `Fp8Dequantize._dequantize_one` (its load-time FP8 / packed-FP4 dequantizer), both projections, NVFP4 layers
    (0-39) and MXFP8 layers (40-47). Accepted: the fp32 block decode bit-equal to both; the bf16 full-mode weight
    bit-equal to TokenSpeed's decode times its per-expert outer scale amax[e] / denominator (the weight half of its
    route scale (x_amax * amax[e]) / (1536 * denominator)) for NVFP4, and to Transformers' bf16 dequantization for
    MXFP8.

experts --tokenspeed TS_DIR [--layers 20,44] [--prompts tool_call,long_1500]
    Checkpoint mode on real activations. The oracle runs in checkpoint mode up to each layer; the experts' inputs
    (rows, picks, weights, in slot order) and output are captured, and TokenSpeed's own
    `torch_block_scaled_moe_routes` (fold_slots, no EP) runs on the same inputs and the layer's raw tensors.
    Accepted: bit-equal; TokenSpeed's router (`block_scaled_router_mm` + `torch_block_scaled_topk`) on the same rows
    picks the same experts in the same slot order the oracle used. Reported: its weights against Transformers' (the
    oracle's), and TokenSpeed's output with the picks in expert-id order (what the fold order changes). Also the
    full mode's experts (Transformers' eager `Inkling2Experts` on the bf16 decode) against TokenSpeed's: they differ
    by the activation quantization only, accepted below --quant-bound (0.25) in relative norm: measured about 0.10
    at NVFP4 layers and 0.035 at MXFP8 ones, where a lost outer scale, a swapped nibble or a wrong scale axis is of
    order 1 or more. The fold order itself rarely shows: the slot sum adds eight bf16 rows in fp32, exact unless
    their exponents spread over more than about 13 binades.

layers --oracle DIR [--layers 40,41,42,43,44,45,46,47] [--prompts chat_short,tool_call]
    Late layers through Transformers' own loader, on the oracle's saved inputs (DIR: a full-mode output folder; the
    prompts must have every row saved, T <= 512). The layers' raw tensors (all 384 MXFP8 experts as stored; a
    subset would change the router GEMM's shape and its rounding) are written in the release's names at indices
    congruent mod 6 (layer type,
    block boundary and attention_input depend only on that; layer L sits at L - 6*floor((min - 1)/6)); missing
    earlier layers stay random on the CPU and are not run. `from_pretrained` loads them with the release's own MXFP8
    quantization config (Transformers' INKLING2_QUANTIZATION_CONFIG: ue8m0, [1, 32]), as it loads the MXFP8 release:
    `FP8Experts` holding the e4m3 weights and ue8m0 scales its conversion mapping renamed and de-interleaved.
    (dequantize=True does not work for Inkling2 at this commit: its converters for `w13_weight` never collect the
    scales, so the raw e4m3 values would load unscaled, and the router is not kept in fp32.) Those tensors are
    decoded by Transformers' `Fp8Dequantize._dequantize_one` and put in an eager `Inkling2Experts`; each layer's
    module is then called as `Inkling2TextModel.forward` calls it, on the oracle's input h.<L-1> and block states
    (embed, h.5, h.11, ...: the inputs of layers 0, 6, ... below L). Accepted: output bit-equal to h.<L>, router
    logits, picks and weights bit-equal, the decoded experts and every other parameter bit-equal to the oracle's.
    Reported: the same layer with Transformers' own `FP8Experts` forward (FP8 compute, not the oracle's arithmetic).

TS_DIR is a TokenSpeed checkout: its tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/block_scaled.py is executed as
is, with the imports its Torch reference does not use (kernel registry, GEMM back ends, mailbox, platform) stubbed.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import types

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling2_ref as R  # noqa: E402

CK = R.CK
BF = R.BF
INT_OF = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16,
          torch.float8_e4m3fn: torch.uint8}


# ----------------------------------------------------------------------------------------------- comparison


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape, dtype and bits (so -0.0 differs from 0.0 and a NaN equals the same NaN)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    b = b.to(a.device)
    if a.dtype in INT_OF:
        a, b = a.contiguous().view(INT_OF[a.dtype]), b.contiguous().view(INT_OF[b.dtype])
    return bool(torch.equal(a, b))


def diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    eq = bit_equal(a, b)
    a, b = a.float(), b.float().to(a.device)
    d = (a - b).abs()
    if d.numel() == 0:
        return {"max_abs": 0.0, "max_rel": 0.0, "norm_rel": 0.0, "bit_equal": eq}
    return {"max_abs": float(d.max()), "max_rel": float((d / b.abs().clamp(min=1e-6)).max()),
            "norm_rel": float(d.norm() / b.norm().clamp(min=1e-30)), "bit_equal": eq}


class Acceptance:
    """Explicit pass/fail items of one check."""

    def __init__(self, report: dict):
        self.items = report.setdefault("acceptance", [])

    def __call__(self, item: str, ok: bool, **numbers) -> bool:
        self.items.append({"item": item, "pass": bool(ok), **numbers})
        if not ok:
            print(f"FAIL {item} {numbers}", flush=True)
        return bool(ok)


def finish(report: dict, path: str) -> int:
    failed = [i["item"] for i in report.get("acceptance", []) if not i["pass"]]
    n = len(report.get("acceptance", []))
    report["summary"] = {"items": n, "passed": n - len(failed), "failed": failed}
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"{report['check']}: {n - len(failed)}/{n} acceptance items pass" + (f"; failed: {failed[:20]}" if failed
                                                                                 else ""), flush=True)
    return 1 if failed or n == 0 else 0


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def load_specs(model_dir: str, names: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    specs = [s for s in R.prompt_specs(tok) if s["name"] in names.split(",")]
    for s in specs:
        s["text"], s["ids"] = R.render(tok, s)
    return specs


# ----------------------------------------------------------------------------------------------- TokenSpeed


def load_tokenspeed(ts_dir: str):
    """TokenSpeed's block_scaled.py, executed unmodified, its unused imports stubbed."""
    path = os.path.join(ts_dir, "tokenspeed-kernel", "python", "tokenspeed_kernel", "ops", "moe", "block_scaled.py")

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    for name in ("tokenspeed_kernel", "tokenspeed_kernel.ops", "tokenspeed_kernel.ops.communication",
                 "tokenspeed_kernel.ops.moe._triton", "tokenspeed_kernel.ops.moe._triton.block_scaled"):
        mod(name)
    mod("tokenspeed_kernel.ops.communication.mailbox", MailboxTarget=type("MailboxTarget", (), {}))
    mod("tokenspeed_kernel.ops.gemm", mm=None)
    mod("tokenspeed_kernel.ops.gemm.cute_dsl", cute_dsl_ll_bf16_router=None, ll_bf16_router_supported=None)
    mod("tokenspeed_kernel.ops.gemm.triton_skinny", skinny_mm=None, skinny_mm_supported=None)
    mod("tokenspeed_kernel.ops.moe", ExpertDispatch=object, dispatch_topk_ids=None)
    mod("tokenspeed_kernel.platform", current_platform=lambda: types.SimpleNamespace(is_nvidia=False))
    mod("tokenspeed_kernel.registry", KernelRegistry=None, Priority=types.SimpleNamespace(REFERENCE=0),
        register_kernel=lambda *a, **k: (lambda f: f))
    mod("tokenspeed_kernel.selection", select_kernel=None)
    mod("tokenspeed_kernel.signature", dense_tensor_format=lambda *a, **k: None, format_signature=lambda **k: None)
    spec = importlib.util.spec_from_file_location("tokenspeed_kernel.ops.moe.block_scaled", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    with open(path, "rb") as f:
        m.SHA256 = hashlib.sha256(f.read()).hexdigest()
    return m


def raw_experts(src: R.Source, prefix: str, dev):
    """The layer's expert tensors as stored, in torch_block_scaled_moe_routes' argument order."""
    g = lambda n: src.get(prefix + n, dev)
    if R.expert_format(src, prefix) == "nvfp4":
        return [g("w13_weight"), g("w13_weight_scale"), g("w13_weight_amax"), g("w13_weight_outer_scale_denominator"),
                g("w2_weight"), g("w2_weight_scale"), g("w2_weight_amax"), g("w2_weight_outer_scale_denominator")]
    return [g("w13_weight"), g("w13_weight_scale"), None, None, g("w2_weight"), g("w2_weight_scale"), None, None]


# ----------------------------------------------------------------------------------------------- truncated


def write_truncated(m: R.Model, cfg, n: int, path: str, subset: dict | None):
    """The release's tensors for the first n layers (experts as bf16, interleaved rows) plus a text config."""
    src, dev = m.src, m.dev
    os.makedirs(path, exist_ok=True)
    wm = {}

    def put(fname, tensors):
        save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, os.path.join(path, fname))
        for k in tensors:
            wm[k] = fname

    top = {CK + k: src.get(CK + k) for k in ("embed.weight", "embed_norm.weight", "norm.weight", "unembed.weight",
                                               "attn_res.output_query")}
    top.update({f"{CK}attn_res.attn_queries.{i}": src.get(f"{CK}attn_res.attn_queries.{i}") for i in range(n - 1)})
    top.update({f"{CK}attn_res.mlp_queries.{i}": src.get(f"{CK}attn_res.mlp_queries.{i}") for i in range(n)})
    put("top.safetensors", top)
    k_max = max(len(v) for v in subset.values()) if subset else cfg.n_routed_experts
    for i in range(n):
        p = f"{CK}layers.{i}."
        t = {}
        for name in src.names(p):
            if ".mlp.experts." not in name:
                t[name] = src.get(name)
        ep = p + "mlp.experts."
        if subset is None:
            gu, dn = [], []
            for a in range(0, cfg.n_routed_experts, 16):
                gu.append(R.dequant_rows(src, ep, "w13", a, a + 16, dev).cpu())
                dn.append(R.dequant_rows(src, ep, "w2", a, a + 16, dev).cpu())
            t[ep + "w13_weight"], t[ep + "w2_weight"] = torch.cat(gu), torch.cat(dn)
        else:
            keep = subset[i]
            pad = k_max - len(keep)
            gu = torch.cat([R.dequant_rows(src, ep, "w13", e, e + 1, dev).cpu() for e in keep])
            dn = torch.cat([R.dequant_rows(src, ep, "w2", e, e + 1, dev).cpu() for e in keep])
            t[ep + "w13_weight"] = torch.cat([gu, gu.new_zeros(pad, *gu.shape[1:])])
            t[ep + "w2_weight"] = torch.cat([dn, dn.new_zeros(pad, *dn.shape[1:])])
            gw, gb = t[p + "mlp.gate.weight"], t[p + "mlp.gate.bias"]
            t[p + "mlp.gate.weight"] = torch.cat([gw[keep], gw.new_zeros(pad, gw.shape[1])])
            t[p + "mlp.gate.bias"] = torch.cat([gb[keep], gb.new_full((pad,), -1e9)])
        put(f"layer{i:02d}.safetensors", t)
        print(f"wrote layer {i}", flush=True)
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(path, "model.safetensors.index.json"), "w"))
    c = cfg.to_dict()
    c.pop("transformers_version", None)
    c.update(model_type="inkling2_text", architectures=["Inkling2ForCausalLM"], n_routed_experts=k_max,
             num_hidden_layers=n, layer_types=cfg.layer_types[:n], torch_dtype="bfloat16")
    for k in [k for k in c if k.startswith("_")]:
        c.pop(k)
    json.dump(c, open(os.path.join(path, "config.json"), "w"), indent=1)


def decode_check(m: R.Model, specs: list, k: int, work: str, n: int, tol: float, acc: Acceptance) -> dict:
    """Cached greedy decode (lazy experts, DynamicCache) against the streamed teacher-forced pass of its tokens."""
    rec = [[] for _ in specs]
    outs, _ = R.greedy_decode(m, specs, k, torch.device("cuda:1"), 600, None, record=rec)
    tf = [dict(s, ids=s["ids"] + o[:-1], name=s["name"] + "_dec") for s, o in zip(specs, outs)]
    d = os.path.join(work, f"dec-L{n}")
    os.makedirs(d, exist_ok=True)
    R.teacher_forced(m, tf, d, n)
    res = {}
    for s, o, r in zip(tf, outs, rec):
        ref = load_file(os.path.join(d, s["name"] + ".safetensors"))
        T = len(s["ids"])
        rows = ref["logits.rows"].tolist()
        steps = {}
        for j in range(len(o)):
            row = T - len(o) + j
            steps[j] = diff(r[j], ref["logits.full"][rows.index(row)])
            steps[j]["argmax_equal"] = int(r[j].argmax()) == int(ref["logits.full"][rows.index(row)].argmax())
        res[s["name"]] = {"tokens": o, "steps": steps}
        worst = max(x["max_abs"] for x in steps.values())
        acc(f"decode {s['name']}: {len(o)} steps, argmax equal", all(x["argmax_equal"] for x in steps.values()))
        acc(f"decode {s['name']}: logits max abs <= {tol}", worst <= tol, max_abs=worst)
    return res


def check_truncated(a, report: dict, acc: Acceptance):
    from transformers import Inkling2ForCausalLM

    n = a.layers
    dev = torch.device("cuda:0")
    cfg = R.text_config(a.model_dir, n)
    specs = load_specs(a.model_dir, a.prompts)
    ref_dir = os.path.join(a.work, f"ref-L{n}")
    os.makedirs(ref_dir, exist_ok=True)
    m = R.Model(a.model_dir, cfg, dev, "full")
    t0 = time.time()
    R.teacher_forced(m, specs, ref_dir, n)
    torch.cuda.empty_cache()
    refs = {s["name"]: load_file(os.path.join(ref_dir, s["name"] + ".safetensors")) for s in specs}
    print(f"streamed run {time.time() - t0:.1f} s", flush=True)

    subset = None
    if a.subset:
        subset = {i: sorted(set().union(*[set(r[f"route.ids.{i}"].flatten().tolist()) for r in refs.values()]))
                  for i in range(n)}
        print("subset sizes", {i: len(v) for i, v in subset.items()}, flush=True)
    hf_dir = os.path.join(a.work, f"hf-L{n}")
    t0 = time.time()
    write_truncated(m, cfg, n, hf_dir, subset)
    print(f"truncated checkpoint {time.time() - t0:.1f} s", flush=True)

    half = (n + 1) // 2
    dm = {"model.embed_tokens": 0, "model.norm": 1, "model.final_input": 1, "lm_head": 1}
    dm.update({f"model.layers.{i}": 0 if i < half else 1 for i in range(n)})
    model, info = Inkling2ForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16, device_map=dm,
                                                      attn_implementation="eager", experts_implementation="eager",
                                                      output_loading_info=True)
    model.eval()
    report.update(layers=n, subset=bool(a.subset), loading_info={k: sorted(map(str, v)) for k, v in info.items()})
    for k, v in report["loading_info"].items():
        acc(f"loader {k} empty", not v, entries=v[:20])

    # parameters: Transformers' loader against the streamed layers' assignment
    pdiff = []
    hf_sd = model.state_dict()
    for i in range(n):
        mine = m.layer(i, experts="lazy", lru=R.ExpertLRU(dev, 1)).state_dict()
        for k, v in mine.items():
            hv = hf_sd[f"model.layers.{i}.{k}"]
            if subset and k.startswith("mlp.gate."):
                hv = hv[: len(subset[i])]
                v = v[subset[i]]
            if not bit_equal(hv.cpu(), v.cpu()):
                pdiff.append(f"layers.{i}.{k}: {hv.dtype} vs {v.dtype}")
        hgu, hdn = hf_sd[f"model.layers.{i}.mlp.experts.gate_up_proj"], hf_sd[f"model.layers.{i}.mlp.experts.down_proj"]
        for j, e in enumerate(subset[i] if subset else [0, 1, 191, 383]):
            gu = R.deinterleave(R.dequant_rows(m.src, f"{CK}layers.{i}.mlp.experts.", "w13", e, e + 1, dev))[0]
            dn = R.dequant_rows(m.src, f"{CK}layers.{i}.mlp.experts.", "w2", e, e + 1, dev)[0]
            hj = j if subset else e
            if not (bit_equal(hgu[hj].to(dev), gu) and bit_equal(hdn[hj].to(dev), dn)):
                pdiff.append(f"layers.{i} expert {e}")
        del mine
    for k in ("model.embed_tokens.weight", "model.embed_tokens.embed_norm.weight", "model.norm.weight",
              "model.final_input.query", "lm_head.weight"):
        report.setdefault("top_dtypes", {})[k] = str(hf_sd[k].dtype)
    acc("parameters equal (names, dtypes, bits)", not pdiff, mismatches=pdiff[:20])

    # forward. Transformers' hidden_states end with the final normed state in place of the last layer's output, so
    # the layers' own outputs are read with hooks (which change nothing)
    routes, louts = {}, {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.gate.register_forward_hook(lambda mod, inp, out, i=i: routes.update({i: out})))
        hooks.append(layer.register_forward_hook(lambda mod, inp, out, i=i: louts.update({i: out})))
    res = {}
    with torch.no_grad():
        for s in specs:
            r = refs[s["name"]]
            ids = torch.tensor([s["ids"]], device=dev)
            out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            rows = r["rows"]
            off = len(hs) - n
            one = {"tokens": len(s["ids"]), "hidden_states_returned": len(hs)}
            if off == 1:
                one["embed"] = diff(hs[0][0, rows], r["embed"])
            one["final"] = diff(hs[-1][0, rows], r["final"])
            for i in range(n):
                lo = louts[i][0] if isinstance(louts[i], tuple) else louts[i]
                one[f"h.{i}"] = diff(lo[0, rows], r[f"h.{i}"])
                lg, w, idx = routes[i]
                if subset:
                    idx = torch.tensor(subset[i], device=idx.device)[idx]
                else:
                    one[f"route.{i}.slot_order_equal"] = bool(torch.equal(idx.cpu().to(torch.int32),
                                                                          r[f"route.slot_ids.{i}"]))
                order = idx.argsort(-1)
                hid, hw = idx.gather(-1, order).cpu().to(torch.int32), w.gather(-1, order).float().cpu()
                one[f"route.{i}"] = {"ids_equal": bool(torch.equal(hid, r[f"route.ids.{i}"])),
                                     "w": diff(hw, r[f"route.w.{i}"])}
            lr = r["logits.rows"]
            one["logits.full"] = diff(out.logits[0, lr.to(dev)].cpu(), r["logits.full"])
            v, k = out.logits[0].topk(64, dim=-1)
            one["topk.ids_equal"] = bool(torch.equal(k.to(torch.int32).cpu(), r["topk.ids"]))
            one["topk.vals"] = diff(v.cpu(), r["topk.vals"])
            res[s["name"]] = one
            p = s["name"]
            for key in ["embed", "final", "logits.full", "topk.vals"] + [f"h.{i}" for i in range(n)]:
                if key in one:
                    acc(f"{p} {key} bit-equal", one[key]["bit_equal"], max_abs=one[key]["max_abs"])
            acc(f"{p} top-64 ids equal", one["topk.ids_equal"])
            for i in range(n):
                acc(f"{p} layer {i} picks equal, weights bit-equal",
                    one[f"route.{i}"]["ids_equal"] and one[f"route.{i}"]["w"]["bit_equal"])
            worst = max(x["max_abs"] for key, x in one.items() if key.startswith("h."))
            print(f"{p} ({len(s['ids'])} tokens): worst layer max abs {worst:.4g}; logits {one['logits.full']}",
                  flush=True)
    report["prompts"] = res
    if a.decode_check:
        for h in hooks:
            h.remove()
        del model, hf_sd, layer, out, hooks, routes, louts
        gc.collect()
        torch.cuda.empty_cache()
        report["decode_check"] = decode_check(m, specs, a.decode_check, a.work, n, a.decode_tol, acc)
    if not a.keep:
        shutil.rmtree(hf_dir)


# ----------------------------------------------------------------------------------------------- weights


def check_weights(a, report: dict, acc: Acceptance):
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    TS = load_tokenspeed(a.tokenspeed)
    report["tokenspeed_block_scaled_sha256"] = TS.SHA256
    dq = Fp8Dequantize(None)
    src = R.Source(a.model_dir)
    dev = torch.device("cuda:0")
    info = report.setdefault("layers", {})
    for L in ints(a.layers):
        ep = f"{CK}layers.{L}.mlp.experts."
        fmt = R.expert_format(src, ep)
        nv = fmt == "nvfp4"
        li = info[str(L)] = {"format": fmt}
        for leaf in ("w13", "w2"):
            den = src.get(f"{ep}{leaf}_weight_outer_scale_denominator", dev) if nv else None
            if nv:
                li[f"{leaf}_denominator"] = den.tolist()
            for e in ints(a.experts):
                w = src.rows(f"{ep}{leaf}_weight", e, e + 1, dev)[0]
                s = src.rows(f"{ep}{leaf}_weight_scale", e, e + 1, dev)[0]
                tag = f"layer {L} {fmt} {leaf} expert {e}"
                mine = R.nvfp4_values(w, s) if nv else R.mxfp8_values(w, s)
                ts = TS._decode_values(w, s, fmt)
                hf = dq._dequantize_one(w.view(torch.int8) if nv else w, s, output_dtype=torch.float32)
                acc(f"{tag}: fp32 block decode = TokenSpeed _decode_values", bit_equal(mine, ts),
                    **diff(mine, ts))
                acc(f"{tag}: fp32 block decode = Transformers Fp8Dequantize", bit_equal(mine, hf),
                    **diff(mine, hf))
                full = R.dequant_rows(src, ep, leaf, e, e + 1, dev)[0]
                if nv:
                    amax = src.rows(f"{ep}{leaf}_weight_amax", e, e + 1, dev)[0]
                    ref = (ts * (amax / den[0])).to(BF)
                    what = "TokenSpeed decode x amax[e] / denominator"
                else:
                    ref = dq._dequantize_one(w, s, output_dtype=BF)
                    what = "Transformers Fp8Dequantize to bf16"
                acc(f"{tag}: bf16 full-mode weight = {what}", bit_equal(full, ref), **diff(full, ref))
                if not nv and e == ints(a.experts)[0]:
                    # the manifest's weight_block_shape is [32, 32]: are the per-row scales constant over 32 rows?
                    g = s.view(s.shape[0] // 32, 32, s.shape[1])
                    li[f"{leaf}_scale_rows_constant_over_32"] = float((g == g[:, :1]).all(1).float().mean())
                li.setdefault(f"{leaf}_shapes", [list(w.shape), str(w.dtype), list(s.shape), str(s.dtype)])
        print(f"layer {L} ({fmt}) done", flush=True)


# ----------------------------------------------------------------------------------------------- experts


def check_experts(a, report: dict, acc: Acceptance):
    TS = load_tokenspeed(a.tokenspeed)
    report["tokenspeed_block_scaled_sha256"] = TS.SHA256
    layers = sorted(ints(a.layers))
    n = layers[-1] + 1
    dev, dev1 = torch.device("cuda:0"), torch.device("cuda:1")
    cfg = R.text_config(a.model_dir, n)
    specs = load_specs(a.model_dir, a.prompts)
    m = R.Model(a.model_dir, cfg, dev, "checkpoint")
    cap = {}
    t0 = time.time()
    with torch.no_grad():
        emb = m.embed()
        states = []
        for s in specs:
            ids = torch.tensor([s["ids"]], device=dev)
            e = emb(ids)
            pos = torch.arange(ids.shape[1], device=dev)[None]
            states.append({"h": e, "blocks": e.new_zeros(1, ids.shape[1], 0, e.shape[-1]),
                           "masks": R.masks(cfg, e, pos), "pos": pos})
        del emb
        for i in range(n):
            layer = m.layer(i)
            for st, s in zip(states, specs):
                got, hooks = {}, []
                if i in layers:
                    hooks.append(layer.mlp.experts.register_forward_hook(
                        lambda mod, inp, out, g=got: g.update(x=inp[0], ids=inp[1], w=inp[2], out=out)))
                    hooks.append(layer.mlp.gate.register_forward_hook(
                        lambda mod, inp, out, g=got: g.update(gx=inp[0], gate=out)))
                R.run_layer(layer, st)
                for h in hooks:
                    h.remove()
                if got:
                    cap[(i, s["name"])] = got
            del layer
            print(f"layer {i} streamed ({time.time() - t0:.0f} s)", flush=True)
    del states
    torch.cuda.empty_cache()

    res = report.setdefault("layers", {})
    mf = R.Model(a.model_dir, cfg, dev1, "full")
    for L in layers:
        ep = f"{CK}layers.{L}.mlp.experts."
        fmt = R.expert_format(m.src, ep)
        raw = raw_experts(m.src, ep, dev)
        gw = m.src.get(f"{CK}layers.{L}.mlp.gate.weight", dev)
        gb = m.src.get(f"{CK}layers.{L}.mlp.gate.bias", dev)
        full = mf.layer(L).mlp.experts
        for s in specs:
            p = s["name"]
            g = cap[(L, p)]
            x, ids, w, out = g["x"], g["ids"], g["w"], g["out"]
            tag = f"layer {L} {fmt} {p} ({x.shape[0]} rows, {ids.unique().numel()} experts)"
            with torch.no_grad():
                ts = TS.torch_block_scaled_moe_routes(
                    x.contiguous(), *raw, w.float().contiguous(), ids.contiguous(), formats=(fmt, fmt),
                    experts=cfg.n_routed_experts, expert_start=0, gate_cap=cfg.mlp_gate_softcap,
                    up_cap=cfg.mlp_up_softcap, zero_remote_routes=False, fold_slots=True, fold_into=None)[:, 0]
                by_id = ids.argsort(-1)
                ts_id = TS.torch_block_scaled_moe_routes(
                    x.contiguous(), *raw, w.float().gather(-1, by_id).contiguous(), ids.gather(-1, by_id).contiguous(),
                    formats=(fmt, fmt), experts=cfg.n_routed_experts, expert_start=0, gate_cap=cfg.mlp_gate_softcap,
                    up_cap=cfg.mlp_up_softcap, zero_remote_routes=False, fold_slots=True, fold_into=None)[:, 0]
                lg = TS.block_scaled_router_mm(g["gx"].contiguous(), gw, solution="torch")
                tw, tid = TS.torch_block_scaled_topk(lg, gb.float(), top_k=cfg.num_experts_per_tok,
                                                     route_scale=cfg.route_scale, dispatch=None)
                fo = full(x.to(dev1), ids.to(dev1), w.to(dev1)).to(dev)
            hl, hw, hid = g["gate"]
            one = {"oracle_vs_tokenspeed": diff(out, ts),
                   "tokenspeed_slot_order_vs_expert_id_order": diff(ts_id, ts) | {
                       "rows_differing": int((ts_id != ts).any(-1).sum())},
                   "router_logits_vs_tokenspeed": diff(hl, lg),
                   "router_weights_vs_tokenspeed": diff(hw, tw),
                   "full_mode_vs_tokenspeed": diff(fo, ts)}
            res[f"{L}/{p}"] = one
            acc(f"{tag}: checkpoint experts = TokenSpeed torch_block_scaled_moe_routes", one["oracle_vs_tokenspeed"]
                ["bit_equal"], **one["oracle_vs_tokenspeed"])
            acc(f"{tag}: picks and slot order = TokenSpeed torch_block_scaled_topk",
                bool(torch.equal(hid.to(torch.int32), tid)),
                rows_differing=int((hid.to(torch.int32) != tid).any(-1).sum()))
            acc(f"{tag}: full-mode experts within {a.quant_bound} (relative norm) of TokenSpeed's",
                one["full_mode_vs_tokenspeed"]["norm_rel"] < a.quant_bound, **one["full_mode_vs_tokenspeed"])
            print(f"{tag}: {json.dumps(one)}", flush=True)
        del raw, full
        gc.collect()
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- late layers


def check_layers(a, report: dict, acc: Acceptance):
    from transformers import Inkling2ForCausalLM
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    from transformers.models.inkling2 import modeling_inkling2 as M
    from transformers.models.inkling2.configuration_inkling2 import INKLING2_QUANTIZATION_CONFIG

    layers = sorted(ints(a.layers))
    base = 6 * ((layers[0] - 1) // 6)
    at = {L: L - base for L in layers}
    n = layers[-1] - base + 1
    cfg = R.text_config(a.model_dir)
    src = R.Source(a.model_dir)
    dev = torch.device("cuda:0")
    names = a.prompts.split(",")
    refs = {p: load_file(os.path.join(a.oracle, p + ".safetensors")) for p in names}
    for p, r in refs.items():
        if len(r["rows"]) != len(r["ids"]):
            raise SystemExit(f"{p}: the oracle kept {len(r['rows'])} of {len(r['ids'])} rows; this check needs all")
    k = cfg.n_routed_experts
    report.update(layers_at=at, model_layers=n)
    print("indices", at, flush=True)

    # the mini checkpoint, in the release's names
    path = os.path.join(a.work, "hf-late")
    os.makedirs(path, exist_ok=True)
    wm = {}
    t0 = time.time()
    for L in layers:
        p, q = f"{CK}layers.{L}.", f"{CK}layers.{at[L]}."
        ep = p + "mlp.experts."
        if R.expert_format(src, ep) != "mxfp8":
            raise SystemExit(f"layer {L}: Transformers loads only the MXFP8 experts (layers 40-47)")
        t = {q + name[len(p):]: src.get(name) for name in src.names(p)}
        t[f"{CK}attn_res.attn_queries.{at[L] - 1}"] = src.get(f"{CK}attn_res.attn_queries.{L - 1}")
        t[f"{CK}attn_res.mlp_queries.{at[L]}"] = src.get(f"{CK}attn_res.mlp_queries.{L}")
        fname = f"layer{L:02d}.safetensors"
        save_file({kk: v.contiguous() for kk, v in t.items()}, os.path.join(path, fname))
        wm.update({kk: fname for kk in t})
        print(f"wrote layer {L} as {at[L]}", flush=True)
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(path, "model.safetensors.index.json"), "w"))
    c = R.text_config(a.model_dir, n).to_dict()
    c.pop("transformers_version", None)
    for kk in [kk for kk in c if kk.startswith("_")]:
        c.pop(kk)
    if any(c["layer_types"][at[L]] != cfg.layer_types[L] for L in layers):
        raise SystemExit("layer types differ at the mapped indices")
    qc = dict(INKLING2_QUANTIZATION_CONFIG)
    qc["modules_to_not_convert"] = [x.replace("model.language_model.", "model.") for x in qc["modules_to_not_convert"]]
    c.update(model_type="inkling2_text", architectures=["Inkling2ForCausalLM"], torch_dtype="bfloat16",
             quantization_config=qc)
    json.dump(c, open(os.path.join(path, "config.json"), "w"), indent=1)
    report["quantization_config"] = qc
    print(f"mini checkpoint {time.time() - t0:.1f} s", flush=True)

    half = len(layers) // 2
    dm = {"model.embed_tokens": "cpu", "model.norm": "cpu", "model.final_input": "cpu", "lm_head": "cpu"}
    dm.update({f"model.layers.{i}": "cpu" for i in range(n)})
    dm.update({f"model.layers.{at[L]}": 0 if j < half else 1 for j, L in enumerate(layers)})
    model, info = Inkling2ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dm,
                                                      attn_implementation="eager", experts_implementation="eager",
                                                      output_loading_info=True)
    model.eval()
    li = {kk: sorted(map(str, v)) for kk, v in info.items()}
    report["loading_info"] = {kk: v[:50] for kk, v in li.items()}
    targets = tuple(f"model.layers.{at[L]}." for L in layers)
    acc("loader: no target-layer tensor missing", not [x for x in li["missing_keys"] if x.startswith(targets)],
        missing=[x for x in li["missing_keys"] if x.startswith(targets)][:20])
    for kk in ("unexpected_keys", "mismatched_keys", "error_msgs"):
        acc(f"loader {kk} empty", not li.get(kk), entries=li.get(kk, [])[:20])

    dq = Fp8Dequantize(None)
    mo = R.Model(a.model_dir, cfg, dev, "full")
    res = report.setdefault("forward", {})
    for L in layers:
        hl = model.model.layers[at[L]]
        d = hl.post_attention_layernorm.weight.device
        hsd = hl.state_dict()
        # parameters: Transformers' loader (conversion mapping, fp32 modules, de-interleave) against the oracle's
        mine = mo.layer(L, experts="lazy", lru=R.ExpertLRU(dev, 1)).state_dict()
        bad = []
        for kk, v in mine.items():
            if not bit_equal(hsd[kk].cpu(), v.cpu()):
                bad.append(f"{kk}: {hsd[kk].dtype} vs {v.dtype}")
        acc(f"layer {L}: non-expert parameters equal (names, dtypes, bits)", not bad, mismatches=bad[:20])
        report.setdefault("loaded_dtypes", {})[str(L)] = {kk: str(v.dtype) for kk, v in hsd.items()}
        del mine
        # experts: Transformers' tensors, decoded by Transformers, against the oracle's decode
        gu, dn = (torch.stack([dq._dequantize_one(hsd[f"mlp.experts.{t}"][j], hsd[f"mlp.experts.{t}_scale_inv"][j],
                                                  output_dtype=BF) for j in range(k)])
                  for t in ("gate_up_proj", "down_proj"))
        ep = f"{CK}layers.{L}.mlp.experts."
        bad = []
        for e in range(k):
            mgu = R.deinterleave(R.dequant_rows(src, ep, "w13", e, e + 1, dev))[0]
            mdn = R.dequant_rows(src, ep, "w2", e, e + 1, dev)[0]
            if not (bit_equal(gu[e].to(dev), mgu) and bit_equal(dn[e].to(dev), mdn)):
                bad.append(e)
        acc(f"layer {L}: all {k} experts as Transformers loads and decodes them = the oracle's decode",
            not bad, experts_differing=bad[:20])
        del hsd
        fp8 = hl.mlp.experts
        with torch.device("meta"):
            ex = M.Inkling2Experts(model.config)
        ex.load_state_dict({"gate_up_proj": gu, "down_proj": dn}, strict=True, assign=True)
        ex.eval()

        # forward on the oracle's inputs: Transformers' FP8Experts (reported), then its eager experts (accepted)
        with torch.no_grad():
            for p, r in refs.items():
                T = len(r["ids"])
                h = r[f"h.{L - 1}"][None].to(d)
                blocks = torch.stack([r["embed"]] + [r[f"h.{j - 1}"] for j in range(6, L, 6)], dim=-2)[None].to(d)
                pos = torch.arange(T, device=d)[None]
                mk = R.masks(model.config, h, pos)

                def call():
                    got = {}
                    hook = hl.mlp.gate.register_forward_hook(lambda mod, inp, out: got.update(o=out))
                    try:
                        out = hl(h, blocks, attention_mask=mk[hl.attention_type], conv_mask=mk["linear_attention"],
                                 past_key_values=None, position_ids=pos)
                    finally:
                        hook.remove()
                    return (out[0] if isinstance(out, tuple) else out), got["o"]

                one = {"blocks": blocks.shape[-2]}
                hl.mlp.experts = fp8
                try:
                    one["fp8_experts_h"] = diff(call()[0][0], r[f"h.{L}"])
                except Exception as exc:  # reported only
                    one["fp8_experts_h"] = f"{type(exc).__name__}: {exc}"
                hl.mlp.experts = ex
                out, (lg, w, idx) = call()
                order = idx.argsort(-1)
                one.update({"h": diff(out[0], r[f"h.{L}"]), "router_logits": diff(lg.cpu(), r[f"route.logits.{L}"]),
                            "ids_equal": bool(torch.equal(idx.gather(-1, order).cpu().to(torch.int32),
                                                          r[f"route.ids.{L}"])),
                            "slot_order_equal": bool(torch.equal(idx.cpu().to(torch.int32), r[f"route.slot_ids.{L}"]))
                            if f"route.slot_ids.{L}" in r else None,
                            "w": diff(w.gather(-1, order).float().cpu(), r[f"route.w.{L}"])})
                res[f"{L}/{p}"] = one
                tag = f"layer {L} (as {at[L]}, {one['blocks']} block states) {p}"
                acc(f"{tag}: output bit-equal to the oracle's h.{L}", one["h"]["bit_equal"], **one["h"])
                acc(f"{tag}: router logits bit-equal", one["router_logits"]["bit_equal"], **one["router_logits"])
                acc(f"{tag}: picks equal, weights bit-equal", one["ids_equal"] and one["w"]["bit_equal"],
                    w_max_abs=one["w"]["max_abs"])
                print(f"{tag}: {json.dumps(one)}", flush=True)
        hl.mlp.experts = fp8
        del ex, gu, dn
        gc.collect()
        torch.cuda.empty_cache()
    if not a.keep:
        shutil.rmtree(path)


# ----------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--work", required=True)
    ap.add_argument("--keep", action="store_true", help="keep the written checkpoints")
    sub = ap.add_subparsers(dest="check", required=True)
    t = sub.add_parser("truncated")
    t.add_argument("--layers", type=int, required=True)
    t.add_argument("--prompts", default="chat_short")
    t.add_argument("--subset", action="store_true")
    t.add_argument("--decode-check", type=int, default=0)
    t.add_argument("--decode-tol", type=float, default=0.25)
    w = sub.add_parser("weights")
    w.add_argument("--tokenspeed", required=True)
    w.add_argument("--layers", default="0,20,39,40,47")
    w.add_argument("--experts", default="0,1,97,191,290,383")
    e = sub.add_parser("experts")
    e.add_argument("--tokenspeed", required=True)
    e.add_argument("--layers", default="20,44")
    e.add_argument("--prompts", default="tool_call,long_1500")
    e.add_argument("--quant-bound", type=float, default=0.25)
    lay = sub.add_parser("layers")
    lay.add_argument("--oracle", required=True)
    lay.add_argument("--layers", default="40,41,42,43,44,45,46,47")
    lay.add_argument("--prompts", default="chat_short,tool_call")
    a = ap.parse_args()

    os.makedirs(a.work, exist_ok=True)
    report = {"check": a.check, "args": {k: v for k, v in vars(a).items() if k not in ("model_dir", "work")}}
    acc = Acceptance(report)
    t0 = time.time()
    try:
        {"truncated": check_truncated, "weights": check_weights, "experts": check_experts,
         "layers": check_layers}[a.check](a, report, acc)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        acc("check ran to completion", False, error=f"{type(exc).__name__}: {exc}")
    report["wall_s"] = round(time.time() - t0, 1)
    suffix = f"-L{a.layers}" if a.check == "truncated" else ""
    return finish(report, os.path.join(a.work, f"check-{a.check}{suffix}.json"))


if __name__ == "__main__":
    sys.exit(main())
