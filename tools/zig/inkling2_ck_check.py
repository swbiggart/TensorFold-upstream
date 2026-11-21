"""Checkpoint-mode experts (cuda_moe.zig, Mode.checkpoint) against TokenSpeed's Torch reference on one layer.

USAGE = python tools/zig/inkling2_ck_check.py make SRC.st OUT.st --rows N
        python tools/zig/inkling2_ck_check.py check MODEL_DIR --layer L IN.st RUN.st [--json OUT.json]
        python tools/zig/inkling2_ck_check.py check MODEL_DIR --layer L ops RUN.st   (RUN: tf-inkling2-cuda --dump-ops)

  make   The first N `m` rows (mlp-norm outputs) of a tf-inkling2-moe input (inkling2_moe_check.py make), without
         torch's routes, so tf-inkling2-moe runs our router and our checkpoint-mode experts on its picks.
  check  tf-inkling2-moe --mode checkpoint's output (ids and w in slot order, y) against TokenSpeed's
         `torch_block_scaled_moe_routes` (inkling2_ref.py's port, fold in slot order) on the same m, ids and w, with
         the layer's checkpoint bytes. Also the same reference with its GEMMs summed in fp64 (rounded to fp32 before
         the scale, as the fp32 GEMM would be): the distance between the two references is the fp32-accumulation
         noise floor that ours is judged against.

Stats: relative norm ||a - b|| / ||b||, max abs, max abs / max |b|, bit-equal fraction, and the bf16 ulp histogram
(ordered distance; buckets 0, 1, 2, 3-4, 5-8, 9-16, >16).
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 1 << 62)]


def make(a):
    m = safe_open(a.src, framework="pt").get_tensor("m")[: a.rows].contiguous()
    save_file({"m": m}, a.out, metadata={"tool": "inkling2_ck_check.py make", "source": os.path.basename(a.src)})
    print(f"{a.out}: {m.shape[0]} rows of m")
    return 0


def ordered_bf16(x):
    v = x.contiguous().view(torch.int16).to(torch.int64)
    return torch.where(v < 0, -(v & 0x7FFF), v)


def stats(ours, ref):
    d = ours.double() - ref.double()
    nb = float(ref.double().norm())
    ulp = (ordered_bf16(ours) - ordered_bf16(ref)).abs()
    hist = [int(((ulp >= lo) & (ulp <= hi)).sum()) for lo, hi in BUCKETS]
    return {"rel": float(d.norm()) / nb if nb > 0 else float(d.norm()), "max_abs": float(d.abs().max()),
            "max_abs_over_max": float(d.abs().max() / ref.double().abs().max()),
            "bit_equal": float((ours.view(torch.int16) == ref.view(torch.int16)).double().mean()),
            "ulp_hist": hist, "nonfinite": int((~torch.isfinite(ours.float())).sum())}


def routes64(ref, x, fetch, weights, ids, formats, gate_cap, up_cap):
    """inkling2_ref._ts_block_scaled_moe_routes with each GEMM summed in fp64 then rounded to fp32."""
    first, first_scale, first_amax = ref._ts_reference_block_quantize(x, formats[0])
    decoded_x = ref._ts_decode_values(first, first_scale, formats[0])
    routes, hidden = ids.numel(), x.shape[1]
    output = torch.zeros((routes, hidden), dtype=torch.bfloat16, device=x.device)
    for expert in ids.unique().tolist():
        locations = (ids.flatten() == expert).nonzero().flatten()
        token_ids = locations // ids.shape[1]
        w13, s13, a13, d13, w2, s2, a2, d2 = fetch(expert)
        weight = ref._ts_decode_values(w13, s13, formats[0])
        gate_up = (decoded_x[token_ids].double() @ weight.double().T).float()
        if formats[0] == "nvfp4":
            gate_up = gate_up * ((first_amax[token_ids] * a13) * (1536.0 * d13[0]).reciprocal())[:, None]
        activated = ref._ts_capped_swiglu(gate_up.to(torch.bfloat16), weights.flatten()[locations], gate_cap, up_cap)
        values, scales, amax = ref._ts_reference_block_quantize(activated.contiguous(), formats[1])
        weight = ref._ts_decode_values(w2, s2, formats[1])
        down = (ref._ts_decode_values(values, scales, formats[1]).double() @ weight.double().T).float()
        if formats[1] == "nvfp4":
            down = down * ((amax * a2) * (1536.0 * d2[0]).reciprocal())[:, None]
        output[locations] = down.to(torch.bfloat16)
    rows = output.view(x.shape[0], ids.shape[1], hidden)
    folded = torch.zeros((x.shape[0], hidden), dtype=torch.float32, device=x.device)
    for slot in range(ids.shape[1]):
        folded += rows[:, slot].float()
    return folded.to(torch.bfloat16)


def check(a):
    import inkling2_ref as ref

    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = safe_open(a.run, framework="pt")
    meta = run.metadata() or {}
    if meta.get("mode") != "checkpoint" or meta.get("world", "1") != "1":
        raise SystemExit(f"{a.run}: mode {meta.get('mode')}, world {meta.get('world')}: want checkpoint at W = 1")
    if a.input == "ops":  # a tf-inkling2-cuda --dump-ops dump: the experts op's own inputs and output
        op = f"op.{a.layer}.moe.experts."
        m = run.get_tensor(op + "in.m").to(dev)
        ids = run.get_tensor(op + "in.ids").to(dev).to(torch.int64)
        w = run.get_tensor(op + "in.w").to(dev).float().contiguous()
        y = run.get_tensor(op + "out.y").to(dev)
    else:
        m = safe_open(a.input, framework="pt").get_tensor("m").to(dev)
        ids = run.get_tensor("ids").to(dev).to(torch.int64)
        w = run.get_tensor("w").to(dev).float().contiguous()
        y = run.get_tensor("y").to(dev)
    N = m.shape[0]
    if ids.shape != (N, 8) or y.shape != m.shape:
        raise SystemExit("input and run disagree on rows")
    cfg = ref.text_config(a.model_dir, 0)
    src = ref.Source(a.model_dir)
    prefix = f"model.llm.layers.{a.layer}.mlp.experts."
    fmt = ref.expert_format(src, prefix)
    names = ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]
    den = [None, None]
    if fmt == "nvfp4":
        names += ["w13_weight_amax", "w2_weight_amax"]
        den = [src.get(f"{prefix}{p}_weight_outer_scale_denominator", dev).float() for p in ("w13", "w2")]
    cache = {}

    def fetch(e):
        if e not in cache:
            cache[e] = {n: src.rows(prefix + n, e, e + 1, dev)[0] for n in names}
        t = cache[e]
        am = [t[f"{p}_weight_amax"].float() if fmt == "nvfp4" else None for p in ("w13", "w2")]
        return t["w13_weight"], t["w13_weight_scale"], am[0], den[0], t["w2_weight"], t["w2_weight_scale"], am[1], den[1]

    caps = (cfg.mlp_gate_softcap, cfg.mlp_up_softcap)
    y32 = ref._ts_block_scaled_moe_routes(m.contiguous(), fetch, w, ids.contiguous(), (fmt, fmt), *caps)
    y64 = routes64(ref, m.contiguous(), fetch, w, ids.contiguous(), (fmt, fmt), *caps)
    out = {"layer": a.layer, "format": fmt, "rows": N, "experts_used": len(cache),
           "ours_vs_ref": stats(y, y32), "ref64_vs_ref": stats(y64, y32), "ours_vs_ref64": stats(y, y64)}
    print(f"layer {a.layer} ({fmt}), {N} rows, {len(cache)} experts routed; our routes (slot order) fed to the reference")
    for k, label in (("ours_vs_ref", "ours  vs TokenSpeed fp32"), ("ref64_vs_ref", "ref64 vs TokenSpeed fp32"),
                     ("ours_vs_ref64", "ours  vs ref64")):
        s = out[k]
        print(f"  {label}: rel {s['rel']:.3e}  max_abs {s['max_abs']:.3e} ({s['max_abs_over_max']:.2e} of max)  "
              f"bit-equal {s['bit_equal']:.4f}  ulp {s['ulp_hist']}  nonfinite {s['nonfinite']}")
    r = out["ours_vs_ref"]["rel"] / max(out["ref64_vs_ref"]["rel"], 1e-30)
    out["ratio_to_floor"] = r
    print(f"  ours / floor (rel against TokenSpeed fp32): {r:.2f}")
    if a.perturb > 0:
        # how much the quantized experts amplify a small input difference (as two engines' attention outputs differ):
        # m with each element moved by a relative N(0, perturb), rounded to bf16
        g = torch.Generator(device=dev).manual_seed(1)
        mp = (m.float() * (1 + a.perturb * torch.randn(m.shape, generator=g, device=dev))).to(torch.bfloat16)
        yp = ref._ts_block_scaled_moe_routes(mp.contiguous(), fetch, w, ids.contiguous(), (fmt, fmt), *caps)
        rin = float((mp.double() - m.double()).norm() / m.double().norm())
        rout = float((yp.double() - y32.double()).norm() / y32.double().norm())
        out["perturb"] = {"input_rel": rin, "output_rel": rout, "gain": rout / max(rin, 1e-30)}
        print(f"  input moved by rel {rin:.3e} (bf16): reference output moves by rel {rout:.3e} (gain {rout / rin:.1f})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("src")
    m.add_argument("out")
    m.add_argument("--rows", type=int, default=128)
    c = sub.add_parser("check")
    c.add_argument("model_dir")
    c.add_argument("input")
    c.add_argument("run")
    c.add_argument("--layer", type=int, required=True)
    c.add_argument("--json", default="")
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--perturb", type=float, default=0.0, help="also: the reference's output change for input noise")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    return {"make": make, "check": check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
