"""Where a layer's distance from the reference enters: each op of one layer on the same inputs, ours against torch
at fp32 reductions (the tight reference), torch at its default bf16 partial reduction (the oracle's), and fp64.
Also a bit-for-bit diff of two runner dumps (row invariance across --chunk).

USAGE = python tools/zig/inkling2_f64.py ops MODEL_DIR CAPTURE DUMP --layer L [--core-rows N] [--json OUT]
        python tools/zig/inkling2_f64.py same DUMP_A DUMP_B [--ops-too]

  ops   CAPTURE: inkling2_op_ref.py capture --fp32-reduction of layer L (one forward from position 0);
        DUMP: tf-inkling2-cuda --layers L:L+1 --feed CAPTURE --dump-ops (each of our ops ran on the capture's
        inputs, whatever --chunk). For every op output of layer L, all on the capture's inputs:
          ours~tight     ours against the capture
          default~tight  torch with the oracle's default reduction, recomputed here, against the capture
          ratio          ours~tight / default~tight (at or below 1: ours is no farther from the tight reference
                         than the oracle's own arithmetic)
        and for the ops with long sums (projections, attention core, output projection, router logits, experts)
        ours, tight and default against fp64 (the same op without intermediate roundings, or for the experts the
        module's own roundings with fp64 GEMMs), with bf16(fp64) as the rounding floor. The attention core's fp64
        runs on N rows spread over the prompt (default 192, plus the last row), the other ops on every row.
        Tight recomputed here must equal the capture bit for bit (checked; a mismatch is reported).
  same  Every tensor the two dumps share (h.*, route.*, final, logits.full; op.* outputs with --ops-too, never the
        "@" state reads, which belong to a forward), bit for bit. Exit 1 when any differs.

Transformers (Apache-2.0) is imported through inkling2_ref.py / inkling2_op_ref.py; this file copies none of it
(see THIRD_PARTY_NOTICES.md).
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BF = torch.bfloat16
ORDER = ["res.attn_input", "res.attn_norm", "res.q_norm", "res.k_norm", "res.v_norm", "attn.project", "conv.kv",
         "attn.heads", "attn.core", "attn.out", "conv.attn", "res.partial", "res.mlp_input", "res.mlp_norm",
         "moe.route", "moe.experts", "conv.mlp", "res.output"]


def rel(a, b):
    a, b = a.double(), b.double()
    n = float(b.norm())
    return float((a - b).norm()) / n if n > 0 else float((a - b).norm())


def equal(a, b):
    if a.dtype != b.dtype:
        return 0.0
    if a.dtype == BF:
        a, b = a.view(torch.int16), b.view(torch.int16)
    elif a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    return float((a == b).double().mean()) if a.numel() else 1.0


def f64_linear(mod, x):
    w = mod.weight.double()
    return F.linear(x.double(), w, None if mod.bias is None else mod.bias.double())


def f64_project(ly, ins):
    sa = ly.self_attn
    xq = ins.get("xq", ins.get("x"))
    xk = ins.get("xk", xq)
    xv = ins.get("xv", xq)
    out = {"q": f64_linear(sa.q_proj, xq), "k": f64_linear(sa.k_proj, xk), "v": f64_linear(sa.v_proj, xv),
           "rel": f64_linear(sa.r_proj, xq), "gate": f64_linear(sa.gate_proj, xq), "sink": f64_linear(sa.sink_proj, xq)}
    if not sa.is_sliding:
        out["dk"] = f64_linear(sa.k_sconv.weight_proj, xk)
        out["dv"] = f64_linear(sa.v_sconv.weight_proj, xv)
    return out


def f64_core(ly, ins, rows):
    """Attention with the relative bias, sink and output gate in fp64 without intermediate roundings, at `rows`."""
    sa = ly.self_attn
    q, k, v, relx = ins["q"], ins["k_ctx@0"], ins["v_ctx@0"], ins["rel"]
    sink, gate = ins["sink"], ins["gate"]
    hq = sink.shape[1]
    hk = k.shape[1] // 128
    proj = sa.rel_logits_proj.proj.double()  # [d_rel, extent]: zero bias at distances past the extent
    ext = proj.shape[-1]
    out = []
    for t in rows:
        lo = max(0, t - (sa.sliding_window - 1)) if sa.is_sliding else 0
        kk = k[lo:t + 1].double().view(-1, hk, 128).repeat_interleave(hq // hk, dim=1)
        vv = v[lo:t + 1].double().view(-1, hk, 128).repeat_interleave(hq // hk, dim=1)
        qq = q[t].double().view(hq, 128)
        rr = relx[t].double().view(hq, -1)
        dist = t - torch.arange(lo, t + 1, device=q.device)
        bias = (rr @ proj)[:, dist.clamp(max=ext - 1)] * (dist < ext).double()[None]
        s = torch.einsum("hd,chd->hc", qq, kk) / math.sqrt(128) + bias
        s = torch.cat([s, sink[t].double()[:, None]], 1)
        p = torch.softmax(s, -1)[:, :-1]
        o = torch.einsum("hc,chd->hd", p, vv).reshape(-1)
        out.append(o * torch.sigmoid(gate[t].double()))
    return torch.stack(out)


def ops(a):
    import inkling2_ref as ref
    from inkling2_moe_check import exact_sums
    from inkling2_op_ref import Ops

    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cap, dump = safe_open(a.capture, framework="pt"), safe_open(a.dump, framework="pt")
    cmeta, dmeta = cap.metadata() or {}, dump.metadata() or {}
    if cmeta.get("bf16_reduced_precision_reduction", "False") != "False":
        print("warning: the capture was not made with --fp32-reduction; 'tight' below is the capture's arithmetic")
    L = a.layer
    pre = f"op.{L}."
    names = {}
    for k in cap.keys():
        if not k.startswith(pre):
            continue
        g, o, d, t = k[len(pre):].split(".", 3)
        names.setdefault(f"{g}.{o}", {"in": [], "out": []})[d].append(t)
    dkeys = set(dump.keys())
    cfg = ref.text_config(a.model_dir, 0)
    model = ref.Model(a.model_dir, cfg, dev, "full")
    ly = model.layer(L)
    mods = Ops(cfg, ly, index=L)
    T = int(cmeta.get("rows", "0")) or cap.get_tensor(f"{pre}res.output.out.h").shape[0]
    core_rows = sorted(set(np.linspace(0, T - 1, a.core_rows).astype(int).tolist()) | {T - 1})
    kind = ("global" if not ly.self_attn.is_sliding else "sliding") + (" boundary" if ly.is_block_boundary else "")
    print(f"layer {L} ({kind}), {T} rows; capture {os.path.basename(a.capture)}, dump {os.path.basename(a.dump)}"
          f" (chunk {dmeta.get('chunk', '?')}, fed: {dmeta.get('feed', 'see the runner log')})")
    print(f"{'op output':26s} {'ours~tight':>10s} {'dflt~tight':>10s} {'ratio':>6s} {'eq(o,t)':>8s}"
          f" | {'ours~f64':>9s} {'tight~f64':>9s} {'dflt~f64':>9s} {'floor':>9s}")
    report = {"layer": L, "rows": T, "kind": kind, "ops": {}}
    for op in [o for o in ORDER if o in names] + sorted(set(names) - set(ORDER)):
        nm = names[op]
        if not nm["out"]:
            continue
        ins = {t: cap.get_tensor(f"{pre}{op}.in.{t}").to(dev) for t in nm["in"]}
        with torch.no_grad():
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            dflt = mods.run(op, ins, 0, 0 if op == "attn.core" else None)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            tight = mods.run(op, ins, 0, 0 if op == "attn.core" else None)
            exact, rows = {}, None
            if op == "attn.project":
                exact = f64_project(ly, ins)
            elif op == "attn.out":
                exact = {"ao": f64_linear(ly.self_attn.o_proj, ins["o"])}
            elif op == "moe.route":
                exact = {"logits": F.linear(ins["m"].double(), ly.mlp.gate.weight.double())}
            elif op == "moe.experts":
                exact = {"y": exact_sums(ly.mlp.experts, ins["m"], ins["ids"].long(), ins["w"].float()).double()}
            elif op == "attn.core":
                rows = torch.tensor(core_rows)
                exact = {"o": f64_core(ly, ins, core_rows)}
        for t in nm["out"]:
            key = f"{pre}{op}.out.{t}"
            want = cap.get_tensor(key)
            if key not in dkeys:
                print(f"{op + '.' + t:26s} (not in the dump)")
                continue
            ours = dump.get_tensor(key)
            d, tt = dflt[t].cpu(), tight[t].cpu()
            if ours.shape != want.shape:
                print(f"{op + '.' + t:26s} shape {tuple(ours.shape)} against the capture's {tuple(want.shape)}")
                continue
            r = {"ours_tight": rel(ours, want), "default_tight": rel(d, want), "eq_ours_tight": equal(ours, want),
                 "tight_recomputed_equal": equal(tt, want)}
            r["ratio"] = r["ours_tight"] / r["default_tight"] if r["default_tight"] > 0 else (
                0.0 if r["ours_tight"] == 0 else float("inf"))
            line = (f"{op + '.' + t:26s} {r['ours_tight']:10.3e} {r['default_tight']:10.3e} {r['ratio']:6.2f}"
                    f" {r['eq_ours_tight']:8.4f}")
            if t in exact:
                e = exact[t].cpu()
                sel = (lambda x: x[rows]) if rows is not None else (lambda x: x)  # noqa: E731
                floor = e.to(want.dtype).double() if want.dtype == BF else e.float().double()
                r |= {"ours_f64": rel(sel(ours), e), "tight_f64": rel(sel(want), e), "default_f64": rel(sel(d), e),
                      "floor_f64": rel(floor, e), "f64_rows": len(rows) if rows is not None else T}
                line += (f" | {r['ours_f64']:9.3e} {r['tight_f64']:9.3e} {r['default_f64']:9.3e} {r['floor_f64']:9.3e}"
                         + (f"  ({len(rows)} rows)" if rows is not None else ""))
            if r["tight_recomputed_equal"] < 1.0:
                line += f"  [tight recomputed here equals the capture in {r['tight_recomputed_equal']:.6f}]"
            print(line, flush=True)
            report["ops"][f"{op}.{t}"] = r
        del ins, dflt, tight, exact
        torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0


def same(a):
    fa, fb = safe_open(a.a, framework="pt"), safe_open(a.b, framework="pt")
    ma, mb = fa.metadata() or {}, fb.metadata() or {}
    print(f"{a.a}: chunk {ma.get('chunk')}; {a.b}: chunk {mb.get('chunk')}")
    n, bad = 0, 0
    for k in sorted(set(fa.keys()) & set(fb.keys())):
        if "@" in k or k in ("ids", "rows", "logits.rows"):
            continue
        if k.startswith("op.") and (not a.ops_too or ".out." not in k):
            continue
        x, y = fa.get_tensor(k), fb.get_tensor(k)
        if x.shape != y.shape:
            print(f"{k:40s} shape {tuple(x.shape)} vs {tuple(y.shape)}")
            bad += 1
            continue
        if x.dtype == BF:
            x, y = x.view(torch.int16), y.view(torch.int16)
        elif x.dtype == torch.float32:
            x, y = x.view(torch.int32), y.view(torch.int32)
        dif = (x != y)
        rows = int(dif.reshape(dif.shape[0], -1).any(-1).sum()) if dif.dim() else int(dif)
        n += 1
        bad += rows > 0
        if rows or a.verbose:
            print(f"{k:40s} {'equal' if rows == 0 else f'DIFFER in {rows} of {x.shape[0]} rows'}")
    print(f"{n} tensors compared; {bad} differ" + (" (BIT-EQUAL)" if bad == 0 else ""))
    return 1 if bad or n == 0 else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("ops")
    o.add_argument("model_dir")
    o.add_argument("capture")
    o.add_argument("dump")
    o.add_argument("--layer", type=int, required=True)
    o.add_argument("--core-rows", type=int, default=192)
    o.add_argument("--json", default="")
    o.add_argument("--device", default="cuda:0")
    s = sub.add_parser("same")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--ops-too", action="store_true")
    s.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    return {"ops": ops, "same": same}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
