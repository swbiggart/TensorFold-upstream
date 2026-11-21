"""G-attn's checks: long-prompt captures, tf-inkling2-attn's outputs against torch, and row invariance.

USAGE = python tools/zig/inkling2_attn_check.py capture MODEL_DIR ORACLE_PROMPT --layers A:B --out CAPTURE [--rows T]
        python tools/zig/inkling2_attn_check.py compare MODEL_DIR CAPTURE ATTN_OUT [--json OUT]
        python tools/zig/inkling2_attn_check.py same DUMP_A DUMP_B [--ops attn.heads,attn.core]
        python tools/zig/inkling2_attn_check.py core-inputs MODEL_DIR CAPTURE --layer L --out FILE
        python tools/zig/inkling2_attn_check.py hist CORE_INPUTS ATTN_OUT [--json OUT] [--device cuda:0]

  capture  inkling2_op_ref.py's capture for any prompt, also one whose oracle file keeps sampled rows (long_1500,
           long_4000): torch's chain starts from the normed embedding of the prompt's ids, runs layers 0 .. B-1 op by
           op, and keeps the ops of layers A .. B-1. Its layer outputs are checked against the oracle's kept rows.
  compare  tf-inkling2-attn's outputs (heads.<form>.<N>.*, core.<form>.<N>.o) against torch: attn.heads against the
           capture's outputs (recomputed with log scaling when the run was made with --pos-offset), attn.core against
           the capture's; max |a - b|, relative norm, bf16 ulp histogram. Then every form and forward size against the
           first, bit for bit: a row's outputs must not depend on its forward's size or the attention's form.
  same     Two tf-inkling2-cuda --dump-ops dumps of the same prompt (e.g. different --chunk), op outputs bit for bit.
  core-inputs  Layer L's attention core inputs from a capture of this tool or inkling2_op_ref.py (q'', r'', sink,
           gate and the keys and values from position 0, as attn.heads and attn.project made them), its output o,
           and the layer's rel_logits_proj.proj from the checkpoint, in one file with the layer kind in its
           metadata: the input of tf-inkling2-attn --oracle --capture (no model needed there).
  hist     The attention core's error on a real layer, in head-scale steps (information beside the
           float64 range check): every form in ATTN_OUT (tf-inkling2-attn MODEL_DIR CAPTURE ATTN_OUT --layer L --forms
           auto,kx: core.<form>.<N>.o; auto is v0.1's core, "v01"), torch's eager output (the core-inputs file's o,
           v0.1's oracle arithmetic) and the fp32 form recomputed here (inkling2_ref.kx_core, v0.2's oracle), each
           against float64 and against the fp32 form. float64 = the core in float64 on the same bf16 inputs and the
           same bf16-rounded bias (the kernels' bias table, relbias_attention_ref.zig's point), sink in the
           denominator. A head-scale step is one bf16 step at H = max over the head's columns of sigmoid(gate_c) *
           sum_j p_j |v_jc| / sum p (float64, sink in sum p): an output's distance / that step. Prints p50, p99, max
           and the share at <= 0.5, 1, 2, 4, 16 steps, per form and reference; forms' runs at different forward sizes
           must be byte-equal (checked). TF32 off (inkling2_ref.tf32_guard).

Transformers (Apache-2.0) is imported through inkling2_ref.py / inkling2_op_ref.py; this file copies none of it
(see THIRD_PARTY_NOTICES.md).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling2_ref as ref  # noqa: E402
import inkling2_op_ref as opr  # noqa: E402
from inkling2_compare import line, stats  # noqa: E402

BF = torch.bfloat16
WINDOW = 1024  # a sliding layer's keys: distance 0 .. 1023
ORDER = ["project", "heads", "core", "out"]  # tf-inkling2-attn's ops, in forward order


def bits(t):
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def stat(ours, want):
    if ours.dtype == BF:
        return stats(bits(ours), bits(want), "bf16")
    return stats(ours.float().numpy(), want.float().numpy(), "float32")


# ----------------------------------------------------------------------------------------------- capture


class Kept(opr.Capture):
    """A capture that keeps only the ops of some layers."""

    def __init__(self, dev, keep):
        super().__init__(dev)
        self.keep = keep

    def put(self, scope, op, d, ts):
        if scope in self.keep:
            super().put(scope, op, d, ts)


def capture(a):
    dev = torch.device(a.device)
    lo, hi = (int(x) for x in a.layers.split(":"))
    cfg = ref.text_config(a.model_dir, 0)
    model = ref.Model(a.model_dir, cfg, dev, "full")
    o = safe_open(a.oracle, framework="pt")
    ids = o.get_tensor("ids")
    T = ids.shape[0] if a.rows == 0 else min(a.rows, ids.shape[0])
    kept = o.get_tensor("rows")
    kept = kept[kept < T].to(dev)
    with torch.no_grad():
        emb = model.embed()
        h = emb(ids[:T].to(dev)[None])[0]
        del emb
    opr.equal_report("embed (kept rows)", h[kept], o.get_tensor("embed").to(dev)[: kept.shape[0]])
    cap = Kept(dev, {str(L) for L in range(lo, hi)})
    blocks = []
    for L in range(hi):
        t0 = time.time()
        ly = model.layer(L)
        t1 = time.time()
        with torch.no_grad():
            h, blocks, r = opr.layer_chain(cap, opr.Ops(cfg, ly, index=L), L, h, blocks, cfg)
        torch.cuda.synchronize()
        print(f"layer {L}: load {t1 - t0:.1f} s, ops {time.time() - t1:.1f} s; kept rows against the oracle:", flush=True)
        opr.equal_report(f"h.{L}", h[kept], o.get_tensor(f"h.{L}").to(dev)[: kept.shape[0]])
        del ly
        torch.cuda.empty_cache()
    meta = {"tool": "inkling2_attn_check.py capture", "oracle": os.path.basename(a.oracle), "layers": a.layers,
            "rows": str(T)}
    save_file(cap.t, a.out, metadata=meta)
    print(f"{len(cap.t)} tensors to {a.out}")
    return 0


# ----------------------------------------------------------------------------------------------- compare


def compare(a):
    f = safe_open(a.out, framework="pt")
    meta = f.metadata() or {}
    L = int(meta["layer"])
    T = int(meta["rows"])
    P = int(meta.get("pos_offset", "0"))
    c = safe_open(a.capture, framework="pt")
    cap = lambda n: c.get_tensor(f"op.{L}.attn.{n}")[:T]  # noqa: E731
    names = sorted({(k.split(".")[0], k.split(".")[-1]) for k in f.keys()}, key=lambda x: (ORDER.index(x[0]), x[1]))
    want = {f"{g}.{t}": cap(f"{g}.out.{t}") for g, t in names}
    if P:
        # torch's heads at the run's positions (log scaling of global layers past 128,000)
        dev = torch.device(a.device)
        cfg = ref.text_config(a.model_dir, 0)
        model = ref.Model(a.model_dir, cfg, dev, "full")
        ops = opr.Ops(cfg, model.layer(L), index=L)
        ins = {t: cap(f"heads.in.{t}").to(dev) for t in ("q", "kc", "vc", "rel")}
        with torch.no_grad():
            got = ops.run("attn.heads", ins, P)
        want |= {f"heads.{t}": v.cpu() for t, v in got.items()}
        if not ops.ly.self_attn.is_sliding:
            unscaled = cap("heads.out.q")
            print(f"log scaling at positions {P}..{P + T - 1}: torch's q' differs from position 0's in "
                  f"{int((bits(want['heads.q']) != bits(unscaled)).sum())} of {unscaled.numel()} values")
    runs = sorted({k.rsplit(".", 1)[0].split(".", 1)[1] for k in f.keys()}, key=lambda r: (r.split(".")[0], int(r.split(".")[1])))
    print(f"layer {L}, {T} rows at positions {P}..{P + T - 1}; runs (form.forward size): {', '.join(runs)}")
    res = {"layer": L, "rows": T, "pos_offset": P, "torch": {}, "invariance": {}}
    first = runs[0]
    bad, compared = 0, 0
    for r in runs:
        for g, op in names:
            t = f"{g}.{op}"
            if f"{g}.{r}.{op}" not in f.keys():
                continue
            ours = f.get_tensor(f"{g}.{r}.{op}")
            s = stat(ours, want[t])
            res["torch"][f"{r}.{t}"] = s
            print(line(f"{t} {r} vs torch", s))
            compared += 1
            # numbers, not only invariance: a kernel that is wrong the same way at every size must fail here
            if s.get("nonfinite") or ("rel" in s and s["rel"] > a.tol_rel) or ("rel" not in s and s.get("differ")):
                print(f"    PAST TOLERANCE vs torch (rel <= {a.tol_rel:g})")
                bad += 1
            if r != first:
                ref0 = f.get_tensor(f"{g}.{first}.{op}")
                diff = int((bits(ours) != bits(ref0)).sum())
                res["invariance"][f"{r}.{t}"] = diff
                if diff:
                    rows = np.nonzero((bits(ours) != bits(ref0)).any(axis=1))[0]
                    print(f"    NOT ROW-INVARIANT: {diff} values differ from {first}, rows {rows[:10].tolist()}...")
                    bad += 1
    print(f"bit equality across {len(runs)} runs (forms x forward sizes): "
          + ("all equal" if all(v == 0 for v in res["invariance"].values()) else "DIFFER"))
    if not compared:
        print("FAIL: no op output was compared with torch")
        bad += 1
    if len(runs) < 2:
        print("FAIL: row invariance needs at least two runs (forms x forward sizes)")
        bad += 1
    res["pass"] = not bad
    print("PASS" if not bad else f"FAIL: {bad} problem(s)")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 1 if bad else 0


# ----------------------------------------------------------------------------------------------- same


def same(a):
    fa, fb = safe_open(a.a, framework="pt"), safe_open(a.b, framework="pt")
    want = set(a.ops.split(",")) if a.ops else None
    n, bad = 0, 0
    for k in sorted(set(fa.keys()) & set(fb.keys())):
        p = k.split(".")
        if len(p) < 6 or p[0] != "op" or p[4] != "out" or "@" in k:
            continue
        if want and f"{p[2]}.{p[3]}" not in want:
            continue
        x, y = fa.get_tensor(k), fb.get_tensor(k)
        if x.dtype == BF:
            d = int((bits(x) != bits(y)).sum())
        else:
            d = int((x.view(torch.int32) != y.view(torch.int32)).sum()) if x.dtype in (torch.float32, torch.int32) else int((x != y).sum())
        n += 1
        bad += d > 0
        print(f"{k:40s} {'equal' if d == 0 else f'DIFFER {d} of {x.numel()}'}")
    print(f"{n} op outputs compared; {bad} differ")
    return 1 if bad or n == 0 else 0


# ----------------------------------------------------------------------------------------------- core-inputs


CORE_IN = ("q", "rel", "sink", "gate", "k_ctx@0", "v_ctx@0")


def core_inputs(a):
    c = safe_open(a.capture, framework="pt")
    meta = c.metadata() or {}
    L = a.layer
    pre = f"op.{L}.attn.core"
    if f"{pre}.in.q" not in c.keys():
        raise SystemExit(f"{a.capture} has no layer {L} attention core")
    T = c.get_slice(f"{pre}.in.q").get_shape()[0]
    out = {f"{pre}.in.{n}": c.get_tensor(f"{pre}.in.{n}").contiguous() for n in CORE_IN}
    out[f"{pre}.out.o"] = c.get_tensor(f"{pre}.out.o").contiguous()
    for k, t in out.items():
        if t.dtype != BF or t.dim() != 2 or t.shape[0] != T:
            raise SystemExit(f"{k}: {t.dtype} {tuple(t.shape)}, expected bf16 [{T}, ...]")
    proj = ref.Source(a.model_dir).get(f"model.llm.layers.{L}.attn.rel_logits_proj.proj")
    if proj.dtype != BF or tuple(proj.shape) != (16, 1024):
        raise SystemExit(f"rel_logits_proj.proj of layer {L}: {proj.dtype} {tuple(proj.shape)}, expected bf16 [16, 1024]")
    out[f"{pre}.in.proj"] = proj.contiguous()
    local = json.load(open(os.path.join(a.model_dir, "config.json")))["text_config"]["local_layer_ids"]
    kind = "sliding" if L in local else "global"
    save_file(out, a.out, metadata={"tool": "inkling2_attn_check.py core-inputs", "capture": os.path.basename(a.capture),
                                     "oracle": meta.get("oracle", ""), "layer": str(L), "kind": kind, "rows": str(T)})
    print(f"layer {L} ({kind}), {T} rows: {len(out)} tensors to {a.out}")
    return 0


# ----------------------------------------------------------------------------------------------- hist

EDGES = (0.5, 1, 2, 4, 16)
SCALE32 = float(np.float32(128 ** -0.5))  # the kernels' fp32 scale (0.08838834764831845f)
NAMES = {"auto": "v01", "fused": "v01-fused", "split": "v01-split", "kx": "kx"}


def hist(a):
    dev = torch.device(a.device)
    ref.tf32_guard(dev)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    c = safe_open(a.inputs, framework="pt")
    meta = c.metadata() or {}
    L, kind = int(meta["layer"]), meta["kind"]
    pre = f"op.{L}.attn.core"
    get = lambda n: c.get_tensor(f"{pre}.{n}").to(dev)  # noqa: E731
    q, rel, sink, gate = get("in.q"), get("in.rel"), get("in.sink"), get("in.gate")
    k, v, proj, eager = get("in.k_ctx@0"), get("in.v_ctx@0"), get("in.proj"), get("out.o")
    T, hq = sink.shape
    hk = k.shape[1] // 128
    ext = proj.shape[1]
    f = safe_open(a.out, framework="pt")
    runs = {}
    for key in sorted(f.keys()):
        p = key.split(".")
        if p[0] == "core" and p[-1] == "o":
            runs.setdefault(p[1], []).append((int(p[2]), key))
    if not runs:
        raise SystemExit(f"{a.out} holds no core.<form>.<N>.o")
    forms, invariance = {}, {}
    for form, items in sorted(runs.items()):
        items.sort()
        first = f.get_tensor(items[0][1])
        for N, key in items[1:]:
            d = int((bits(f.get_tensor(key)) != bits(first)).sum())
            invariance[f"{form}.{N} vs {form}.{items[0][0]}"] = d
        forms[NAMES.get(form, form)] = first.to(dev)
    R = min([T] + [t.shape[0] for t in forms.values()])
    forms["torch-eager"] = eager
    forms["torch-fp32"] = None  # computed per block
    refs = ("float64", "torch-fp32")
    errs = {(fm, r): [] for fm in forms for r in refs if fm != r}
    zero_h = 0
    g = hq // hk
    for r0 in range(0, R, a.block):
        r1 = min(R, r0 + a.block)
        lo = max(0, r0 - (WINDOW - 1)) if kind == "sliding" else 0
        rows = torch.arange(r0, r1, device=dev)
        keys = torch.arange(lo, r1, device=dev)
        dist = rows[:, None] - keys[None, :]
        allowed = dist >= 0
        if kind == "sliding":
            allowed &= dist <= WINDOW - 1
        qb = q[r0:r1].view(-1, hq, 128).transpose(0, 1)  # [hq, Rb, D]
        kb = k[lo:r1].view(-1, hk, 128).transpose(0, 1).repeat_interleave(g, dim=0)
        vb = v[lo:r1].view(-1, hk, 128).transpose(0, 1).repeat_interleave(g, dim=0)
        rl = rel[r0:r1].view(-1, hq, proj.shape[0]).transpose(0, 1)  # [hq, Rb, d_rel]
        table = torch.matmul(rl, proj)  # bf16 [hq, Rb, extent], fp32 sums: the kernels' bf16 bias table
        bias = table.gather(-1, dist.clamp(0, ext - 1).expand(hq, -1, -1))
        bias = bias.masked_fill((dist < 0) | (dist >= ext), 0.0)
        sk = sink[r0:r1].transpose(0, 1)[..., None]  # [hq, Rb, 1]
        gt = gate[r0:r1].view(-1, hq, 128).transpose(0, 1)  # [hq, Rb, D]
        y, phi = ref.kx_core(qb[None], kb[None], vb[None], bias[None], allowed[None, None], SCALE32, sk[None])
        forms["torch-fp32"] = ref.kx_output(y[0].transpose(0, 1).reshape(r1 - r0, -1), gate[r0:r1],
                                            phi[0].transpose(0, 1))
        s = torch.matmul(qb.double(), kb.double().transpose(-1, -2)) * SCALE32 + bias.double()
        s = s.masked_fill(~allowed, float("-inf"))
        m = torch.maximum(s.amax(-1, keepdim=True), sk.double())
        w = torch.exp(s - m)
        den = w.sum(-1, keepdim=True) + torch.exp(sk.double() - m)
        sg = torch.sigmoid(gt.double())
        point = (torch.matmul(w, vb.double()) / den) * sg  # [hq, Rb, D]
        H = ((torch.matmul(w, vb.double().abs()) / den) * sg).amax(-1, keepdim=True)
        step = torch.exp2(torch.floor(torch.log2(H)) - 7)
        zero_h += int((H == 0).sum())
        step = torch.where(H > 0, step, torch.full_like(step, float("nan")))
        want = {"float64": point,
                "torch-fp32": forms["torch-fp32"].view(-1, hq, 128).transpose(0, 1).double()}
        for (fm, rf) in errs:
            got = (forms[fm] if fm == "torch-fp32" else forms[fm][r0:r1]).view(-1, hq, 128).transpose(0, 1).double()
            e = ((got - want[rf]).abs() / step).flatten()
            errs[(fm, rf)].append(e[~torch.isnan(e)].float().cpu())
    res = {"layer": L, "kind": kind, "rows": R, "heads": [hq, hk], "invariance": invariance,
           "zero_head_scale_outputs": zero_h, "stats": {}}
    print(f"HIST layer {L} ({kind}), {R} rows, heads {hq}/{hk}; forms: {', '.join(forms)}; runs byte-equal across "
          f"forward sizes: {'yes' if all(d == 0 for d in invariance.values()) else 'NO ' + str(invariance)}")
    for (fm, rf), parts in errs.items():
        e = torch.cat(parts).numpy()
        st = {"n": int(e.size), "p50": float(np.percentile(e, 50)), "p99": float(np.percentile(e, 99)),
              "max": float(e.max()), "le": {str(x): float((e <= x).mean()) for x in EDGES}}
        res["stats"][f"{fm} vs {rf}"] = st
        print(f"HIST L{L} {fm:11s} vs {rf:10s}: p50 {st['p50']:.3f} p99 {st['p99']:.3f} max {st['max']:.3f} "
              f"head-scale steps; " + " ".join(f"<={x}: {100 * st['le'][str(x)]:.2f}%" for x in EDGES))
    k64, v64 = res["stats"].get("kx vs float64"), res["stats"].get("v01 vs float64")
    if k64 and v64:
        print(f"HIST L{L} line: kx vs float64 p50 {k64['p50']:.3f} p99 {k64['p99']:.3f} max {k64['max']:.3f}; v01 "
              f"p50 {v64['p50']:.3f} p99 {v64['p99']:.3f} max {v64['max']:.3f} (head-scale steps); kx "
              f"{'no further' if k64['p50'] <= v64['p50'] and k64['p99'] <= v64['p99'] else 'FURTHER'} than v01 at "
              f"p50 and p99")
    bad = any(d for d in invariance.values())
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("model_dir")
    c.add_argument("oracle")
    c.add_argument("--layers", required=True, help="A:B")
    c.add_argument("--out", required=True)
    c.add_argument("--rows", type=int, default=0)
    c.add_argument("--device", default="cuda:0")
    k = sub.add_parser("compare")
    k.add_argument("model_dir")
    k.add_argument("capture")
    k.add_argument("out")
    k.add_argument("--json", default="")
    k.add_argument("--device", default="cuda:0")
    k.add_argument("--tol-rel", type=float, default=1e-2, help="largest relative norm error of an op against torch")
    s = sub.add_parser("same")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--ops", default="")
    ci = sub.add_parser("core-inputs")
    ci.add_argument("model_dir")
    ci.add_argument("capture")
    ci.add_argument("--layer", type=int, required=True)
    ci.add_argument("--out", required=True)
    h = sub.add_parser("hist")
    h.add_argument("inputs", help="a core-inputs file (this tool's core-inputs)")
    h.add_argument("out", help="tf-inkling2-attn's output for the same capture and layer")
    h.add_argument("--json", default="")
    h.add_argument("--block", type=int, default=128, help="rows a step")
    h.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"capture": capture, "compare": compare, "same": same, "core-inputs": core_inputs, "hist": hist}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
