"""G-moe alone at any row count: inputs for tf-inkling2-moe from op captures, torch's answer, and the comparison.

USAGE = python tools/zig/inkling2_moe_check.py make MODEL_DIR --layer L --captures A.st[,B.st...] [--from-layers A:B]
            --rows N --out INPUT.st
        python tools/zig/inkling2_moe_check.py compare INPUT.st OUT.st [--tol-rel 0.01]
        python tools/zig/inkling2_moe_check.py same DUMP_A.st DUMP_B.st

  make     Gathers real `m` rows (torch's mlp-norm outputs, op.<L'>.moe.route.in.m of inkling2_op_ref.py captures;
           layer L' from --from-layers, default L alone; repeated in order until N rows) and runs layer L's
           moe.route and moe.experts on them in torch, as the per-op oracle does (inkling2_op_ref.Ops: Transformers'
           Inkling2TopKRouter and Inkling2Experts over the decoded bf16 weights, the full mode). Writes m, torch's
           routes (ids, w: tf-inkling2-moe runs its experts on them too) and ref.logits, ref.y.
  compare  tf-inkling2-moe's output against them: routes (rows whose 8 experts equal torch's), weights and logits
           (fp32: max abs, relative norm, ulp histogram), and the experts (bf16 ulp histogram) both on torch's routes
           (y_fed: the op on torch's exact inputs) and on ours (y, rows whose routes agree); the "floor" lines
           measure ours, torch, and torch without reduced-precision bf16 GEMM reduction against fp64-summed GEMMs.
  same     Two tf-inkling2-cuda --dump-ops dumps of one prompt at different --chunk: the moe op outputs, row
           for row (they must be bit-equal: a row does not depend on its forward).

Ulp buckets [0, 1, 2, 3-4, 5-8, 9-16, >16]: ordered distance in the tensor's own format (bf16 or fp32).
"""

import argparse
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 1 << 62)]


def make(a):
    import inkling2_ref as ref
    from inkling2_op_ref import Ops

    dev = torch.device(a.device)
    lo, hi = (int(x) for x in (a.from_layers or f"{a.layer}:{a.layer + 1}").split(":"))
    pool = []
    for path in a.captures.split(","):
        f = safe_open(path, framework="pt")
        keys = set(f.keys())
        for L in range(lo, hi):
            k = f"op.{L}.moe.route.in.m"
            if k in keys:
                pool.append(f.get_tensor(k))
    if not pool:
        raise SystemExit("no op.<L>.moe.route.in.m in the captures for those layers")
    rows = torch.cat(pool)
    reps = -(-a.rows // rows.shape[0])
    m = rows.repeat(reps, 1)[: a.rows].contiguous()
    print(f"{m.shape[0]} rows from {len(pool)} capture tensors ({rows.shape[0]} distinct)", flush=True)
    cfg = ref.text_config(a.model_dir, 0)
    model = ref.Model(a.model_dir, cfg, dev, "full")
    ly = model.layer(a.layer)
    ops = Ops(cfg, ly, index=a.layer)
    with torch.no_grad():
        r = ops.run("moe.route", {"m": m.to(dev)}, 0)
        y = ops.run("moe.experts", {"m": m.to(dev), "ids": r["ids"], "w": r["w"]}, 0)["y"]
        ideal = exact_sums(ly.mlp.experts, m.to(dev), r["ids"].long(), r["w"].float())
        # torch's default lets cuBLAS reduce bf16 GEMMs in reduced precision (split K); the same op without it
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        y32 = ops.run("moe.experts", {"m": m.to(dev), "ids": r["ids"], "w": r["w"]}, 0)["y"]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    out = {"m": m, "ids": r["ids"].cpu().contiguous(), "w": r["w"].cpu().contiguous(),
           "ref.logits": r["logits"].cpu().contiguous(), "ref.y": y.cpu().contiguous(),
           "ideal.y": ideal.cpu().contiguous(), "ref32.y": y32.cpu().contiguous()}
    save_file(out, a.out, metadata={"tool": "inkling2_moe_check.py make", "layer": str(a.layer),
                                    "captures": ",".join(os.path.basename(p) for p in a.captures.split(",")),
                                    "from_layers": f"{lo}:{hi}"})
    print(f"wrote {a.out}")
    return 0


# exact_sums: adapted from huggingface/transformers src/transformers/models/inkling2/modeling_inkling2.py
# (Inkling2Experts.forward) at 109a7d19, Copyright 2026 the HuggingFace Team, Apache License 2.0; modified for
# TensorFold: both GEMMs in fp64 (see THIRD_PARTY_NOTICES.md).
def exact_sums(ex, m, ids, w):
    """Inkling2Experts' forward with both GEMMs summed in fp64 and rounded once to bf16 (the same decoded weights,
    activation and fp32 expert-id-order sum): the bf16-rounding floor any summation order is measured against."""
    out = torch.zeros(m.shape, dtype=torch.float32, device=m.device)
    for e in sorted(set(ids.flatten().tolist())):
        tok, slot = torch.where(ids == e)
        gu = (m[tok].double() @ ex.gate_up_proj[e].double().T).to(m.dtype)
        gate, up = gu.chunk(2, dim=-1)
        act = ex.act_fn(gate, up)
        d = (act.double() @ ex.down_proj[e].double().T).to(m.dtype)
        out.index_add_(0, tok, d * w[tok, slot, None])
    return out.to(m.dtype)


def ordered_f32(x):
    b = x.astype(np.float32).view(np.int32).astype(np.int64)
    return np.where(b < 0, -(b & 0x7FFFFFFF), b)


def ordered_bf16(x):
    b = x.view(np.uint16).astype(np.int64)
    return np.where(b & 0x8000, -(b & 0x7FFF), b)


def report(name, ours, want, kind):
    if kind == "bf16":
        a, b = ours.view(torch.int16).numpy().view(np.uint16), want.view(torch.int16).numpy().view(np.uint16)
        fa, fb = ours.float().numpy(), want.float().numpy()
        ulp = np.abs(ordered_bf16(a) - ordered_bf16(b))
    else:
        fa, fb = ours.numpy().astype(np.float32), want.numpy().astype(np.float32)
        ulp = np.abs(ordered_f32(fa) - ordered_f32(fb))
    d = fa.astype(np.float64) - fb.astype(np.float64)
    nb = float(np.linalg.norm(fb.astype(np.float64)))
    rel = float(np.linalg.norm(d) / nb) if nb > 0 else float(np.linalg.norm(d))
    hist = [int(((ulp >= lo) & (ulp <= hi)).sum()) for lo, hi in BUCKETS]
    nonfinite = int((~np.isfinite(fa)).sum())
    print(f"  {name:30s} rel {rel:.3e}  max_abs {float(np.abs(d).max()) if d.size else 0.0:.3e}  equal "
          f"{float((ulp == 0).mean()) if ulp.size else 1.0:.6f}  ulps[0,1,2,3-4,5-8,9-16,>16] {','.join(map(str, hist))}"
          + (f"  NONFINITE {nonfinite}" if nonfinite else ""))
    return rel, nonfinite


def compare(a):
    inp, out = safe_open(a.input, framework="pt"), safe_open(a.out, framework="pt")
    meta = out.metadata() or {}
    ids_t, ids_o = inp.get_tensor("ids"), out.get_tensor("ids")
    N = ids_t.shape[0]
    print(f"layer {meta.get('layer')} ({meta.get('format')}), {N} rows, forward size {meta.get('chunk')}")
    same = torch.tensor([set(x) == set(y) for x, y in zip(ids_o.tolist(), ids_t.tolist())])
    exact = (ids_o == ids_t).all(1)
    print(f"  routes: rows whose 8 experts equal torch's {int(same.sum())} of {N} ({float(same.float().mean()):.6f}),"
          f" same order {int(exact.sum())}")
    bad = []
    # the router is gated on its own: rows dropped from the expert compares below must not hide wrong picks
    if N == 0:
        bad.append("no rows")
    elif int((~same).sum()) > a.tol_route * N:
        bad.append(f"routes ({int((~same).sum())} rows pick other experts)")
    for name, ours, want, kind, rows in [
            ("route.logits", out.get_tensor("logits"), inp.get_tensor("ref.logits"), "f32", None),
            ("route.w (equal-route rows)", out.get_tensor("w"), inp.get_tensor("w"), "f32", exact),
            ("experts.y on torch's routes", out.get_tensor("y_fed"), inp.get_tensor("ref.y"), "bf16", None),
            ("experts.y on ours (equal rows)", out.get_tensor("y"), inp.get_tensor("ref.y"), "bf16", same),
            ("floor: ours vs fp64 sums", out.get_tensor("y_fed"), inp.get_tensor("ideal.y"), "bf16", None),
            ("floor: torch vs fp64 sums", inp.get_tensor("ref.y"), inp.get_tensor("ideal.y"), "bf16", None),
            ("floor: torch fp32-reduce vs fp64", inp.get_tensor("ref32.y"), inp.get_tensor("ideal.y"), "bf16", None)]:
        if rows is not None:
            ours, want = ours[rows], want[rows]
        if ours.numel() == 0:
            print(f"  {name:30s} NO ROWS TO COMPARE")
            if not name.startswith("floor"):
                bad.append(f"{name} (empty)")
            continue
        rel, nf = report(name, ours, want, kind)
        if rel > a.tol_rel or nf:
            bad.append(name)
    print("FAIL: " + ", ".join(bad) if bad else "PASS (within tolerance)")
    return 1 if bad else 0


def same(a):
    """Two tf-inkling2-cuda --dump-ops dumps of one prompt (different --chunk): the G-moe op outputs row for row."""
    fa, fb = safe_open(a.dump_a, framework="pt"), safe_open(a.dump_b, framework="pt")
    ka = {k for k in fa.keys() if ".moe." in k and ".out." in k and "@" not in k}
    kb = {k for k in fb.keys() if ".moe." in k and ".out." in k and "@" not in k}
    ma, mb = fa.metadata() or {}, fb.metadata() or {}
    print(f"{a.dump_a}: chunk {ma.get('chunk')}; {a.dump_b}: chunk {mb.get('chunk')}; {len(ka & kb)} moe outputs in both")
    done = min(int(ma.get("rows_done", 1 << 30)), int(mb.get("rows_done", 1 << 30)))
    bad = 0
    if not ka or ka != kb:
        print(f"FAIL: coverage: {len(ka)} and {len(kb)} moe outputs, {len(ka ^ kb)} in only one dump")
        return 1
    if done <= 0:
        print("FAIL: no completed rows")
        return 1
    for k in sorted(ka & kb, key=lambda s: (int(s.split(".")[1]), s)):
        x, y = fa.get_tensor(k)[:done], fb.get_tensor(k)[:done]
        if x.dtype == torch.bfloat16:
            x, y = x.view(torch.int16), y.view(torch.int16)
        rows = int((x != y).any(dim=-1).sum())
        bad += rows
        print(f"  {k:34s} rows differing {rows} of {x.shape[0]}")
    print("BIT-EQUAL" if bad == 0 else f"DIFFER: {bad} rows")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("same")
    s.add_argument("dump_a")
    s.add_argument("dump_b")
    m = sub.add_parser("make")
    m.add_argument("model_dir")
    m.add_argument("--layer", type=int, required=True)
    m.add_argument("--captures", required=True)
    m.add_argument("--from-layers", default="")
    m.add_argument("--rows", type=int, default=2048)
    m.add_argument("--out", required=True)
    m.add_argument("--device", default="cuda:0")
    c = sub.add_parser("compare")
    c.add_argument("input")
    c.add_argument("out")
    c.add_argument("--tol-rel", type=float, default=1e-2)
    c.add_argument("--tol-route", type=float, default=0.0, help="allowed fraction of rows that pick other experts")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"make": make, "compare": compare, "same": same}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
