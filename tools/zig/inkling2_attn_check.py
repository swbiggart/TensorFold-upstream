"""G-attn's checks: long-prompt captures, tf-inkling2-attn's outputs against torch, and row invariance.

USAGE = python tools/zig/inkling2_attn_check.py capture MODEL_DIR ORACLE_PROMPT --layers A:B --out CAPTURE [--rows T]
        python tools/zig/inkling2_attn_check.py compare MODEL_DIR CAPTURE ATTN_OUT [--json OUT]
        python tools/zig/inkling2_attn_check.py same DUMP_A DUMP_B [--ops attn.heads,attn.core]

  capture  inkling2_op_ref.py's capture for any prompt, also one whose oracle file keeps sampled rows (long_1500,
           long_4000): torch's chain starts from the normed embedding of the prompt's ids, runs layers 0 .. B-1 op by
           op, and keeps the ops of layers A .. B-1. Its layer outputs are checked against the oracle's kept rows.
  compare  tf-inkling2-attn's outputs (heads.<form>.<N>.*, core.<form>.<N>.o) against torch: attn.heads against the
           capture's outputs (recomputed with log scaling when the run was made with --pos-offset), attn.core against
           the capture's; max |a - b|, relative norm, bf16 ulp histogram. Then every form and forward size against the
           first, bit for bit: a row's outputs must not depend on its forward's size or the attention's form.
  same     Two tf-inkling2-cuda --dump-ops dumps of the same prompt (e.g. different --chunk), op outputs bit for bit.

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
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"capture": capture, "compare": compare, "same": same}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
