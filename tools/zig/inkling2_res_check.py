"""Row-invariance and precision checks for tf-inkling2-cuda's op dumps (the G-res and G-head lanes' evidence).

USAGE = python tools/zig/inkling2_res_check.py ulps DUMP CAPTURE [--groups res,head]
        python tools/zig/inkling2_res_check.py rows DUMP DUMP... [--groups res,head]
        python tools/zig/inkling2_res_check.py tile ORACLE_PROMPT CAPTURE --rows N --oracle-out O --capture-out C
                                                  [--layers A:B] [--head]

  ulps  Every op output of the named groups in DUMP (tf-inkling2-cuda --dump-ops) against the same tensor of an
        inkling2_op_ref.py capture: max |a - b|, ||a - b|| / ||b||, and an ulp histogram in the output's own type
        (fp32 outputs: fp32 ulps, and the bf16 ulps of both sides rounded to bf16, which is what the next op reads).
        Ops the dump marks as fed stubs are skipped.
  rows  The same op outputs (and h.<L>, embed, final, logits.full) in several dumps of one prompt made with different
        --chunk sizes: every row must be bit-equal across the dumps (a row's bits must not depend on the forward's
        row count). Prints the rows that differ per tensor.
  tile  A synthetic prompt of N rows from a real one: every row-indexed tensor of the oracle prompt file and of the
        capture's op outputs repeated along the rows (row r is row r mod T of the original), logits.rows the rows
        whose original is a logits row. The runner then runs an N-row forward (--chunk N) on real data, fed with
        torch's real outputs; each row's expected outputs are its original row's.
"""

import argparse
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from inkling2_compare import Store, op_parts  # noqa: E402

BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 1 << 62)]


def hist(ulp):
    return [int(((ulp >= lo) & (ulp <= hi)).sum()) for lo, hi in BUCKETS]


def ordered32(f):
    b = f.astype(np.float32).view(np.int32).astype(np.int64)
    return np.where(b < 0, -(b & 0x7FFFFFFF), b)


def ordered16(bits):
    b = bits.astype(np.int32)
    return np.where(b & 0x8000, -(b & 0x7FFF), b & 0x7FFF)


def bf16_bits(f):
    return torch.from_numpy(np.ascontiguousarray(f.astype(np.float32))).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def bf16_float(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def stat(a, b, kind):
    if kind == "bf16":
        fa, fb = bf16_float(a), bf16_float(b)
        u = np.abs(ordered16(a) - ordered16(b))
        out = {"ulps": hist(u), "max_ulp": int(u.max())}
    else:
        fa, fb = a.astype(np.float32), b.astype(np.float32)
        u = np.abs(ordered32(fa) - ordered32(fb))
        ub = np.abs(ordered16(bf16_bits(fa)) - ordered16(bf16_bits(fb)))
        out = {"ulps": hist(u), "max_ulp": int(u.max()), "bf16_ulps": hist(ub)}
    d = fa.astype(np.float64) - fb.astype(np.float64)
    nb = float(np.linalg.norm(fb.astype(np.float64)))
    out["max_abs"] = float(np.abs(d).max())
    out["rel"] = float(np.linalg.norm(d) / nb) if nb > 0 else float(np.linalg.norm(d))
    out["n"] = int(a.size)
    return out


def fmt(name, s):
    t = f"{name:36s} rel {s['rel']:.3e}  max_abs {s['max_abs']:.3e}  n {s['n']}  ulps[0,1,2,3-4,5-8,9-16,>16] "
    t += ",".join(str(x) for x in s["ulps"]) + f" (max {s['max_ulp']})"
    if "bf16_ulps" in s:
        t += "  as bf16 " + ",".join(str(x) for x in s["bf16_ulps"])
    return t


def selected(name, groups):
    p = op_parts(name)
    return p is not None and p[2] == "out" and "@" not in name and p[1].split(".")[0] in groups


def ulps(a):
    dump, cap = Store(a.dump), Store(a.capture)
    stubs = set(x for x in dump.meta.get("fed_stubs", "").split(",") if x)
    groups = a.groups.split(",")
    n, shape_bad = 0, 0
    for name in sorted(dump.keys):
        if not selected(name, groups) or not cap.has(name):
            continue
        p = op_parts(name)
        if f"{p[0]}.{p[1]}" in stubs:
            continue
        x, kind = dump.get(name)
        y, _ = cap.get(name)
        if x.shape != y.shape:
            print(f"{name}: shape {x.shape} against {y.shape}  FAIL")
            shape_bad += 1
            continue
        print(fmt(name, stat(x, y, kind)))
        n += 1
    print(f"{n} op outputs compared")
    return 1 if shape_bad else (0 if n else 2)


def rows(a):
    dumps = [Store(p) for p in a.dumps]
    if len(dumps) < 2:
        print("FAIL: row invariance needs at least two dumps")
        return 1
    groups = a.groups.split(",")
    want = lambda d: {n for n in d.keys if selected(n, groups) or n.startswith("h.")  # noqa: E731
                      or n in ("embed", "final", "logits.full")}
    keep = sorted(want(dumps[0]))
    chunks = [d.meta.get("chunk", "?") for d in dumps]
    print(f"{len(keep)} tensors in all of {len(dumps)} dumps (chunks {', '.join(chunks)})")
    bad = 0
    # coverage: every dump holds the same tensors, and every run finished
    for d, c in zip(dumps, chunks):
        if want(d) != set(keep):
            print(f"chunk {c}: {len(want(d) ^ set(keep))} tensors not in every dump  FAIL")
            bad += 1
        if d.meta.get("status", "complete") != "complete":
            print(f"chunk {c}: run did not finish ({d.meta.get('status')})  FAIL")
            bad += 1
    if not keep:
        print("FAIL: no tensors to compare")
        return 1
    keep = [n for n in keep if all(d.has(n) for d in dumps)]
    for name in keep:
        x0, _ = dumps[0].get(name)
        worst = 0
        for d, c in zip(dumps[1:], chunks[1:]):
            x, _ = d.get(name)
            if x.shape != x0.shape:
                print(f"{name}: shape {x.shape} in chunk {c} against {x0.shape}  FAIL")
                bad += 1
                continue
            diff = (x.reshape(x.shape[0], -1) != x0.reshape(x0.shape[0], -1)).any(axis=1)
            if diff.any():
                worst = max(worst, int(diff.sum()))
                print(f"{name}: {int(diff.sum())} of {x.shape[0]} rows differ between chunk {chunks[0]} and {c}"
                      f" (first {np.flatnonzero(diff)[:8].tolist()})  FAIL")
        if worst:
            bad += 1
        else:
            print(f"{name:40s} {x0.shape[0]} rows bit-equal across chunks {', '.join(chunks)}")
    print(("FAIL: %d tensors depend on the chunk" % bad) if bad else f"PASS: {len(keep)} tensors row-invariant")
    return 1 if bad else 0


def tile(a):
    lo, hi = (int(x) for x in a.layers.split(":")) if a.layers else (0, 0)
    o = safe_open(a.oracle, framework="pt")
    T = o.get_tensor("ids").shape[0]
    if o.get_tensor("rows").shape[0] != T:
        raise SystemExit("tile needs an oracle prompt that keeps every row")
    N = a.rows
    idx = torch.arange(N) % T
    lrows = o.get_tensor("logits.rows")
    pos = {int(r): j for j, r in enumerate(lrows.tolist())}
    head = [r for r in range(N) if int(idx[r]) in pos]
    hidx = torch.tensor([pos[int(idx[r])] for r in head])
    out = {}
    for k in o.keys():
        t = o.get_tensor(k)
        if k == "rows":
            out[k] = torch.arange(N, dtype=t.dtype)
        elif k == "logits.rows":
            out[k] = torch.tensor(head, dtype=t.dtype)
        elif k == "logits.full":
            out[k] = t[hidx].contiguous()
        elif t.dim() >= 1 and t.shape[0] == T:
            out[k] = t[idx].contiguous()
    save_file(out, a.oracle_out, metadata={"tool": "inkling2_res_check.py tile", "source": a.oracle, "rows": str(N)})
    print(f"oracle: {len(out)} tensors, {N} rows, {len(head)} logits rows -> {a.oracle_out}")
    c = safe_open(a.capture, framework="pt")
    cap = {}
    for k in c.keys():
        p = op_parts(k)
        if p is None or p[2] != "out" or "@" in k:
            continue
        if p[0] == "model":
            if not a.head:
                continue
        elif not (lo <= int(p[0]) < hi):
            continue
        t = c.get_tensor(k)
        if p[1] in ("head.unembed", "head.softcap"):
            cap[k] = t[hidx].contiguous()
        elif t.shape[0] == T:
            cap[k] = t[idx].contiguous()
    save_file(cap, a.capture_out, metadata={"tool": "inkling2_res_check.py tile", "source": a.capture, "rows": str(N)})
    print(f"capture: {len(cap)} op outputs -> {a.capture_out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("ulps")
    u.add_argument("dump")
    u.add_argument("capture")
    u.add_argument("--groups", default="res,head")
    r = sub.add_parser("rows")
    r.add_argument("dumps", nargs="+")
    r.add_argument("--groups", default="res,head")
    t = sub.add_parser("tile")
    t.add_argument("oracle")
    t.add_argument("capture")
    t.add_argument("--rows", type=int, required=True)
    t.add_argument("--oracle-out", required=True)
    t.add_argument("--capture-out", required=True)
    t.add_argument("--layers", default="")
    t.add_argument("--head", action="store_true")
    a = ap.parse_args()
    return {"ulps": ulps, "rows": rows, "tile": tile}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
