"""How far apart valid reference forms of Inkling-2-Small put the same oracle rows: the top-1/top-2 gap spread, and
each form's yardstick score (the scores v0.2's token gate bars come from).

USAGE = python3 tools/zig/inkling2_tolerance.py MARGINS_JSON [MARGINS_JSON ...] [--json OUT_JSON] [--top 10]
        python3 tools/zig/inkling2_tolerance.py MARGINS_JSON [MARGINS_JSON ...] --yardstick

  python3 tools/zig/inkling2_tolerance.py gated/margins.json ext/margins.json --json tol.json

Inputs: inkling2_margins.py outputs (margins.json) over the fp32 oracle's replies (the gated prompts' decode.json and
the extended set's ext/decode.json), each holding every oracle sequence ("NAME/oracle") in every reference form: the
modes eager (Transformers' eager attention, cuBLAS bf16 partial reduction), eager32 (the same, fp32 reduction) and
fp32 (v0.2's oracle form, kx's multiplier sink), each on the decode path (prefill + one-row steps) and the
teacher-forced path (one whole-sequence forward). Branch sequences (a server's reply) are ignored. Every form must be
present for every oracle position; a missing one is an error, not a smaller set.

At every oracle position (prompt, j) and in every form f it reads the softcapped logits row's top-5 and computes:

  margin(f)  f's own top-1 minus top-2 (the two tokens f ranks first);
  pair(f)    logit(a) - logit(b) in f, a and b the fp32 decode path's top-1 and top-2 at that position (the oracle's
             own two contenders, the pair the gate examines); signed, so a form that swaps them counts in full.
             When b (or a) is not in f's top-5 the row is far from a tie; it is counted as missing.

spread(prompt, j) = max over both measures of (max over forms - min over forms). The tolerance is 2 x the largest
spread over all positions, printed with the position and the forms where the largest spread occurs, and rounded up
at the fourth decimal (information: a per-departure tolerance is not the gate).

--yardstick scores every form as recipe/inkling2-tp4/smoke.py yardstick scores a server: at each oracle position, is
the form's top-1 the oracle's token (the fp32 decode path's), and is it in the oracle's top-2. An exact tie at a
form's top makes its top-1 ambiguous, so each count is printed as a range (low: every tied token must qualify; high:
one is enough) around the form's own argmax, beside smoke.py's bars. Standard library only.
"""

import argparse
import hashlib
import json
import math
import os
import sys

FORMS = [(m, p) for m in ("eager", "eager32", "fp32") for p in ("decode", "tf")]
ORACLE_FORM = ("fp32", "decode")


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def collect(paths: list) -> dict:
    """{(prompt, j): {(mode, path): row}} over the oracle sequences of every input."""
    pos = {}
    for path in paths:
        for sq in json.load(open(path))["sequences"]:
            if not sq["label"].endswith("/oracle"):
                continue
            form = (sq["mode"], sq["path"])
            for r in sq["rows"]:
                at = pos.setdefault((sq["prompt"], r["j"]), {})
                if form in at:
                    raise SystemExit(f"{sq['prompt']} j={r['j']} {form}: given twice")
                at[form] = r
    return pos


def logit_of(row: dict, tok: int):
    return row["top_logits"][row["top"].index(tok)] if tok in row["top"] else None


def spreads(pos: dict) -> list:
    out = []
    for (prompt, j), at in sorted(pos.items()):
        missing = [f for f in FORMS if f not in at]
        if missing:
            raise SystemExit(f"{prompt} j={j}: no rows for {missing}")
        o = at[ORACLE_FORM]
        a = o["token"]  # topk may list an exact tie at the top in either order; the oracle's pick is the token
        b = o["top"][1] if o["top"][0] == a else o["top"][0]
        if a not in o["top"][:2] or o["top_logits"][0] != logit_of(o, a):
            raise SystemExit(f"{prompt} j={j}: the fp32 decode row's top-1 {a} is not the oracle's token {o['token']}")
        margin = {f: at[f]["margin"] for f in FORMS}
        pair, pair_missing = {}, []
        for f in FORMS:
            la, lb = logit_of(at[f], a), logit_of(at[f], b)
            if la is None or lb is None:
                pair_missing.append(f)
            else:
                pair[f] = la - lb
        rec = {"prompt": prompt, "j": j, "a": a, "b": b, "oracle_gap": o["margin"], "pair_missing": pair_missing}
        best = None
        for name, d in (("margin", margin), ("pair", pair)):
            if len(d) < 2:
                continue
            hi, lo = max(d, key=d.get), min(d, key=d.get)
            s = d[hi] - d[lo]
            rec[f"{name}_spread"] = s
            if best is None or s > best[0]:
                best = (s, name, hi, d[hi], lo, d[lo])
        rec.update({"spread": best[0], "measure": best[1], "form_hi": "/".join(best[2]), "value_hi": best[3],
                    "form_lo": "/".join(best[4]), "value_lo": best[5]})
        out.append(rec)
    return out


def yardstick(pos: dict) -> dict:
    """{form: {"top1": (low, argmax, high), "top2": (low, argmax, high)}} over every oracle position."""
    out = {}
    for f in FORMS:
        lo1 = hi1 = lo2 = hi2 = am1 = am2 = 0
        for (prompt, j), at in sorted(pos.items()):
            o = at[ORACLE_FORM]
            a = o["token"]
            top2 = set(o["top"][:2]) | {a}  # as smoke.yardstick_row reads decode.json's top1/top2
            r = at[f]
            tied = [t for t, lg in zip(r["top"], r["top_logits"]) if lg == r["top_logits"][0]]
            lo1 += all(t == a for t in tied)
            hi1 += any(t == a for t in tied)
            lo2 += all(t in top2 for t in tied)
            hi2 += any(t in top2 for t in tied)
            am1 += r["argmax"] == a
            am2 += r["argmax"] in top2
        out["/".join(f)] = {"top1": (lo1, am1, hi1), "top2": (lo2, am2, hi2)}
    return out


def ceil4(x: float) -> float:
    return math.ceil(x * 1e4 - 1e-9) / 1e4


def cmd_yardstick(a) -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "recipe", "inkling2-tp4"))
    import smoke

    pos = collect(a.margins)
    for p in a.margins:
        print(f"input {sha256(p)} {p}")
    for (prompt, j), at in sorted(pos.items()):
        missing = [f for f in FORMS if f not in at]
        if missing:
            raise SystemExit(f"{prompt} j={j}: no rows for {missing}")
    print(f"{len(pos)} oracle positions; bars (smoke.py): top-1 >= {smoke.YARDSTICK_TOP1}, in the oracle's top-2 >= "
          f"{smoke.YARDSTICK_TOP2} of {smoke.YARDSTICK_POSITIONS}")
    print("| form | top-1: argmax (low-high) | in the oracle's top-2: argmax (low-high) |")
    print("|---|---|---|")
    for f, r in yardstick(pos).items():
        t1, t2 = r["top1"], r["top2"]
        print(f"| {f} | {t1[1]} ({t1[0]}-{t1[2]}) | {t2[1]} ({t2[0]}-{t2[2]}) |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("margins", nargs="+")
    ap.add_argument("--json", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--yardstick", action="store_true", help="each form's yardstick score instead of the spreads")
    a = ap.parse_args()
    if a.yardstick:
        return cmd_yardstick(a)
    rows = spreads(collect(a.margins))
    worst = max(rows, key=lambda r: r["spread"])
    tol = 2.0 * worst["spread"]
    by_prompt = {}
    for r in rows:
        by_prompt[r["prompt"]] = by_prompt.get(r["prompt"], 0) + 1
    res = {"tool": "inkling2_tolerance.py", "inputs": {p: sha256(p) for p in a.margins}, "forms": ["/".join(f) for f in
           FORMS], "positions": len(rows), "prompts": by_prompt,
           "pair_missing_positions": sum(bool(r["pair_missing"]) for r in rows),
           "largest_spread": worst, "tolerance_exact": tol, "tolerance": ceil4(tol),
           "max_margin_spread": max(r.get("margin_spread", 0.0) for r in rows),
           "max_pair_spread": max(r.get("pair_spread", 0.0) for r in rows),
           "top": sorted(rows, key=lambda r: -r["spread"])[: a.top]}
    for p, h in res["inputs"].items():
        print(f"input {h} {p}")
    print(f"{len(rows)} oracle positions over {len(by_prompt)} prompts "
          f"({', '.join(f'{k} {n}' for k, n in by_prompt.items())}), forms {', '.join(res['forms'])}; "
          f"pair not in some form's top-5 at {res['pair_missing_positions']} positions")
    print(f"largest margin spread {res['max_margin_spread']:.6f}, largest pair spread {res['max_pair_spread']:.6f}")
    print("| prompt | j | a | b | fp32 decode gap | measure | spread | high form | high | low form | low |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["top"]:
        print(f"| {r['prompt']} | {r['j']} | {r['a']} | {r['b']} | {r['oracle_gap']:.4f} | {r['measure']} | "
              f"{r['spread']:.6f} | {r['form_hi']} | {r['value_hi']:.4f} | {r['form_lo']} | {r['value_lo']:.4f} |")
    w = worst
    print(f"TOLERANCE {res['tolerance']:.4f} (2 x {w['spread']:.6f} = {tol:.6f}, rounded up at 1e-4): {w['measure']} "
          f"spread at {w['prompt']} j={w['j']}, {w['form_hi']} {w['value_hi']:.4f} vs {w['form_lo']} "
          f"{w['value_lo']:.4f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
