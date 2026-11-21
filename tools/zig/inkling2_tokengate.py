"""v0.2's token gate, offline: the yardstick (recipe/inkling2-tp4/smoke.py's yardstick_score and yardstick_verdict,
one definition, which users run from the tarball; this file imports them) on recorded forced tokens, and the
free-running greedy departures as information. Standard library only.

USAGE = python3 tools/zig/inkling2_tokengate.py score DECODE_JSON FORCED_JSON [--ext DIR]
        python3 tools/zig/inkling2_tokengate.py check DECODE_JSON SERVER_JSON [--ext DIR]
        python3 tools/zig/inkling2_tokengate.py order EXT_DECODE_JSON
        python3 tools/zig/inkling2_tokengate.py margins DECODE_JSON [--below 0.5]

The gate (smoke.py yardstick): the server is forced along the fp32 oracle's greedy replies, one greedy token at every
position j (the prompt plus the oracle's first j reply tokens), and over the 445 positions of chat_short, tool_call
and the extended set's 7 prompts its token must equal the oracle's top-1 at >= smoke.YARDSTICK_TOP1 positions and be
in the oracle's top-2 (decode.json's margins[j]) at >= smoke.YARDSTICK_TOP2. The bars are the worst scores of the
reference implementation's own valid variants on the same positions (smoke.py says which).

  score    the yardstick on FORCED_JSON (smoke.py yardstick --json's output, or {"name": [one token per oracle
           position], ...}): the per-prompt lines, every position off the oracle's top-1 with upstream TensorFold's
           one-bf16-step label (information), the YARDSTICK line; exit 1 when it fails.
  check    information: SERVER_JSON = {"name": [ids], ...} (a server's free-running greedy replies); smoke.py oracle's
           per-prompt lines (EQUAL, DEPARTS at j with the oracle's gap and the one-step label; DIFFER only for an
           oracle file without margins) and the EXT summary.
  order    the extended set's ORDER, as the frozen ORDER file was made: each prompt's first position whose oracle
           margin is at most one bf16 step (the reply's length when there is none), latest first; ties by name.
  margins  every position whose oracle margin is at most one bf16 step, or under --below (information).

verdict(oracle_row, oracle_token, server_token, server_row=None) describes one free-running departure as a dict
(information only); with a server row (a debug dump's top-2, never available in the release) it also reports
upstream's literal two-sided rule.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "recipe", "inkling2-tp4"))
import smoke  # noqa: E402

bf16_spacing = smoke.bf16_spacing
near_tie = smoke.near_tie
compare = smoke.compare
yardstick_score = smoke.yardstick_score
yardstick_verdict = smoke.yardstick_verdict


def verdict(oracle_row: dict, oracle_token: int, server_token: int, server_row: dict = None) -> dict:
    """One free-running departure, information only (the gate is the yardstick). oracle_row: decode.json's margins[j].
    server_row: {"top1", "top2", "top1_logit", "gap"} of the server's own row j, from a debug dump."""
    near, gap, step, why = near_tie(oracle_row, oracle_token, server_token)
    in_top2 = smoke.yardstick_row(oracle_row, oracle_token, server_token)[1] if oracle_row else False
    out = {"upstream_one_step_info": near, "in_oracle_top2": in_top2, "oracle_gap": gap, "oracle_step": step,
           "why": why, "label": "oracle side: top-2 = {oracle token, server token}, gap <= one bf16 step of top-1"}
    if oracle_row and oracle_row.get("gap") is not None:
        out["oracle_gap_pre_softcap"] = oracle_row.get("gap_pre")
        out["oracle_gap_under_0p5"] = oracle_row["gap"] < 0.5
    if server_row is not None:
        s_top = {server_row.get("top1"), server_row.get("top2")}
        s_gap = server_row.get("gap")
        s_ok = (s_top == {oracle_token, server_token} and s_gap is not None
                and s_gap <= bf16_spacing(float(server_row["top1_logit"])))
        out["upstream_two_sided_info"] = bool(near and s_ok)
        out["server_gap"] = s_gap
    return out


def first_near_tie(entry: dict, threshold=None) -> int:
    """The first position whose oracle margin is at most one bf16 step (threshold None) or under threshold; the reply's
    length when there is none."""
    for j, m in enumerate(entry.get("margins") or []):
        if m.get("gap") is None:
            continue
        lim = bf16_spacing(float(m["top1_logit"])) if threshold is None else threshold
        if (m["gap"] <= lim) if threshold is None else (m["gap"] < lim):
            return j
    return len(entry["ids"])


def prompts(dec: dict) -> list:
    return sorted(k for k, v in dec.items() if not k.startswith("_") and isinstance(v, dict) and "ids" in v)


def order(dec: dict) -> list:
    """ORDER: latest first oracle near tie first, ties by name."""
    names = prompts(dec)
    for n in names:
        if "margins" not in dec[n] or len(dec[n]["margins"]) != len(dec[n]["ids"]):
            raise SystemExit(f"{n}: no margins for every token (an oracle file made with inkling2_ref.py --attn fp32)")
    return sorted(names, key=lambda n: (-first_near_tie(dec[n]), n))


def cmd_order(a) -> int:
    dec = json.load(open(a.decode))
    names = order(dec)
    for n in names:
        print(n)
    total = 0
    print("# prompt | reply tokens | first near tie (one bf16 step) | first gap < 0.5", file=sys.stderr)
    for n in names:
        e = dec[n]
        total += len(e["ids"])
        print(f"# {n} | {len(e['ids'])} | {first_near_tie(e)} | {first_near_tie(e, 0.5)}", file=sys.stderr)
    print(f"# {len(names)} prompts, {total} reply tokens; tokens before the first near tie: "
          f"{sum(first_near_tie(dec[n]) for n in names)} (gap < 0.5: {sum(first_near_tie(dec[n], 0.5) for n in names)})",
          file=sys.stderr)
    return 0


def check(dec: dict, server: dict, gated: list, ext: dict = None, ext_order: list = None, ext_server: dict = None,
          out=print) -> dict:
    """smoke.py oracle's lines on recorded free-running replies (information, except DIFFER for an oracle file without
    margins). Returns {"fails", "results": {name: (status, tokens equal before the departure, oracle tokens)}}."""
    fails, results = 0, {}

    def one(name, d, srv):
        status, j, n, note = compare(d[name]["ids"], d[name].get("margins"), srv[name])
        out(f"{name}: oracle {len(d[name]['ids'])} tokens, server {len(srv[name])}: {status}"
            f"{'' if status == 'EQUAL' else f' at {j}'}{note}")
        results[name] = (status, n, len(d[name]["ids"]))
        return status

    for name in gated:
        fails += one(name, dec, server) == "DIFFER"
    if ext is not None:
        for name in ext_order:
            fails += one(name, ext, ext_server) == "DIFFER"
        done = [results[n] for n in ext_order]
        c = {k: sum(v[0] == k for v in done) for k in ("EQUAL", "DEPARTS", "DIFFER")}
        n, total = sum(v[1] for v in done), sum(v[2] for v in done)
        out(f"EXT summary: {len(done)} prompts, {c['EQUAL']} EQUAL, {c['DEPARTS']} DEPARTS, {c['DIFFER']} DIFFER; "
            f"{n} of {total} oracle tokens equal before the first departure ({100.0 * n / max(total, 1):.1f}%)")
    return {"fails": fails, "results": results}


def read_ext(path: str):
    with open(os.path.join(path, "decode.json")) as f:
        ext = json.load(f)
    with open(os.path.join(path, "ORDER")) as f:
        order = [x.strip() for x in f if x.strip() and not x.startswith("#")]
    return ext, order


def cmd_check(a) -> int:
    dec = json.load(open(a.decode))
    server = json.load(open(a.server))
    gated = [n for n in prompts(dec) if n in server] if not a.prompts else a.prompts.split(",")
    ext, ext_order = read_ext(a.ext) if a.ext else (None, None)
    r = check(dec, server, gated, ext, ext_order, server)
    return 1 if r["fails"] else 0


def score(dec: dict, forced: dict, gated: list, ext: dict = None, ext_order: list = None, out=print) -> dict:
    """The yardstick on recorded forced tokens: smoke.py yardstick's lines. Returns the score with "passed"."""
    parts = [(n, dec[n]) for n in gated] + [(n, ext[n]) for n in (ext_order or [])]
    rows = []
    for name, e in parts:
        if name not in forced:
            raise SystemExit(f"{name}: no forced tokens")
        one = yardstick_score([(name, e["ids"], e.get("margins"), forced[name])])
        out(f"{name}: {len(e['ids'])} positions: top-1 {one['top1']}, in the oracle's top-2 {one['top2']}")
        smoke.yardstick_lines(one, out)
        rows.append((name, e["ids"], e.get("margins"), forced[name]))
    r = yardstick_score(rows)
    r["passed"], line = yardstick_verdict(r)
    out(line)
    return r


def cmd_score(a) -> int:
    dec = json.load(open(a.decode))
    forced = json.load(open(a.forced))
    forced = forced.get("forced", forced)
    ext, ext_order = read_ext(a.ext) if a.ext else (None, [])
    gated = a.prompts.split(",") if a.prompts else [n for n in prompts(dec) if n in forced]
    return 0 if score(dec, forced, gated, ext, ext_order)["passed"] else 1


def cmd_margins(a) -> int:
    dec = json.load(open(a.decode))
    for n in prompts(dec):
        rows = dec[n].get("margins") or []
        hits = [(j, m) for j, m in enumerate(rows) if m.get("gap") is not None
                and (m["gap"] <= bf16_spacing(float(m["top1_logit"])) or m["gap"] < a.below)]
        print(f"{n}: {len(dec[n]['ids'])} tokens, first near tie (one bf16 step) at {first_near_tie(dec[n])}, "
              f"first gap < {a.below:g} at {first_near_tie(dec[n], a.below)}")
        for j, m in hits:
            step = bf16_spacing(float(m["top1_logit"]))
            print(f"  j={j}: top1 {m['top1']} {m['top1_logit']:.4f}, top2 {m['top2']} {m['top2_logit']:.4f}, gap "
                  f"{m['gap']:.4f} ({'<=' if m['gap'] <= step else '>'} one bf16 step {step:.4g}), pre-softcap gap "
                  f"{m['gap_pre']:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("order")
    o.add_argument("decode")
    sc = sub.add_parser("score")
    sc.add_argument("decode")
    sc.add_argument("forced")
    sc.add_argument("--prompts", default="", help="the gated prompts (default: every prompt both files hold)")
    sc.add_argument("--ext", default="")
    c = sub.add_parser("check")
    c.add_argument("decode")
    c.add_argument("server")
    c.add_argument("--prompts", default="", help="the gated prompts (default: every prompt both files hold)")
    c.add_argument("--ext", default="")
    m = sub.add_parser("margins")
    m.add_argument("decode")
    m.add_argument("--below", type=float, default=0.5)
    a = ap.parse_args()
    return {"score": cmd_score, "check": cmd_check, "order": cmd_order, "margins": cmd_margins}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
