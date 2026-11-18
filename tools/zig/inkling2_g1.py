"""Gate G1's table from inkling2_compare.py JSONs: every layer of tf-inkling2-cuda, teacher-fed, against the
tight reference (inkling2_op_ref.py layers --teacher --fp32-reduction), next to the oracle's own distance from it.

USAGE = python tools/zig/inkling2_g1.py DIR [--prompts a,b,...] [--chain DIR2] [--md OUT.md] [--json OUT.json]

DIR holds, per prompt P:
  cmpt-teacher-P-A-B[-TAG].json  our teacher-fed range [A, B) against the tight reference (TAG: a --chunk variant)
  cmp-tightref-P.json            the tight reference against the oracle (the oracle's own rounding distance)
  teacher-P-A-B[-TAG].st         the dumps: each TAG variant is diffed bit for bit with the untagged run (h.*, route.*,
                                 final, logits.full): a row's bits must not depend on its forward's size
--chain DIR2 adds the chained drift: cmp-ours-tight-P.json (our 48 layers chained from the embedding against the tight
oracle's chain) and cmp-oracle-tight-P.json (the default oracle's chain against it); head-P.json (inkling2_chain.py
head: both chains' predictions at every row against the tight chain's) when present.

The rule a layer passes by (teacher-fed, both sides on the oracle's input of that layer):
  R1 every row whose 8 experts differ from the tight reference's is a tie within rounding (inkling2_compare.py --model:
     the reference's score gap between the swapped experts is within twice our score error on them);
  R2 over the rows with equal picks, the hidden rows' relative distance from the tight reference is at most 1e-2 and
     at most the oracle's own distance from it over its equal-pick rows (ratio* <= 1);
  R3 router logits: our relative distance from the tight reference is at most the oracle's (ratio_r <= 1).
Per prompt: R4 our rows with differing picks number no more than the oracle's (summed over the layers), and every
--chunk variant is bit-equal to the whole-prompt run. The head passes when top-1 agreement with the tight reference is
at least the oracle's and the largest KL(tight || ours) per row is at most 1e-2.
Chained (with --chain): top-1 agreement with the tight chain at the oracle's logits.rows at least the oracle's; at
every row (head-P.json), the rows whose top-1 differs from the tight chain's at most the oracle's count plus two
Poisson deviations (2 sqrt(count)): past ~10 chained layers both chains' rounding has grown into chaotic drift, and
which near-tie rows flip is chance (two equally valid arithmetic variants of ours differ from each other by as much).
"""

import argparse
import glob
import json
import os
import re
import sys

PROMPTS = ["chat_short", "tool_call", "long_1500", "long_4000"]


def kind(L):
    return {"sliding": L % 6 != 5, "global": L % 6 == 5, "boundary": L % 6 == 0, "NVFP4": L < 40, "MXFP8": L >= 40}


GROUPS = [("sliding (40)", "sliding"), ("global (5, 11, ..., 47)", "global"), ("block boundary (0, 6, ..., 42)", "boundary"),
          ("NVFP4 experts (0-39)", "NVFP4"), ("MXFP8 experts (40-47)", "MXFP8")]


def load_layers(path):
    return json.load(open(path))["layers"] if os.path.exists(path) else {}


def same_rel(x):
    return x.get("h_same_routes", x["h"])["rel"]


def bit_diff(a, b):
    """h.*, route.*, final, logits.full of two dumps, bit for bit: (tensors compared, tensors differing)."""
    import torch
    from safetensors import safe_open

    fa, fb = safe_open(a, framework="pt"), safe_open(b, framework="pt")
    n = bad = 0
    for k in sorted(set(fa.keys()) & set(fb.keys())):
        if not (k.startswith("h.") or k.startswith("route.") or k in ("final", "logits.full")):
            continue
        x, y = fa.get_tensor(k), fb.get_tensor(k)
        if x.dtype in (torch.bfloat16, torch.float16):
            x, y = x.view(torch.int16), y.view(torch.int16)
        elif x.dtype == torch.float32:
            x, y = x.view(torch.int32), y.view(torch.int32)
        n += 1
        bad += int(x.shape != y.shape or not torch.equal(x, y))
    return n, bad


def prompt_rows(d, p):
    ours, variants = {}, []
    for f in sorted(glob.glob(f"{d}/cmpt-teacher-{p}-*.json")):
        m = re.search(rf"cmpt-teacher-{p}-(\d+)-(\d+)(-[^.]+)?\.json$", f)
        if not m:
            continue
        if m.group(3):
            variants.append((int(m.group(1)), int(m.group(2)), m.group(3)[1:]))
            continue
        ours |= load_layers(f)
    tight = load_layers(f"{d}/cmp-tightref-{p}.json")
    rows = {}
    for k, o in ours.items():
        if not k.isdigit():
            continue
        t = tight.get(k)
        r = {"rel_eq": same_rel(o), "rel_all": o["h"]["rel"], "eq_frac": o["h"]["equal_frac"],
             "swaps": o.get("route", {}).get("mismatch_rows", 0),
             "untied": o.get("route", {}).get("untied_rows", 0) if o.get("route") else 0,
             "rows": o.get("route", {}).get("rows", 0), "pass_cmp": o.get("pass")}
        if t:
            r["oracle_rel_eq"] = same_rel(t)
            r["oracle_swaps"] = t.get("route", {}).get("mismatch_rows", 0)
            r["ratio"] = r["rel_eq"] / r["oracle_rel_eq"] if r["oracle_rel_eq"] > 0 else float("inf")
            if "route_logits" in o and "route_logits" in t:
                r["ratio_r"] = o["route_logits"]["rel"] / max(t["route_logits"]["rel"], 1e-30)
                r["route_rel"] = o["route_logits"]["rel"]
                r["oracle_route_rel"] = t["route_logits"]["rel"]
        sw = [s for s in o.get("route", {}).get("swaps", []) if "score_err" in s]
        r["max_score_err"] = max((s["score_err"] / 2 for s in sw), default=0.0)
        r["ok"] = (r["untied"] == 0 and r["rel_eq"] <= 1e-2 and r.get("ratio", float("inf")) <= 1.0
                   and r.get("ratio_r", 0.0) <= 1.0)
        rows[int(k)] = r
    head = None
    if "logits" in ours:
        o, t = ours["logits"], tight.get("logits", {})
        head = {"top1": o.get("top1_equal_frac"), "oracle_top1": t.get("top1_equal_frac"), "kl_mean": o.get("kl_mean"),
                "kl_max": o.get("kl_max"), "rel": o.get("rel"), "oracle_rel": t.get("rel"),
                "final_rel": ours.get("final", {}).get("rel"), "oracle_kl_max": t.get("kl_max")}
        head["ok"] = (head["top1"] is not None and head["top1"] >= (head["oracle_top1"] or 0) and head["kl_max"] <= 1e-2)
    inv = []
    for a, b, tag in variants:
        base = f"{d}/teacher-{p}-{a}-{b}.st"
        var = f"{d}/teacher-{p}-{a}-{b}-{tag}.st"
        if os.path.exists(base) and os.path.exists(var):
            n, bad = bit_diff(base, var)
            chunk = json.load(open(f"{d}/cmpt-teacher-{p}-{a}-{b}-{tag}.json"))["meta"].get("chunk")
            inv.append({"range": f"{a}:{b}", "tag": tag, "chunk": chunk, "tensors": n, "differ": bad})
    return rows, head, inv


def chain_rows(c, p):
    o, t = load_layers(f"{c}/cmp-ours-tight-{p}.json"), load_layers(f"{c}/cmp-oracle-tight-{p}.json")
    if not o:
        return None
    out = {}
    for k in o:
        if k.isdigit() and k in t:
            out[int(k)] = {"ours_eq": same_rel(o[k]), "oracle_eq": same_rel(t[k]), "ours_all": o[k]["h"]["rel"],
                           "oracle_all": t[k]["h"]["rel"], "ours_swaps": o[k].get("route", {}).get("mismatch_rows", 0),
                           "oracle_swaps": t[k].get("route", {}).get("mismatch_rows", 0)}
    lo, lt = o.get("logits", {}), t.get("logits", {})
    head = {"ours_top1": lo.get("top1_equal_frac"), "oracle_top1": lt.get("top1_equal_frac"),
            "ours_kl_mean": lo.get("kl_mean"), "ours_kl_max": lo.get("kl_max"), "oracle_kl_mean": lt.get("kl_mean"),
            "oracle_kl_max": lt.get("kl_max"), "ours_final": o.get("final", {}).get("rel"),
            "oracle_final": t.get("final", {}).get("rel")}
    every = json.load(open(f"{c}/head-{p}.json")) if os.path.exists(f"{c}/head-{p}.json") else {}
    return {"layers": out, "head": head, "every_row": every}


def f(x, n=2):
    return "n/a" if x is None else f"{x:.{n}e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--chain", default="")
    ap.add_argument("--md", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    prompts = a.prompts.split(",")
    res = {p: dict(zip(("layers", "head", "invariance"), prompt_rows(a.dir, p))) for p in prompts}
    md = []
    md.append("| Layer type | " + " | ".join(prompts) + " |")
    md.append("| --- |" + " --- |" * len(prompts))
    verdict = True
    for title, key in GROUPS:
        cells = []
        for p in prompts:
            ls = {L: r for L, r in res[p]["layers"].items() if kind(L)[key]}
            if not ls:
                cells.append("not run")
                continue
            ok = all(r["ok"] for r in ls.values())
            verdict &= ok
            worst = max(ls, key=lambda L: ls[L]["rel_eq"])
            wr = max(ls, key=lambda L: ls[L].get("ratio", 0))
            sw = sum(r["swaps"] for r in ls.values())
            un = sum(r["untied"] for r in ls.values())
            osw = sum(r.get("oracle_swaps", 0) for r in ls.values())
            cells.append(f"{'PASS' if ok else 'FAIL'} {len(ls)} layers; rel* max {f(ls[worst]['rel_eq'])} (L{worst}),"
                         f" ratio* max {ls[wr].get('ratio', float('nan')):.2f} (L{wr}); picks differ {sw} rows,"
                         f" {sw - un} ties (oracle {osw})")
        md.append(f"| {title} | " + " | ".join(cells) + " |")
    cells = []
    for p in prompts:
        h = res[p]["head"]
        if not h:
            cells.append("not run")
            continue
        verdict &= h["ok"]
        cells.append(f"{'PASS' if h['ok'] else 'FAIL'} top-1 {h['top1']:.4f} (oracle {h['oracle_top1']:.4f}),"
                     f" KL mean {f(h['kl_mean'])} max {f(h['kl_max'])}, logits rel {f(h['rel'])}")
    md.append("| head / logits | " + " | ".join(cells) + " |")
    cells = []
    for p in prompts:
        ls = res[p]["layers"]
        cover = sorted(ls)
        missing = [L for L in range(48) if L not in ls]
        sw = sum(r["swaps"] for r in ls.values())
        osw = sum(r.get("oracle_swaps", 0) for r in ls.values())
        inv = res[p]["invariance"]
        invok = all(x["differ"] == 0 for x in inv)
        verdict &= sw <= osw and invok
        cells.append(f"{len(cover)} of 48 layers{'' if not missing else ' (missing ' + str(missing) + ')'}; picks differ"
                     f" {sw} rows vs the oracle's {osw} ({'R4 ok' if sw <= osw else 'R4 FAIL'}); chunk variants: "
                     + ("; ".join(f"{x['range']} chunk {x['chunk']} {'bit-equal' if x['differ'] == 0 else 'DIFFER ' + str(x['differ'])}"
                                  for x in inv) or "none"))
    md.append("| coverage, R4, row invariance | " + " | ".join(cells) + " |")
    md.append("")
    md.append(f"G1 teacher-fed verdict: {'PASS' if verdict else 'FAIL'}")
    md.append("")
    md.append("Per layer (teacher-fed): rel* = ours~tight over equal-pick rows; ratio* = rel* / the oracle's; ratio_r ="
              " router-logit distance / the oracle's; picks = rows differing from tight (ties) / the oracle's.")
    md.append("")
    md.append("| L | type | " + " | ".join(f"{p} rel* / ratio* / ratio_r / picks" for p in prompts) + " |")
    md.append("| --- | --- |" + " --- |" * len(prompts))
    for L in range(48):
        k = kind(L)
        t = ("global" if k["global"] else "sliding") + (" boundary" if k["boundary"] else "") + (" NVFP4" if k["NVFP4"] else " MXFP8")
        cells = []
        for p in prompts:
            r = res[p]["layers"].get(L)
            if not r:
                cells.append("-")
                continue
            cells.append(f"{f(r['rel_eq'])} / {r.get('ratio', float('nan')):.2f} / {r.get('ratio_r', float('nan')):.2f} /"
                         f" {r['swaps']} ({r['swaps'] - r['untied']}) / {r.get('oracle_swaps', '?')}"
                         + ("" if r["ok"] else " FAIL"))
        md.append(f"| {L} | {t} | " + " | ".join(cells) + " |")
    if a.chain:
        md.append("")
        md.append("Chained drift (our 48 layers chained from the embedding, in five ranges, against the tight oracle's"
                  " chain; the default oracle's chain against the same):")
        md.append("")
        md.append("| prompt | layer 11 ours / oracle | layer 23 | layer 35 | layer 47 | ours/oracle ratio: max (layer), median"
                  " | final ours / oracle | logits top-1 ours / oracle | KL mean ours / oracle | KL max ours / oracle |")
        md.append("| --- |" + " --- |" * 9)
        res["chain"] = {}
        for p in prompts:
            c = chain_rows(a.chain, p)
            if not c:
                continue
            res["chain"][p] = c
            ls, h = c["layers"], c["head"]
            cell = lambda L: f"{f(ls[L]['ours_eq'])} / {f(ls[L]['oracle_eq'])}" if L in ls else "-"  # noqa: E731
            rat = {L: ls[L]["ours_eq"] / ls[L]["oracle_eq"] for L in ls if ls[L]["oracle_eq"] > 0}
            wl = max(rat, key=rat.get) if rat else None
            med = sorted(rat.values())[len(rat) // 2] if rat else float("nan")
            md.append(f"| {p} | {cell(11)} | {cell(23)} | {cell(35)} | {cell(47)} | "
                      f"{rat[wl]:.2f} (L{wl}), median {med:.2f} | {f(h['ours_final'])} / {f(h['oracle_final'])} | "
                      f"{h['ours_top1']} / {h['oracle_top1']} | {f(h['ours_kl_mean'])} / {f(h['oracle_kl_mean'])} | "
                      f"{f(h['ours_kl_max'])} / {f(h['oracle_kl_max'])} |" if wl is not None else f"| {p} | incomplete |")
        rows = [(p, c["every_row"]) for p, c in res["chain"].items() if c.get("every_row")]
        if rows:
            md.append("")
            md.append("The chained head at every row (torch's unembed on each chain's final rows, against the tight chain's):")
            md.append("")
            md.append("| prompt | rows | top-1 equal ours / oracle | tight top-1 in top 5 ours / oracle | KL mean ours / oracle |"
                      " KL p99 ours / oracle | KL max ours / oracle |")
            md.append("| --- |" + " --- |" * 6)
            for p, e in rows:
                o, t = e.get("ours", {}), e.get("oracle", {})
                md.append(f"| {p} | {o.get('rows')} | {o.get('top1_equal_frac', 0):.4f} / {t.get('top1_equal_frac', 0):.4f} |"
                          f" {o.get('ref_top1_in_top5_frac', 0):.4f} / {t.get('ref_top1_in_top5_frac', 0):.4f} |"
                          f" {f(o.get('kl_mean'))} / {f(t.get('kl_mean'))} | {f(o.get('kl_p99'))} / {f(t.get('kl_p99'))} |"
                          f" {f(o.get('kl_max'))} / {f(t.get('kl_max'))} |")
        md.append("")
        for p, c in res["chain"].items():
            # the oracle's logits.rows: strictly at least the oracle's agreement
            o, t = c["head"]["ours_top1"] or 0, c["head"]["oracle_top1"] or 1
            ok = o >= t
            verdict &= ok
            md.append(f"Chained head, {p} (logits.rows): top-1 agreement with the tight chain, ours {o:.4f} against the"
                      f" oracle's {t:.4f}: {'ok' if ok else 'FAIL'}")
            e = c.get("every_row") or {}
            if e:
                # every row: a chained forward is chaotic past ~10 layers, so which rows flip is chance; the count of
                # rows whose top-1 leaves the tight chain's is held to the oracle's count plus two Poisson deviations
                n = e["ours"]["rows"]
                do = round(n * (1 - e["ours"]["top1_equal_frac"]))
                dt = round(n * (1 - e["oracle"]["top1_equal_frac"]))
                lim = dt + 2 * max(dt, 1) ** 0.5
                ok = do <= lim
                verdict &= ok
                md.append(f"Chained head, {p} (every row): rows whose top-1 differs from the tight chain's, ours {do} of {n}"
                          f" against the oracle's {dt} (limit {lim:.1f}; strictly {'no more' if do <= dt else 'MORE'}):"
                          f" {'ok' if ok else 'FAIL'}")
        md.append("")
        md.append(f"G1 verdict (teacher-fed table, invariance and the chained head): {'PASS' if verdict else 'FAIL'}")
    text = "\n".join(md)
    print(text)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(text + "\n")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
