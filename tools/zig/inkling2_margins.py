"""Teacher-forced token margins of Inkling-2's greedy oracle replies: is each departure from the oracle a near tie?

USAGE = python tools/zig/inkling2_margins.py MODEL_DIR ORACLE_DECODE_JSON --out OUT_DIR
            [--prompts chat_short,tool_call|ext] [--ids NAME=IDS_JSON,...] [--branch LABEL=PROMPT:REPLY_JSON ...]
            [--modes eager,eager32,fp32] [--tf-dir ORACLE_DIR] [--device cuda:0]

  python tools/zig/inkling2_margins.py MODEL ORACLE/full/decode.json --out OUT \
      --ids chat_short=data/chat_short.ids.json,tool_call=data/tool_call.ids.json \
      --branch kx=tool_call:kx-tool_call.json --tf-dir ORACLE/full

Each sequence is a prompt (rendered by inkling2_ref.py's prompt_specs, and checked equal to --ids when given) plus a
reply: the oracle's (ORACLE_DECODE_JSON, inkling2_ref.py --decode) or a branch's (REPLY_JSON: {"ids": [...]} or
{"text": "..."}, the text encoded by the checkpoint's tokenizer without special-token insertion, its decode checked
equal to the text). One whole-sequence forward per sequence and mode, layers streamed as in inkling2_ref.py's
teacher-forced pass (full mode: experts decoded to bf16, Transformers' eager modules). At every reply position j
(the logits row of position P-1+j, P the prompt length) it records the top-5 tokens and logits, the margin
top1 - top2, the reply's own token, its logit and rank, and the row's argmax (the oracle's greedy pick: torch's tie
winner, the first maximal ID). Top-1 agreement is the argmax equal to the reply token; rank 0 (no logit above the
reply's) is counted apart, as the reply at the maximum. At every departure (the first position where a branch leaves
the oracle reply) it records, in both sequences, the logits of the two contenders and both tie rules:

  upstream (tools/zig/qwen27_margins.py): near tie when the top two are exactly the two contenders and top1 - top2 is
           at most one bf16 spacing of the top-1 logit (on both sides; here the reference side only)
  0.5:     whether the oracle-side margin between the two contenders is under 0.5 logit (information; a server's
           side, from prompt_logprobs, is not visible to this tool)

Modes (the attention core and the cuBLAS reduction; everything else is the reference's own code):
  eager    Transformers' eager attention (bf16 scores, bias add, softmax in fp32 rounded to bf16 P, bf16 P.V, output
           times sigmoid(gate) rounded to bf16), cuBLAS bf16 partial reduction allowed (torch's default). The
           arithmetic of v0.1's oracle (inkling2_ref.py --decode without --fp32-reduction).
  eager32  eager attention, bf16 matmuls reduced in fp32 (inkling2_ref.py --fp32-reduction).
  fp32     v0.2's oracle form (inkling2_ref.py --attn fp32: the one definition,
           inkling2_ref.kx_core): q', k', v', the bias (the module's 16-term product, rounded to bf16) and the sink
           logit from the eager heads as they are, then scores q.k in fp32 times the scale plus the bias, m and z over
           the keys only, weights e^(s - m) in fp32 (never rounded), o = (weights . v) / z in fp32, times sigmoid(gate)
           in fp32, times sigmoid((m + log z) - sink) (the sink as kx's multiplier), one bf16 rounding (at the output
           projection's input). bf16 matmuls reduced in fp32. Until 2026-11-17 this mode folded the sink into z as a
           logit (z + e^(sink - m)); the same value in real arithmetic, other fp32 roundings.
TF32 is off in every mode and checked (inkling2_ref.tf32_guard: exit 3 if TF32 could be on).

Output: OUT_DIR/margins.json (every row), OUT_DIR/margins.txt (the tables), OUT_DIR/manifest.json (provenance,
modes, input sha256, wall times).

Transformers (Apache-2.0) is imported through inkling2_ref.py; its eager attention is called, not copied. The fp32
core is written from the reference document (see THIRD_PARTY_NOTICES.md).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling2_ref as R  # noqa: E402

BF = torch.bfloat16

MODES = {
    "eager": {"attn": "eager", "fp32_reduction": False},
    "eager32": {"attn": "eager", "fp32_reduction": True},
    "fp32": {"attn": "fp32", "fp32_reduction": True},
}
TOPK = 5
HALF_LOGIT = 0.5
bf16_spacing = R.bf16_spacing


def set_mode(name: str):
    R.ATTN["form"] = MODES[name]["attn"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not MODES[name]["fp32_reduction"]


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_branch(tok, path: str) -> list:
    d = json.load(open(path))
    if "ids" in d:
        return [int(t) for t in d["ids"]]
    ids = tok.encode(d["text"], add_special_tokens=False)
    if tok.decode(ids, skip_special_tokens=False) != d["text"]:
        raise SystemExit(f"{path}: the text does not survive encode/decode")
    if "tokens" in d and len(ids) != d["tokens"]:
        raise SystemExit(f"{path}: {len(ids)} tokens encoded, the server reported {d['tokens']}")
    return ids


def first_diff(a: list, b: list):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)


def agreement(lg: torch.Tensor, reply: list) -> tuple:
    """Rows j of lg (a logits row a reply position) against reply token j: each row's greedy pick as the oracle takes
    it, int(lg[j].argmax()) (inkling2_ref.py --decode; torch's tie winner, the first maximal ID), the rows whose pick is
    the reply token (top-1 agreement), and the rows where no logit exceeds the reply token's (rank 0: at a tie at the
    maximum, every tied token counts here, the pick only in the agreement)."""
    picks = [int(lg[j].argmax()) for j in range(len(reply))]
    agree = sum(p == t for p, t in zip(picks, reply))
    at_max = sum(int((lg[j] > lg[j, t]).sum()) == 0 for j, t in enumerate(reply))
    return picks, agree, at_max


def tie_rules(lg: torch.Tensor, raw: torch.Tensor, a: int, b: int) -> dict:
    """lg: one softcapped fp32 logits row, raw: the same row before the softcap (the bf16 unembed output in fp32).
    a: the oracle's token, b: the branch's token. The upstream rule is applied to both rows: to the softcapped logits
    the sampler ranks (literal), and to the bf16 values the rounding happens in (pre-softcap)."""
    out = {}
    for name, row in (("softcapped", lg), ("pre_softcap", raw)):
        v, i = row.topk(2)
        top = [int(i[0]), int(i[1])]
        margin = float(v[0] - v[1])
        sp = bf16_spacing(float(v[0]))
        gap = float(row[a] - row[b])
        out[name] = {"top2": top, "top2_logits": [float(v[0]), float(v[1])], "top_margin": margin,
                     "bf16_spacing": sp, "contender_logits": [float(row[a]), float(row[b])],
                     "gap_oracle_minus_branch": gap, "gap_in_bf16_steps": gap / sp,
                     "upstream_near_tie_ref_side": set(top) == {a, b} and margin <= sp}
    out["under_0p5_ref_side"] = abs(out["softcapped"]["gap_oracle_minus_branch"]) < HALF_LOGIT
    return out


def tf_pass(model, cfg, seqs, specs, modes, dev, times):
    """One whole-sequence forward per (sequence, mode), dense layers streamed (inkling2_ref.py's teacher-forced
    pass)."""
    emb = model.embed()
    states = []
    for s in seqs:
        ids = torch.tensor([specs[s["prompt"]]["ids"] + s["reply"]], device=dev)
        e = emb(ids)
        pos = torch.arange(ids.shape[1], device=dev)[None]
        mk = R.masks(cfg, e, pos)
        for m in modes:
            states.append((s, m, {"h": e, "blocks": e.new_zeros(1, ids.shape[1], 0, e.shape[-1]), "masks": mk,
                                  "pos": pos}))
    del emb
    torch.cuda.empty_cache()
    for i in range(cfg.num_hidden_layers):
        t0 = time.time()
        layer = model.layer(i)  # inkling2_ref.attach_attn: the fp32 form's sink factor and bf16 rounding
        t1 = time.time()
        for s, m, st in states:
            set_mode(m)
            R.run_layer(layer, st)
        del layer
        torch.cuda.synchronize()
        times.append(("tf", i, round(t1 - t0, 2), round(time.time() - t1, 2)))
        print(f"tf layer {i}: load {t1 - t0:.1f} s, run {time.time() - t1:.1f} s", flush=True)
    head = model.head()
    _, _, lm, cap = head
    rows = {}
    for s, m, st in states:
        set_mode(m)
        hn = R.finish(head, st)
        P, n = s["P"], len(s["reply"])
        raw = lm(hn[0, P - 1:P - 1 + n]).float()[..., : cfg.unpadded_vocab_size]
        rows[(s["label"], m, "tf")] = (cap(raw).cpu(), raw.cpu())
    del head, states
    torch.cuda.empty_cache()
    return rows


def decode_pass(model, cfg, seqs, specs, modes, dev, lru_cap, times):
    """The oracle's own path (inkling2_ref.py's greedy_decode: the prompt as one prefill, then one row a step through
    a DynamicCache, experts decoded on use), each step fed the reply's token instead of the argmax."""
    from transformers.cache_utils import DynamicCache

    lru = R.ExpertLRU(dev, lru_cap)
    t0 = time.time()
    emb = model.embed()
    layers = [model.layer(i, experts="lazy", lru=lru) for i in range(cfg.num_hidden_layers)]
    head = model.head()
    _, _, lm, cap = head
    print(f"decode: resident layers built in {time.time() - t0:.1f} s", flush=True)
    streams = [(s, m) for s in seqs for m in modes]
    caches = [DynamicCache(config=cfg) for _ in streams]
    cur = [torch.tensor([specs[s["prompt"]]["ids"]], device=dev) for s, _ in streams]
    got = [([], []) for _ in streams]
    steps = max(len(s["reply"]) for s, _ in streams)
    for step in range(steps):
        ts = time.time()
        live = [k for k, (s, _) in enumerate(streams) if step < len(s["reply"])]
        states = {}
        for k in live:
            c, ids = caches[k], cur[k]
            e = emb(ids)
            pos = (torch.arange(ids.shape[1], device=dev) + c.get_seq_length())[None]
            states[k] = {"h": e, "blocks": e.new_zeros(1, ids.shape[1], 0, e.shape[-1]),
                         "masks": R.masks(cfg, e, pos, c), "pos": pos}
        for ly in layers:
            for k in live:
                set_mode(streams[k][1])
                R.run_layer(ly, states[k], caches[k])
        for k in live:
            s, m = streams[k]
            set_mode(m)
            raw = lm(R.finish(head, states[k])[0, -1:]).float()[0, : cfg.unpadded_vocab_size]
            got[k][0].append(cap(raw).cpu())
            got[k][1].append(raw.cpu())
            cur[k] = torch.tensor([[s["reply"][step]]], device=dev)
        times.append(("decode", step, len(live), round(time.time() - ts, 2)))
        print(f"decode step {step}: {len(live)} streams, {time.time() - ts:.1f} s, expert hits {lru.hits} "
              f"misses {lru.misses}", flush=True)
    rows = {(s["label"], m, "decode"): (torch.stack(g[0]), torch.stack(g[1])) for (s, m), g in zip(streams, got)}
    del layers, head, caches, lru
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("oracle_decode")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="chat_short,tool_call", help="prompt_specs or ext_specs names; 'ext' "
                    "stands for all of ext_specs (in their order)")
    ap.add_argument("--ids", default="", help="NAME=IDS_JSON,...: the prompt ids the server is given; must equal the "
                    "reference's rendering")
    ap.add_argument("--branch", action="append", default=[], help="LABEL=PROMPT:REPLY_JSON (repeatable)")
    ap.add_argument("--modes", default="eager,eager32,fp32")
    ap.add_argument("--paths", default="tf,decode", help="tf: one whole-sequence forward; decode: the oracle's own "
                    "prefill + one-row steps, fed the reply")
    ap.add_argument("--tf-dir", default="", help="the oracle's teacher-forced folder (inkling2_ref.py --out .../full): "
                    "its last prompt row's top-64 logits compared with this run's eager row P-1 of each path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expert-cache", type=int, default=1000, help="decode path: experts kept on --device")
    ap.add_argument("--layers", type=int, default=0, help="only the first N layers, then the head (smoke; 0: all)")
    a = ap.parse_args()
    t_start = time.time()
    from transformers import AutoTokenizer

    modes = a.modes.split(",")
    paths = a.paths.split(",")
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m}")
    for p in paths:
        if p not in ("tf", "decode"):
            raise SystemExit(f"unknown path {p}")
    dev = torch.device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    cfg = R.text_config(a.model_dir, a.layers)
    specs = {s["name"]: s for s in R.prompt_specs(tok) + R.ext_specs(tok)}
    for s in specs.values():
        s["text"], s["ids"] = R.render(tok, s)
    inputs = {"oracle_decode": {"path": a.oracle_decode, "sha256": file_sha(a.oracle_decode)}}
    for kv in filter(None, a.ids.split(",")):
        name, path = kv.split("=", 1)
        given = [int(t) for t in json.load(open(path))]
        if given != specs[name]["ids"]:
            raise SystemExit(f"{name}: {path} differs from the reference's rendering")
        inputs[f"ids.{name}"] = {"path": path, "sha256": file_sha(path), "tokens": len(given)}
    oracle = json.load(open(a.oracle_decode))
    seqs = []
    names = [n for p in a.prompts.split(",") for n in ([e["name"] for e in R.ext_specs(tok)] if p == "ext" else [p])]
    for name in names:
        ids = oracle[name]["ids"]
        seqs.append({"label": f"{name}/oracle", "prompt": name, "reply": ids, "oracle": ids})
    for br in a.branch:
        label, rest = br.split("=", 1)
        name, path = rest.split(":", 1)
        ids = load_branch(tok, path)
        inputs[f"branch.{label}"] = {"path": path, "sha256": file_sha(path), "tokens": len(ids), "ids": ids}
        seqs.append({"label": f"{name}/{label}", "prompt": name, "reply": ids, "oracle": oracle[name]["ids"]})
    for s in seqs:
        s["P"] = len(specs[s["prompt"]]["ids"])
        s["departure"] = first_diff(s["reply"], s["oracle"])

    tf32 = R.tf32_guard(dev)
    model = R.Model(a.model_dir, cfg, dev, "full")  # installs inkling2_ref's attention dispatch
    rows, times = {}, []
    with torch.no_grad():
        if "tf" in paths:
            rows.update(tf_pass(model, cfg, seqs, specs, modes, dev, times))
        if "decode" in paths:
            rows.update(decode_pass(model, cfg, seqs, specs, modes, dev, a.expert_cache, times))

    res = {"modes": {m: MODES[m] for m in modes}, "paths": paths, "checks": {}, "sequences": [], "departures": []}
    if a.tf_dir and "eager" in modes:
        from safetensors.torch import load_file
        for s in seqs:
            if not s["label"].endswith("/oracle"):
                continue
            ref = load_file(os.path.join(a.tf_dir, s["prompt"] + ".safetensors"))
            ri, rv = ref["topk.ids"][-1].long(), ref["topk.vals"][-1]
            for p in paths:
                g = rows[(s["label"], "eager", p)][0][0]
                res["checks"][f"{s['prompt']} {p} eager row P-1 vs the oracle's teacher-forced file"] = {
                    "top1_equal": int(ri[0]) == int(g.argmax()), "top64_max_abs_diff": float((g[ri] - rv).abs().max()),
                    "top64_bit_equal": bool(torch.equal(g[ri], rv))}
    if "decode" in paths and "eager" in modes:
        for s in seqs:
            if s["label"].endswith("/oracle") and "prefill_top5" in oracle[s["prompt"]]:
                g = rows[(s["label"], "eager", "decode")][0][0]
                res["checks"][f"{s['prompt']} decode eager prefill top-5 vs decode.json"] = {
                    "equal": g.topk(5).indices.tolist() == oracle[s["prompt"]]["prefill_top5"]}
    for s in seqs:
        for p in paths:
            for m in modes:
                lg, raw = rows[(s["label"], m, p)]
                picks, agree, at_max = agreement(lg, s["reply"])
                out = []
                for j, t in enumerate(s["reply"]):
                    v, i = lg[j].topk(TOPK)
                    rv = raw[j].topk(2).values
                    out.append({"j": j, "token": t, "tok": tok.convert_ids_to_tokens(t), "logit": float(lg[j, t]),
                                "rank": int((lg[j] > lg[j, t]).sum()), "argmax": picks[j],
                                "top": [int(x) for x in i],
                                "top_tok": tok.convert_ids_to_tokens([int(x) for x in i]),
                                "top_logits": [float(x) for x in v], "margin": float(v[0] - v[1]),
                                "bf16_spacing": bf16_spacing(float(v[0])),
                                "pre_softcap_top2": [float(x) for x in rv],
                                "pre_softcap_margin_steps": float(rv[0] - rv[1]) / bf16_spacing(float(rv[0]))})
                res["sequences"].append({"label": s["label"], "prompt": s["prompt"], "mode": m, "path": p,
                                         "prompt_tokens": s["P"], "reply_tokens": len(s["reply"]),
                                         "departure": s["departure"], "top1_agrees": agree,
                                         "reply_at_max": at_max, "rows": out})
                lo = min(out, key=lambda r: r["margin"])
                print(f"{s['label']} {p} {m}: top-1 (argmax) equals the reply at {agree}/{len(out)} (the reply at "
                      f"the maximum, ties included: {at_max}); min margin "
                      f"{lo['margin']:.4f} at j={lo['j']}", flush=True)
        d = s["departure"]
        if d is None or s["label"].endswith("/oracle"):
            continue
        oa, ob = s["oracle"][d], s["reply"][d]
        o_label = f"{s['prompt']}/oracle"
        for p in paths:
            for m in modes:
                dep = {"branch": s["label"], "mode": m, "path": p, "j": d, "oracle_token": oa,
                       "oracle_tok": tok.convert_ids_to_tokens(oa), "branch_token": ob,
                       "branch_tok": tok.convert_ids_to_tokens(ob),
                       "in_branch_sequence": tie_rules(rows[(s["label"], m, p)][0][d], rows[(s["label"], m, p)][1][d],
                                                       oa, ob)}
                if (o_label, m, p) in rows:
                    dep["in_oracle_sequence"] = tie_rules(rows[(o_label, m, p)][0][d], rows[(o_label, m, p)][1][d],
                                                          oa, ob)
                res["departures"].append(dep)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "margins.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "margins.txt"), "w") as f:
        f.write(tables(res))
    man = R.provenance(a.model_dir)
    man.update({"tool": "inkling2_margins.py", "layers": cfg.num_hidden_layers, "argv": sys.argv[1:], "inputs": inputs,
                "device": a.device,
                "tf32": tf32,
                "times_s": times, "wall_s": round(time.time() - t_start, 1)})
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(tables(res), flush=True)


def tables(res: dict) -> str:
    L = [f"{k}: {v}" for k, v in res["checks"].items()]
    for sq in res["sequences"]:
        L.append("")
        L.append(f"## {sq['label']} [{sq['path']}, {sq['mode']}] prompt {sq['prompt_tokens']}, reply "
                 f"{sq['reply_tokens']}, top-1 (argmax) = reply at {sq['top1_agrees']}/{sq['reply_tokens']} (reply at "
                 f"the maximum: {sq['reply_at_max']}), departure from the oracle: {sq['departure']}")
        L.append("| j | reply token | logit | rank | top-1 | top-2 | margin | bf16 step | pre-softcap margin (steps) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in sq["rows"]:
            L.append(f"| {r['j']} | {r['token']} {r['tok']!r} | {r['logit']:.4f} | {r['rank']} | "
                     f"{r['top'][0]} {r['top_tok'][0]!r} {r['top_logits'][0]:.4f} | "
                     f"{r['top'][1]} {r['top_tok'][1]!r} {r['top_logits'][1]:.4f} | {r['margin']:.4f} | "
                     f"{r['bf16_spacing']:.4g} | {r['pre_softcap_margin_steps']:.2f} |")
    if res["departures"]:
        L.append("")
        L.append("## departures (oracle token a, branch token b; gap = logit(a) - logit(b); steps = gap / one bf16 "
                 "spacing at the top-1 value)")
        L.append("| branch | path | mode | j | a | b | sequence | logit a | logit b | gap | top-2 | "
                 "upstream (softcapped) | pre-softcap a / b | gap steps | upstream (pre-softcap) | |gap| < 0.5 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for d in res["departures"]:
            for where in ("in_oracle_sequence", "in_branch_sequence"):
                if where not in d:
                    continue
                c, r = d[where]["softcapped"], d[where]["pre_softcap"]
                L.append(f"| {d['branch']} | {d['path']} | {d['mode']} | {d['j']} | {d['oracle_token']} "
                         f"{d['oracle_tok']!r} | {d['branch_token']} {d['branch_tok']!r} | {where.split('_')[1]} | "
                         f"{c['contender_logits'][0]:.4f} | {c['contender_logits'][1]:.4f} | "
                         f"{c['gap_oracle_minus_branch']:.4f} | {c['top2']} | {c['upstream_near_tie_ref_side']} | "
                         f"{r['contender_logits'][0]:.4f} / {r['contender_logits'][1]:.4f} | "
                         f"{r['gap_in_bf16_steps']:.2f} | {r['upstream_near_tie_ref_side']} | "
                         f"{d[where]['under_0p5_ref_side']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
