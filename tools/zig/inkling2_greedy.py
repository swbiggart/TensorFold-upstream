#!/usr/bin/env python3
"""Greedy text by teacher forcing, for runs of tf-inkling2-cuda that cannot hold the whole model at once.

Greedy decoding is determined by the argmax at each step given the tokens before it. So one forward over the prompt
plus a candidate continuation (the oracle's greedy reply, tools/zig/inkling2_ref.py's decode.json) checks a whole
greedy reply at once: if the engine's top-1 after every prefix equals the next candidate token, the engine's greedy
text is exactly the candidate; at the first row where it differs, the engine's greedy text leaves the candidate there.
Layer ranges chain through files in the oracle's format (ids, rows, logits.rows, embed, h.<L>), so each range can run
on its own load.

  make  ORACLE_PROMPT DECODE_JSON OUT [--tokens N]  the prompt's ids plus the first N tokens of its greedy reply
  feed  BASE OUT DUMP...                           BASE plus the embed and h.<L> rows of the dumps (next range's input)
  read  BASE DUMP --model DIR                      the dump's logits.full: top-1 at each continuation row, the text
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def make(a):
    f = safe_open(a.prompt, framework="pt")
    ids = f.get_tensor("ids").to(torch.int64)
    name = os.path.basename(a.prompt).removesuffix(".safetensors")
    cont = json.load(open(a.decode))[name]["ids"][: a.tokens]
    T0 = ids.shape[0]
    full = torch.cat([ids, torch.tensor(cont, dtype=torch.int64)])
    T = full.shape[0]
    # row T0 - 1 predicts cont[0]; row T0 - 1 + j predicts cont[j]; the last row predicts the token after them
    lrows = torch.arange(T0 - 1, T, dtype=torch.int64)
    meta = {"tool": "inkling2_greedy.py make", "prompt": name, "prompt_rows": str(T0), "continuation": json.dumps(cont)}
    save_file({"ids": full, "rows": torch.arange(T, dtype=torch.int64), "logits.rows": lrows}, a.out, metadata=meta)
    print(f"{a.out}: {T0} prompt rows + {len(cont)} continuation tokens; head rows {T0 - 1}..{T - 1}")
    return 0


def feed(a):
    b = safe_open(a.base, framework="pt")
    out = {k: b.get_tensor(k) for k in b.keys()}
    meta = b.metadata() or {}
    for p in a.dumps:
        d = safe_open(p, framework="pt")
        m = d.metadata() or {}
        if m.get("status") != "complete":
            raise SystemExit(f"{p}: the run did not finish ({m.get('status')})")
        for k in d.keys():
            if k == "embed" or k.startswith("h."):
                out[k] = d.get_tensor(k)
    save_file({k: v.contiguous() for k, v in out.items()}, a.out, metadata=meta)
    hs = sorted(int(k[2:]) for k in out if k.startswith("h."))
    print(f"{a.out}: layer outputs h.{hs[0]}..h.{hs[-1]}, {len(out)} tensors")
    return 0


def read(a):
    from tokenizers import Tokenizer

    b = safe_open(a.base, framework="pt")
    meta = b.metadata()
    cont = json.loads(meta["continuation"])
    d = safe_open(a.dump, framework="pt")
    m = d.metadata() or {}
    if m.get("status") != "complete":
        raise SystemExit(f"{a.dump}: not a finished run (status: {m.get('status')})")
    lg = d.get_tensor("logits.full").float()
    top = lg.argmax(-1).tolist()
    tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    agree = 0
    for j, t in enumerate(cont):
        if top[j] != t:
            break
        agree += 1
    ours = top[:agree] + ([top[agree]] if agree < len(top) else [])
    res = {"continuation_tokens": len(cont), "greedy_equal_prefix": agree, "next_after": top[len(cont)] if len(top) > len(cont) else None,
           "text_candidate": tok.decode(cont, skip_special_tokens=False),
           "text_ours": tok.decode(ours, skip_special_tokens=False)}
    if agree < len(cont):
        p = torch.softmax(lg[agree].double(), -1)
        res["first_difference"] = {"row": agree, "ours": top[agree], "candidate": cont[agree],
                                   "p_ours": float(p[top[agree]]), "p_candidate": float(p[cont[agree]])}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"greedy: {agree} of {len(cont)} tokens equal the candidate" + (" (every one)" if agree == len(cont) else ""))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("prompt")
    m.add_argument("decode")
    m.add_argument("out")
    m.add_argument("--tokens", type=int, default=32)
    f = sub.add_parser("feed")
    f.add_argument("base")
    f.add_argument("out")
    f.add_argument("dumps", nargs="+")
    r = sub.add_parser("read")
    r.add_argument("base")
    r.add_argument("dump")
    r.add_argument("--model", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    return {"make": make, "feed": feed, "read": read}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
