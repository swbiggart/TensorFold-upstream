"""Inkling-2-Small's DSpark drafter (mtp.safetensors) in torch: the reference for the CUDA drafter and the offline
acceptance measurement.

The drafter math follows TokenSpeed's native draft (`models/inkling2_mtp.py`, `execution/drafter/inkling2.py` and
`dflash.py`, MIT; notice kept below):

  taps       the target's raw block partial at the entry of layers 5, 11, ..., 41 (the oracle's h.4, h.10, ...,
             h.40) and the final normed hidden (the oracle's `final`): 8 taps of 4608 per position
  mixtures   fuse_alpha = softmax(fuse_weight fp32).bf16 [6, 8]; stage s (layer s = 0..4, Markov head = 5):
             dflare_norm_s(sum_t tap_t fp32 * alpha[s, t]) -> bf16
  context    per draft layer, per target position: K = rope(k_norm_ctx(mix @ k_proj_ctx)), V = mix @ v_proj_ctx
  block      8 rows at positions p..p+7: row 0 is the target's normed embedding of the anchor (the token at p, the
             last one the target produced), rows 1..7 mask_embed; 5 dense layers (q/k norm, neox rope theta 1e6,
             GQA 48/8, non-causal in the block, the block sees context positions [0, p) - in local layers 0, 1, 3, 4
             only the last 1024), pre-norm residuals with the fp32 add-norm, SwiGLU 18432; final norm
  head       rows 1..7: base = row @ unembed (bf16, uncapped); recurrent state 256 from zero, previous token = the
             anchor, f_rnn = Markov-head mixture of the taps at p - 1; per step proj = rnn_proj([state, w1[prev],
             row, f_rnn]); gate, cand, out; state = g*state + (1-g)*tanh(cand); latent = tanh(out);
             logits = softcap30(base + fold_b + latent @ fold_w^T) with fold = unembed @ hidden_out in fp64 -> bf16;
             the greedy pick feeds the next step
  proposals  7 tokens for positions p+1..p+7 (block 8, verify width 8)

USAGE
  inkling2_dspark_ref.py gen MODEL OUT [--prompts a,b] [--steps N] [--mode full|checkpoint] [--expert-cache N]
      target greedy continuations with every position's 8 taps (oracle modules, layer weights resident, experts
      fetched per pick): OUT/<prompt>.safetensors {ids, n_prompt, taps [T, 8, 4608] bf16}
  inkling2_dspark_ref.py accept MODEL OUT/<prompt>.safetensors... [--json FILE] [--causal]
      offline acceptance under greedy verify: proposals at every anchor, then the mean accepted length (tokens a
      round, bonus included) for 1..7 drafts
  inkling2_dspark_ref.py fold MODEL OUT.safetensors
      the Markov head's folds (unembed . hidden_out in fp64, stored bf16) the CUDA drafter loads
  inkling2_dspark_ref.py dump MODEL OUT/<prompt>.safetensors --anchor P --out FILE
      one round's inputs and every intermediate (the CUDA drafter's op check)

Notice for the ported drafter math (TokenSpeed, https://github.com/lightseekorg/tokenspeed), under its MIT licence:

  Copyright (c) 2026 LightSeek Foundation

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
  to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
  the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BF = torch.bfloat16
H, NQ, NKV, HD, RANK, BLOCK = 4608, 48, 8, 128, 256, 8
TAP_LAYERS = (4, 10, 16, 22, 28, 34, 40)  # the oracle's h.L: the partial entering layers 5, 11, ..., 41
LOCAL = (0, 1, 3, 4)
WINDOW = 1024
EPS = 1e-6
SOFTCAP = 30.0
VALID = 200058
THETA = 1e6

torch.backends.cuda.matmul.allow_tf32 = False


# ----------------------------------------------------------------------------------------------- gen


def extra_specs():
    return [{"name": "chat_long", "effort": "none",
             "messages": [{"role": "user", "content": "Explain how a refrigerator keeps food cold, step by step, "
                                                      "in about 200 words."}]}]


def gen(a):
    import inkling2_ref as R
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    dev = torch.device(a.device)
    cfg = R.text_config(a.model_dir, 0)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    specs = R.prompt_specs(tok) + extra_specs()
    for s in specs:
        s["text"], s["ids"] = R.render(tok, s)
    want = a.prompts.split(",")
    specs = [s for s in specs if s["name"] in want]
    os.makedirs(a.out, exist_ok=True)
    model = R.Model(a.model_dir, cfg, dev, a.mode)
    lru = R.ExpertLRU(dev, a.expert_cache)
    t0 = time.time()
    emb = model.embed()
    layers = [model.layer(i, experts="lazy", lru=lru) for i in range(cfg.num_hidden_layers)]
    head = model.head()
    print(f"resident layers built in {time.time() - t0:.1f} s", flush=True)
    caches = [DynamicCache(config=cfg) for _ in specs]
    outs = [[] for _ in specs]
    taps = [[] for _ in specs]
    done = [False] * len(specs)
    eos = 200006
    with torch.no_grad():
        cur = [torch.tensor([s["ids"]], device=dev) for s in specs]
        for step in range(a.steps + 1):
            ts = time.time()
            live = [k for k in range(len(specs)) if not done[k]]
            if not live:
                break
            states = {}
            for k in live:
                e = emb(cur[k])
                past = caches[k].get_seq_length()
                pos = (torch.arange(cur[k].shape[1], device=dev) + past)[None]
                states[k] = {"h": e, "blocks": e.new_zeros(1, cur[k].shape[1], 0, e.shape[-1]),
                             "masks": R.masks(cfg, e, pos, caches[k]), "pos": pos, "taps": []}
            for i, layer in enumerate(layers):
                for k in live:
                    R.run_layer(layer, states[k], caches[k])
                    if i in TAP_LAYERS:
                        states[k]["taps"].append(states[k]["h"][0].cpu())
            for k in live:
                st = states[k]
                hn = R.finish(head, st)
                st["taps"].append(hn[0].cpu())
                taps[k].append(torch.stack(st["taps"], dim=1))  # [rows, 8, H]
                lg = R.logits_of(head, cfg, hn[0, -1:])[0]
                t = int(lg.argmax())
                if step < a.steps:
                    outs[k].append(t)
                if t == eos or step >= a.steps:
                    done[k] = True
                cur[k] = torch.tensor([[t]], device=dev)
            print(f"step {step}: {time.time() - ts:.1f} s, live {len(live)}, expert hits {lru.hits} misses "
                  f"{lru.misses}", flush=True)
    res = {}
    for k, s in enumerate(specs):
        ids = s["ids"] + outs[k]
        tp = torch.cat(taps[k])[: len(ids)]
        save_file({"ids": torch.tensor(ids, dtype=torch.int64), "n_prompt": torch.tensor([len(s["ids"])]),
                   "taps": tp.contiguous()}, os.path.join(a.out, s["name"] + ".safetensors"))
        res[s["name"]] = {"n_prompt": len(s["ids"]), "generated": len(outs[k]),
                          "text": tok.decode(outs[k], skip_special_tokens=False)}
        print(f"--- {s['name']} ({len(outs[k])} tokens)\n{res[s['name']]['text']}", flush=True)
    res["_wall_s"] = round(time.time() - t0, 1)
    res["_mode"] = a.mode
    json.dump(res, open(os.path.join(a.out, "gen.json"), "w"), indent=1, ensure_ascii=False)


# ----------------------------------------------------------------------------------------------- drafter


def rms(x, w, eps=EPS):
    """PrecisionRMSNorm: fp32 statistics, mean by reciprocal width, weight in fp32, bf16 out."""
    x = x.float()
    n = x.shape[-1]
    ms = x.square().sum(-1, keepdim=True) * (1.0 / n)
    return (x * torch.rsqrt(ms + eps) * w.float()).to(BF)


def add_norm(h, res, w):
    total = h.float() + res.float()
    return rms(total, w), total.to(BF)


def rope(x, pos, nheads):
    """Neox rotation, theta 1e6, bf16 cos/sin table (TokenSpeed's RotaryEmbedding cache dtype)."""
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float32, device=x.device) / HD))
    f = pos.to(torch.float32)[:, None] * inv[None, :]
    cos, sin = f.cos().to(BF).float()[:, None], f.sin().to(BF).float()[:, None]
    r = x.view(x.shape[0], nheads, HD).float()
    l, rr = r.chunk(2, -1)
    return torch.cat((l * cos - rr * sin, rr * cos + l * sin), -1).to(BF).view(x.shape[0], nheads * HD)


def head_norm(x, w, nheads):
    return rms(x.view(x.shape[0], nheads, HD), w).view(x.shape[0], nheads * HD)


def lin(x, w, b=None):
    return F.linear(x, w, b)


class Drafter:
    def __init__(self, model_dir: str, dev):
        self.dev = dev
        f = safe_open(os.path.join(model_dir, "mtp.safetensors"), framework="pt", device="cpu")
        g = lambda n: f.get_tensor(n).to(dev)
        self.alpha = torch.softmax(g("fuse_weight").float(), dim=-1).to(BF)  # [6, 8]
        self.mask_embed = g("mask_embed")
        self.norm = g("norm.weight")
        self.L = []
        for i in range(5):
            p = f"layers.{i}."
            self.L.append({k: g(p + n) for k, n in (
                ("dflare", "dflare_norm.weight"), ("in_ln", "input_layernorm.weight"),
                ("post_ln", "post_attention_layernorm.weight"), ("q", "self_attn.q_proj.weight"),
                ("k", "self_attn.k_proj.weight"), ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
                ("qn", "self_attn.q_norm.weight"), ("kn", "self_attn.k_norm.weight"),
                ("kc", "self_attn.k_proj_ctx.weight"), ("vc", "self_attn.v_proj_ctx.weight"),
                ("knc", "self_attn.k_norm_ctx.weight"), ("gate", "mlp.gate_proj.weight"),
                ("up", "mlp.up_proj.weight"), ("down", "mlp.down_proj.weight"))})
        m = "markov_head."
        self.m_dflare = g(m + "dflare_norm.weight")
        self.w1 = g(m + "markov_w1.weight")
        self.rnn_w, self.rnn_b = g(m + "rnn_proj.weight"), g(m + "rnn_proj.bias")
        self.ho_w, self.ho_b = g(m + "hidden_out.weight"), g(m + "hidden_out.bias")
        import inkling2_ref as R
        src = R.Source(model_dir)
        self.unembed = src.get(R.CK + "unembed.weight", dev).to(BF)
        self.embed_w = src.get(R.CK + "embed.weight", dev).to(BF)
        self.embed_norm = src.get(R.CK + "embed_norm.weight", dev)
        V = self.unembed.shape[0]
        self.fold_w = torch.empty(V, RANK, dtype=BF, device=dev)
        self.fold_b = torch.empty(V, dtype=BF, device=dev)
        for a in range(0, V, 4096):
            rows = self.unembed[a:a + 4096].double()
            self.fold_w[a:a + 4096] = (rows @ self.ho_w.double()).to(BF)
            self.fold_b[a:a + 4096] = (rows @ self.ho_b.double()).to(BF)

    def embed(self, ids):
        """Transformers' Inkling2NormedEmbedding: fp32 rms (mean as x^2.mean), weight, bf16."""
        e = self.embed_w[ids].float()
        return (e * torch.rsqrt(e.square().mean(-1, keepdim=True) + EPS) * self.embed_norm.float()).to(BF)

    def mix(self, taps, stage):
        """taps [T, 8, H] bf16 -> dflare_norm(sum_t tap_t * alpha[stage, t]) [T, H] bf16 (fp32 sum in tap order)."""
        acc = torch.zeros(taps.shape[0], H, dtype=torch.float32, device=taps.device)
        for t in range(taps.shape[1]):
            acc += taps[:, t].float() * self.alpha[stage, t].float()
        w = self.m_dflare if stage == 5 else self.L[stage]["dflare"]
        return rms(acc, w)

    def context(self, taps):
        """Every draft layer's context K (normed, rotated) and V for target positions 0..T-1, plus f_rnn rows."""
        T = taps.shape[0]
        pos = torch.arange(T, device=taps.device)
        ctx = []
        for i, Lw in enumerate(self.L):
            m = self.mix(taps, i)
            k = rope(head_norm(lin(m, Lw["kc"]), Lw["knc"], NKV), pos, NKV)
            v = lin(m, Lw["vc"])
            ctx.append((k, v))
        return ctx, self.mix(taps, 5)

    def block(self, anchor_id, p, ctx, causal=False, rec=None):
        """The 8-row block forward at positions p..p+7 against context positions [0, p): final rows [8, H]."""
        dev = self.dev
        x = torch.cat((self.embed(torch.tensor([anchor_id], device=dev)), self.mask_embed[None].expand(7, H)))
        pos = torch.arange(p, p + BLOCK, device=dev)
        if rec is not None:
            rec["block.in"] = x
        res, h = x, None
        for i, Lw in enumerate(self.L):
            if i == 0:
                hn = rms(x, Lw["in_ln"])
            else:
                hn, res = add_norm(h, res, Lw["in_ln"])
            q = rope(head_norm(lin(hn, Lw["q"]), Lw["qn"], NQ), pos, NQ)
            k = rope(head_norm(lin(hn, Lw["k"]), Lw["kn"], NKV), pos, NKV)
            v = lin(hn, Lw["v"])
            lo = max(0, p - WINDOW) if i in LOCAL else 0
            kc, vc = ctx[i]
            K = torch.cat((kc[lo:p], k)).view(-1, NKV, HD).float()
            Vv = torch.cat((vc[lo:p], v)).view(-1, NKV, HD).float()
            qh = q.view(BLOCK, NQ, HD).float()
            K = K.repeat_interleave(NQ // NKV, dim=1)
            Vv = Vv.repeat_interleave(NQ // NKV, dim=1)
            s = torch.einsum("qhd,khd->hqk", qh, K) * (HD ** -0.5)
            if causal:
                n = K.shape[0]
                mask = torch.ones(BLOCK, n, dtype=torch.bool, device=dev)
                mask[:, n - BLOCK:] = torch.tril(torch.ones(BLOCK, BLOCK, dtype=torch.bool, device=dev))
                s = s.masked_fill(~mask[None], float("-inf"))
            pr = torch.softmax(s, dim=-1)
            att = torch.einsum("hqk,khd->qhd", pr, Vv).reshape(BLOCK, NQ * HD).to(BF)
            o = lin(att, Lw["o"])
            hn2, res = add_norm(o, res, Lw["post_ln"])
            gt, up = lin(hn2, Lw["gate"]), lin(hn2, Lw["up"])
            act = (F.silu(gt.float()) * up.float()).to(BF)
            h = lin(act, Lw["down"])
            if rec is not None:
                rec.update({f"l{i}.q": q, f"l{i}.k": k, f"l{i}.v": v, f"l{i}.att": att, f"l{i}.o": o,
                            f"l{i}.act": act, f"l{i}.h": h, f"l{i}.res": res})
        out, _ = add_norm(h, res, self.norm)
        if rec is not None:
            rec["block.out"] = out
        return out

    def propose(self, anchor_id, rows, f, rec=None):
        """Markov head over rows [7, H] with features f [H]: 7 greedy proposals."""
        base = lin(rows, self.unembed)  # [7, V] bf16, uncapped
        state = torch.zeros(RANK, dtype=BF, device=self.dev)
        prev = anchor_id
        toks = []
        for s in range(rows.shape[0]):
            inp = torch.cat((state, self.w1[prev], rows[s], f))
            pj = lin(inp[None], self.rnn_w, self.rnn_b)[0]  # bf16
            g, c, o = pj.float().chunk(3)
            g = torch.sigmoid(g)
            state = (g * state.float() + (1 - g) * torch.tanh(c)).to(BF)
            latent = torch.tanh(o).to(BF)
            corr = torch.addmm(self.fold_b[None], latent[None], self.fold_w.T)[0]  # bf16
            lg = (SOFTCAP * torch.tanh((base[s] + corr).float() / SOFTCAP)).to(BF)
            lg[VALID:] = float("-inf")
            prev = int(lg.argmax())
            toks.append(prev)
            if rec is not None:
                rec[f"m{s}.proj"] = pj
                rec[f"m{s}.state"] = state
                rec[f"m{s}.logits"] = lg
        if rec is not None:
            rec["base"] = base
            rec["proposals"] = torch.tensor(toks, dtype=torch.int32)
        return toks


# ----------------------------------------------------------------------------------------------- accept


def accept(a):
    dev = torch.device(a.device)
    d = Drafter(a.model_dir, dev)
    out = {}
    with torch.no_grad():
        for path in a.files:
            name = os.path.basename(path).replace(".safetensors", "")
            t = load_file(path)
            ids, n0, taps = t["ids"].tolist(), int(t["n_prompt"][0]), t["taps"].to(dev)
            T = len(ids)
            ctx, frnn = d.context(taps)
            t0 = time.time()
            match = {}  # anchor position -> accepted drafts (matching prefix, 0..7)
            for p in range(n0, T - 1):
                rows = d.block(ids[p], p, ctx, causal=a.causal)[1:]
                props = d.propose(ids[p], rows, frnn[p - 1])
                k = 0
                while k < 7 and p + 1 + k < T and props[k] == ids[p + 1 + k]:
                    k += 1
                match[p] = (k, min(7, T - 1 - p))  # matched, how many could be checked
            per = {}
            for D in range(1, 8):
                p, rounds, toks = n0, 0, 0
                while p < T - 1:
                    k, avail = match[p]
                    if avail < D:  # the tail can't hold a full round
                        break
                    acc = min(k, D)
                    rounds += 1
                    toks += acc + 1
                    p += acc + 1
                per[D] = {"rounds": rounds, "tokens": toks, "mean_accepted": round(toks / max(rounds, 1), 3)}
            hist = [0] * 8
            for p, (k, avail) in match.items():
                if avail == 7:
                    hist[k] += 1
            out[name] = {"generated": T - n0, "anchors": len(match), "accepted_hist_at_7": hist,
                         "per_drafts": per, "seconds": round(time.time() - t0, 1)}
            print(name, json.dumps(out[name]), flush=True)
    tot = {}
    for D in range(1, 8):
        r = sum(v["per_drafts"][D]["rounds"] for v in out.values())
        k = sum(v["per_drafts"][D]["tokens"] for v in out.values())
        tot[D] = round(k / max(r, 1), 3)
    out["_mean_accepted_by_drafts"] = tot
    out["_causal"] = a.causal
    print("mean accepted length (tokens a round, bonus included) by drafts:", tot, flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


def fold(a):
    """The Markov head's folds (fp64 products, bf16 stores, TokenSpeed's refresh_fold) for the CUDA drafter."""
    import inkling2_ref as R
    torch.set_num_threads(os.cpu_count() or 8)
    f = safe_open(os.path.join(a.model_dir, "mtp.safetensors"), framework="pt", device="cpu")
    ho_w = f.get_tensor("markov_head.hidden_out.weight").double()
    ho_b = f.get_tensor("markov_head.hidden_out.bias").double()
    un = R.Source(a.model_dir).get(R.CK + "unembed.weight", "cpu").to(BF)
    V = un.shape[0]
    fw = torch.empty(V, RANK, dtype=BF)
    fb = torch.empty(V, dtype=BF)
    t0 = time.time()
    for s in range(0, V, 4096):
        rows = un[s:s + 4096].double()
        fw[s:s + 4096] = (rows @ ho_w).to(BF)
        fb[s:s + 4096] = (rows @ ho_b).to(BF)
    save_file({"fold_w": fw.contiguous(), "fold_b": fb.contiguous()}, a.out)
    print(f"folds {tuple(fw.shape)} in {time.time() - t0:.1f} s -> {a.out}")


def check(a):
    """The CUDA drafter's dump (tf-inkling2-draft --out) against this reference's (`dump`): relative errors."""
    x, r = load_file(a.cuda), load_file(a.ref)
    ok = True
    for k in ["block.out", "frnn", "base"] + [f"ctx{i}.{kv}" for i in range(5) for kv in "kv"]:
        if k not in x or k not in r:
            continue
        p, q = x[k].float(), r[k].float()
        if k.startswith("ctx") and p.dim() == 2 and q.dim() == 2 and p.shape[0] < q.shape[0] and p.shape[1] == q.shape[1]:
            q = q[q.shape[0] - p.shape[0]:]  # a local layer's ring: the last positions only
        if p.shape != q.shape:
            print(f"{k}: shape {tuple(p.shape)} vs {tuple(q.shape)}")
            ok = False
            continue
        rel = float((p - q).norm() / q.norm().clamp_min(1e-30))
        same = float((p == q).float().mean())
        print(f"{k:10s} rel {rel:.3e}  bit-equal {same:.4f}  max |diff| {float((p - q).abs().max()):.3e}")
        ok = ok and rel < a.tol
    pr, pq = x["proposals"].tolist(), r["proposals"].tolist()
    print("proposals cuda", pr, "\n          ref ", pq)
    print("PASS" if ok and pr == pq else "FAIL")


def dump(a):
    dev = torch.device(a.device)
    d = Drafter(a.model_dir, dev)
    t = load_file(a.file)
    ids, taps = t["ids"], t["taps"].to(dev)
    p = a.anchor
    with torch.no_grad():
        ctx, frnn = d.context(taps[:p])
        rec = {}
        rows = d.block(int(ids[p]), p, ctx, rec=rec)[1:]
        d.propose(int(ids[p]), rows, frnn[p - 1], rec=rec)
    rec["ids"] = ids[: p + 1]
    rec["taps"] = taps[:p]
    rec["anchor"] = torch.tensor([p])
    rec["frnn"] = frnn[p - 1]
    for i in range(5):
        rec[f"ctx{i}.k"], rec[f"ctx{i}.v"] = ctx[i]
    save_file({k: v.detach().cpu().contiguous() for k, v in rec.items()}, a.out)
    print("proposals", rec["proposals"].tolist(), "targets", t["ids"][p + 1:p + 8].tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("model_dir")
    g.add_argument("out")
    g.add_argument("--prompts", default="chat_long,code,tool_call,thinking")
    g.add_argument("--steps", type=int, default=160)
    g.add_argument("--mode", choices=["full", "checkpoint"], default="checkpoint")
    g.add_argument("--expert-cache", type=int, default=1500)
    g.add_argument("--device", default="cuda:0")
    c = sub.add_parser("accept")
    c.add_argument("model_dir")
    c.add_argument("files", nargs="+")
    c.add_argument("--json", default="")
    c.add_argument("--causal", action="store_true", help="causal inside the block (a check of the trained form)")
    c.add_argument("--device", default="cuda:0")
    ch = sub.add_parser("check")
    ch.add_argument("cuda")
    ch.add_argument("ref")
    ch.add_argument("--tol", type=float, default=2e-2)
    fo = sub.add_parser("fold")
    fo.add_argument("model_dir")
    fo.add_argument("out")
    u = sub.add_parser("dump")
    u.add_argument("model_dir")
    u.add_argument("file")
    u.add_argument("--anchor", type=int, required=True)
    u.add_argument("--out", required=True)
    u.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"gen": gen, "accept": accept, "dump": dump, "fold": fold, "check": check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
