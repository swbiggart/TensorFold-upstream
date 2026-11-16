"""Inkling-2's forward op by op in torch, in tf-inkling2-cuda's op names and layouts: the per-op oracle.

Every op of the CUDA forward (zig/src/families/inkling2/cuda_forward.zig) is one function here, built from the
Transformers modules the oracle uses (tools/zig/inkling2_ref.py loads them; Transformers is Apache-2.0 and is
imported; `tau_of` and `dyn_conv` adapt parts of it and carry its notice). Tensors are [rows, columns] as the
runner dumps them: op.<scope>.<group>.<op>.<in|out>.<t> with scope the layer number or "model"; a convolution state
or gathered cache rows carry "@<first row>".

USAGE = python tools/zig/inkling2_op_ref.py capture MODEL_DIR ORACLE_PROMPT --layers A:B --out CAPTURE [--head]
            [--fp32-reduction]
        python tools/zig/inkling2_op_ref.py check MODEL_DIR DUMP [--tol-rel 0.01] [--ops a,b] [--fp32-reduction]
        python tools/zig/inkling2_op_ref.py layers MODEL_DIR ORACLE_PROMPT --layers A:B --out FILE [--head]
            [--teacher] [--fp32-reduction]

  capture  From the oracle's input of layer A (h.<A-1> or embed, and the block states below A), run layers
           A..B-1 (and with --head the final mix, norm, unembed and softcap at logits.rows) op by op in torch,
           and save every op's inputs and outputs. The chain's layer outputs are checked against the oracle's
           h.<L> (bit-equal expected: the same modules on the same inputs). Feed it to the runner
           (--feed CAPTURE): each op then runs on torch's inputs, and ops without a kernel pass with torch's
           outputs; inkling2_compare.py --ops compares the runner's op outputs with the capture. With
           --fp32-reduction (as for layers) the capture is the tight per-op reference.
  check    For a runner dump made with --dump-ops, recompute each op from the inputs the runner recorded and
           compare with the runner's outputs (an op tested on exactly what it was given).
  layers   The same op chain keeping only what crosses layers, in the oracle's format (ids, rows, h.<L>,
           route.*.<L>, final, logits.rows, logits.full; every row), so the file serves inkling2_compare.py both
           as a dump and as an oracle. --teacher starts every layer from the oracle's input (as tf-inkling2-cuda
           --teacher does; with --head the final mix runs on the oracle's h.47 and blocks and the unembed on its
           final rows). --fp32-reduction turns off cuBLAS's bf16 partial reduction (torch's
           allow_bf16_reduced_precision_reduction, on in the oracle): the same layer on the same input with its
           bf16 matmuls reduced in fp32, a tighter reference that tells the oracle's own rounding from ours.

Lane recipe (one op group alone), e.g. G-conv on layer 5 with the chat_short prompt:
  python tools/zig/inkling2_op_ref.py capture MODEL ORACLE/full/chat_short.safetensors --layers 5:6 --out cap5.st
  tf-inkling2-cuda MODEL ORACLE/full/chat_short.safetensors run5.st --layers 5:6 --feed cap5.st --dump-ops
  python tools/zig/inkling2_compare.py run5.st ORACLE/full/chat_short.safetensors --ops cap5.st
The compare lists every implemented op's output against torch; ops listed as fed stubs are skipped.
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling2_ref as ref  # noqa: E402

BF = torch.bfloat16
H = 4608
D = 128
WINDOW = 1024


# ----------------------------------------------------------------------------------------------- the ops


# tau_of and dyn_conv: adapted from huggingface/transformers src/transformers/models/inkling2/modeling_inkling2.py
# at 109a7d19 (the attention's log-scaling tau and Inkling2DynamicShortConvolution.forward), Copyright 2026 the
# HuggingFace Team, Apache License 2.0; modified for TensorFold: split into ops, the per-token taps and the
# convolution state passed in (see THIRD_PARTY_NOTICES.md).
def tau_of(cfg, pos):
    t = 1.0 + cfg.log_scaling_alpha * torch.log(((pos + 1).float() / cfg.log_scaling_n_floor).clamp(min=1.0))
    return t[:, None, None]


def short_conv(mod, x, state):
    """Inkling2ShortConvolution over [state; x] (state: the 3 input rows before x), the last R rows. A zero state
    (a stream's start) runs the module on x alone, exactly as Transformers' teacher-forced pass does."""
    if not state.any():
        return mod(x[None])[0]
    ext = torch.cat([state.to(x.dtype), x], 0)[None]
    return mod(ext)[0, state.shape[0]:]


def dyn_conv(mod, x, dyn, state):
    """Inkling2DynamicShortConvolution's forward with its per-token taps given (`dyn` = weight_proj's output)."""
    if not state.any():
        state = state[:0]
    n = state.shape[0]
    ext = torch.cat([state.to(x.dtype), x], 0)[None]
    dext = torch.cat([dyn.new_zeros(n, dyn.shape[1]), dyn], 0)[None]
    batch_size, seq_len, hidden_size = ext.shape
    k = mod.conv_kernel_size
    dynamic_weights = dext.float().view(batch_size, seq_len, mod.num_heads, 1, k)
    weights = mod.static_weight.float().T.view(mod.num_heads, -1, k) + dynamic_weights
    hidden = ext.float()
    residual = hidden
    context = hidden.transpose(1, 2)
    context = F.pad(context, (k - 1, 0))[..., -(seq_len + k - 1):]
    windows = context.unfold(-1, k, 1).flip(-1).transpose(1, 2)
    windows = windows.reshape(batch_size, seq_len, mod.num_heads, -1, k)
    out = (windows * weights).sum(-1).reshape(batch_size, seq_len, hidden_size)
    out = (out + residual) / math.sqrt(2 * k + 1)
    return out.to(x.dtype)[0, n:]


def stack_blocks(ins):
    bl = [ins[f"block{i}"] for i in range(8) if f"block{i}" in ins]
    cand = ins["cand"]
    if not bl:
        return cand.new_zeros(1, cand.shape[0], 0, H).to(BF)
    return torch.stack(bl, dim=-2)[None]


class Ops:
    """The ops of one layer (or the model head) over its Transformers modules."""

    def __init__(self, cfg, layer=None, head=None, index=None):
        self.cfg, self.ly, self.head, self.L = cfg, layer, head, index

    def run(self, op, ins, pos0, kv_lo=None):
        ly, cfg = self.ly, self.cfg
        R = next(v.shape[0] for k, v in ins.items() if "@" not in k)
        pos = torch.arange(pos0, pos0 + R, device=next(iter(ins.values())).device)
        if op == "res.attn_input":
            outs = ly.attention_input(stack_blocks(ins), ins["cand"][None])
            return {f"x{j}": o[0] for j, o in enumerate(outs)}
        if op == "res.mlp_input":
            return {"x": ly.mlp_input(stack_blocks(ins), ins["cand"][None])[0][0]}
        if op == "res.final_input":
            fi = self.head[0]
            return {"x": fi(stack_blocks(ins), ins["cand"][None])[0][0]}
        if op in ("res.attn_norm", "res.q_norm", "res.k_norm", "res.v_norm", "res.mlp_norm"):
            mod = {"res.attn_norm": "input_layernorm", "res.q_norm": "q_input_layernorm", "res.k_norm": "k_input_layernorm",
                   "res.v_norm": "v_input_layernorm", "res.mlp_norm": "post_attention_layernorm"}[op]
            return {"y": getattr(ly, mod)(ins["x"]).to(BF)}
        if op == "attn.project":
            sa = ly.self_attn
            xq = ins.get("xq", ins.get("x"))
            xk = ins.get("xk", xq)
            xv = ins.get("xv", xq)
            out = {"q": sa.q_proj(xq), "k": sa.k_proj(xk), "v": sa.v_proj(xv), "rel": sa.r_proj(xq),
                   "gate": sa.gate_proj(xq), "sink": sa.sink_proj(xq)}
            if not sa.is_sliding:
                out["dk"] = sa.k_sconv.weight_proj(xk)
                out["dv"] = sa.v_sconv.weight_proj(xv)
            return out
        if op == "conv.kv":
            sa = ly.self_attn
            sk = state_of(ins, "state_k", pos0)
            sv = state_of(ins, "state_v", pos0)
            if sa.is_sliding:
                return {"kc": short_conv(sa.k_sconv, ins["k"], sk), "vc": short_conv(sa.v_sconv, ins["v"], sv)}
            return {"kc": dyn_conv(sa.k_sconv, ins["k"], ins["dk"], sk), "vc": dyn_conv(sa.v_sconv, ins["v"], ins["dv"], sv)}
        if op in ("conv.attn", "conv.mlp"):
            mod = ly.post_attention_convolution if op == "conv.attn" else ly.post_mlp_convolution
            return {"y": short_conv(mod, ins["x"], state_of(ins, "state", pos0))}
        if op == "attn.heads":
            sa = ly.self_attn
            hq, hk = sa.num_heads, sa.num_key_value_heads
            q = sa.q_l2_clip(ins["q"].view(R, hq, D))
            k = sa.k_l2_clip(ins["kc"].view(R, hk, D))
            v = sa.v_norm(ins["vc"].view(R, hk, D).float()).to(BF)
            rel = sa.rel_l2_clip(ins["rel"].view(R, hq, -1))
            if not sa.is_sliding:
                t = tau_of(cfg, pos)
                q = (q.float() * t).to(q.dtype)
                rel = (rel.float() * t).to(rel.dtype)
            return {"q": q.reshape(R, -1), "k": k.reshape(R, -1), "v": v.reshape(R, -1), "rel": rel.reshape(R, -1)}
        if op == "attn.core":
            return {"o": self.core(ins, pos0, R, kv_lo)}
        if op == "attn.out":
            return {"ao": ly.self_attn.o_proj(ins["o"])}
        if op == "res.partial":
            a = ins["a"].float()
            return {"partial": a if ly.is_block_boundary else ins["h_in"].float() + a}
        if op == "moe.route":
            logits, w, idx = ly.mlp.gate(ins["m"])
            order = idx.argsort(-1)
            return {"ids": idx.gather(-1, order).to(torch.int32), "w": w.gather(-1, order).float(), "logits": logits.float()}
        if op == "moe.experts":
            return {"y": ly.mlp.experts(ins["m"], ins["ids"].long(), ins["w"].float())}
        if op == "res.output":
            return {"h": (ins["partial"] + ins["mconv"].float()).to(BF)}
        if op == "head.final_norm":
            return {"y": self.head[1](ins["x"]).to(BF)}
        if op == "head.unembed":
            return {"logits": self.head[2](ins["y"])}
        if op == "head.softcap":
            return {"out": self.head[3](ins["logits"].float())[..., : cfg.unpadded_vocab_size]}
        raise KeyError(f"no torch form for {op}")

    def core(self, ins, pos0, R, kv_lo):
        from transformers.models.inkling2.modeling_inkling2 import eager_attention_forward

        sa = self.ly.self_attn
        hq, hk = sa.num_heads, sa.num_key_value_heads
        kx = find_whole(ins, "k_ctx", pos0)
        vx = find_whole(ins, "v_ctx", pos0)
        if kx is None:  # a capture of one forward from position 0: the context is this forward's own rows
            kx, vx, kv_lo = ins["k"], ins["v"], pos0
        C = kx.shape[0]
        lo = pos0 + R - C if kv_lo is None else kv_lo
        dev = kx.device
        qpos = torch.arange(pos0, pos0 + R, device=dev)
        kpos = torch.arange(lo, lo + C, device=dev)
        q = ins["q"].view(R, hq, D).transpose(0, 1)[None]
        k = kx.view(C, hk, D).transpose(0, 1)[None]
        v = vx.view(C, hk, D).transpose(0, 1)[None]
        rel = ins["rel"].view(R, hq, -1)[None]
        bias = sa.rel_logits_proj(rel, qpos, kpos)
        dist = qpos[:, None] - kpos[None, :]
        ok = dist >= 0
        if sa.is_sliding:
            ok = ok & (dist < sa.sliding_window)
        mask = torch.zeros(R, C, dtype=q.dtype, device=dev).masked_fill(~ok, torch.finfo(q.dtype).min)[None, None]
        sinks = ins["sink"][None].transpose(1, 2).unsqueeze(-1)
        o, _ = eager_attention_forward(sa, q, k, v, mask, dropout=0.0, scaling=sa.scaling, sliding_window=sa.sliding_window,
                                       position_bias=bias, sinks=sinks)
        o = o.reshape(R, -1).contiguous()
        return (o.float() * torch.sigmoid(ins["gate"].float())).to(o.dtype)


def find_whole(ins, name, pos0):
    return ins.get(f"{name}@{pos0}")


def state_of(ins, name, pos0):
    s = find_whole(ins, name, pos0)
    if s is not None:
        return s
    x = next(v for k, v in ins.items() if "@" not in k)
    return x.new_zeros(3, x.shape[1])


# ----------------------------------------------------------------------------------------------- capture


class Capture:
    def __init__(self, dev):
        self.t, self.dev = {}, dev

    def put(self, scope, op, d, ts):
        for k, v in ts.items():
            self.t[f"op.{scope}.{op}.{d}.{k}"] = v.detach().contiguous().cpu()

    def op(self, ops, scope, op, ins, pos0=0, kv_lo=None):
        self.put(scope, op, "in", ins)
        with torch.no_grad():
            outs = ops.run(op, ins, pos0, kv_lo)
        self.put(scope, op, "out", outs)
        return outs


def equal_report(name, got, want):
    g, w = got.float().cpu(), want.float().cpu()
    eq = float((got.cpu() == want.cpu()).float().mean())
    rel = float((g - w).norm() / w.norm().clamp(min=1e-30))
    print(f"  {name}: equal {eq:.6f}  max_abs {float((g - w).abs().max()):.3e}  rel {rel:.3e}", flush=True)
    return eq


def layer_chain(cap, ops, L, h, blocks, cfg):
    """One decoder layer as the runner's op sequence; returns h_out and the blocks after it."""
    ly = ops.ly
    s = str(L)
    zero = lambda c: h.new_zeros(3, c)  # noqa: E731
    if ly.attention_input is not None:
        ins = {f"block{i}": b for i, b in enumerate(blocks)} | {"cand": h}
        xs = list(cap.op(ops, s, "res.attn_input", ins).values())
    else:
        xs = [h]
    sliding = ly.self_attn.is_sliding
    if sliding:
        xn = [cap.op(ops, s, "res.attn_norm", {"x": xs[0]})["y"]] * 3
        p = cap.op(ops, s, "attn.project", {"x": xn[0]})
    else:
        if len(xs) == 1:
            xs = xs * 3
        xn = [cap.op(ops, s, f"res.{n}_norm", {"x": x})["y"] for n, x in zip("qkv", xs)]
        p = cap.op(ops, s, "attn.project", {"xq": xn[0], "xk": xn[1], "xv": xn[2]})
    kvc = p["k"].shape[1]
    ins = {"k": p["k"], "v": p["v"], "state_k@0": zero(kvc), "state_v@0": zero(kvc)}
    if not sliding:
        ins |= {"dk": p["dk"], "dv": p["dv"]}
    c = cap.op(ops, s, "conv.kv", ins)
    hd = cap.op(ops, s, "attn.heads", {"q": p["q"], "kc": c["kc"], "vc": c["vc"], "rel": p["rel"]})
    o = cap.op(ops, s, "attn.core", {"q": hd["q"], "rel": hd["rel"], "sink": p["sink"], "gate": p["gate"],
                                     "k_ctx@0": hd["k"], "v_ctx@0": hd["v"]}, kv_lo=0)["o"]
    ao = cap.op(ops, s, "attn.out", {"o": o})["ao"]
    a = cap.op(ops, s, "conv.attn", {"x": ao, "state@0": zero(H)})["y"]
    part = cap.op(ops, s, "res.partial", {"h_in": h, "a": a})["partial"]
    if ly.is_block_boundary:
        blocks = blocks + [h]
    ins = {f"block{i}": b for i, b in enumerate(blocks)} | {"cand": part}
    mx = cap.op(ops, s, "res.mlp_input", ins)["x"]
    m = cap.op(ops, s, "res.mlp_norm", {"x": mx})["y"]
    r = cap.op(ops, s, "moe.route", {"m": m})
    y = cap.op(ops, s, "moe.experts", {"m": m, "ids": r["ids"], "w": r["w"]})["y"]
    cm = cap.op(ops, s, "conv.mlp", {"x": y, "state@0": zero(H)})["y"]
    h_out = cap.op(ops, s, "res.output", {"partial": part, "mconv": cm})["h"]
    return h_out, blocks, r


def capture(a):
    dev = torch.device(a.device)
    lo, hi = (int(x) for x in a.layers.split(":"))
    cfg = ref.text_config(a.model_dir, 0)
    model = ref.Model(a.model_dir, cfg, dev, "full")
    o = safe_open(a.oracle, framework="pt")
    T = o.get_tensor("ids").shape[0]
    if o.get_tensor("rows").shape[0] != T:
        raise SystemExit("the oracle keeps sampled rows of this prompt: capture needs a prompt of at most 512 rows")
    rows_of = lambda n: o.get_tensor(n).to(dev)  # noqa: E731
    h = rows_of("embed") if lo == 0 else rows_of(f"h.{lo - 1}")
    blocks = [rows_of("embed") if b == 0 else rows_of(f"h.{b - 1}") for b in range(0, lo, 6)]
    cap = Capture(dev)
    worst = 1.0
    for L in range(lo, hi):
        t0 = time.time()
        ly = model.layer(L)
        t1 = time.time()
        h, blocks, r = layer_chain(cap, Ops(cfg, ly, index=L), L, h, blocks, cfg)
        torch.cuda.synchronize()
        print(f"layer {L}: load {t1 - t0:.1f} s, ops {time.time() - t1:.1f} s; against the oracle:", flush=True)
        worst = min(worst, equal_report(f"h.{L}", h, rows_of(f"h.{L}")))
        equal_report(f"route.ids.{L}", r["ids"], rows_of(f"route.ids.{L}"))
        del ly
        torch.cuda.empty_cache()
    if a.head:
        if hi != cfg.num_hidden_layers:
            raise SystemExit("--head needs the last layer")
        head = model.head()
        ops = Ops(cfg, head=head)
        ins = {f"block{i}": b for i, b in enumerate(blocks)} | {"cand": h}
        x = cap.op(ops, "model", "res.final_input", ins)["x"]
        y = cap.op(ops, "model", "head.final_norm", {"x": x})["y"]
        equal_report("final", y, rows_of("final"))
        lr = o.get_tensor("logits.rows").to(dev)
        lg = cap.op(ops, "model", "head.unembed", {"y": y[lr]})["logits"]
        out = cap.op(ops, "model", "head.softcap", {"logits": lg})["out"]
        equal_report("logits.full", out, rows_of("logits.full"))
    meta = {"tool": "inkling2_op_ref.py capture", "oracle": os.path.basename(a.oracle), "layers": a.layers,
            "rows": str(T), "worst_layer_equal_frac": f"{worst:.6f}",
            "bf16_reduced_precision_reduction": str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)}
    save_file(cap.t, a.out, metadata=meta)
    print(f"{len(cap.t)} tensors to {a.out}; worst layer bit-equal fraction {worst:.6f}")
    return 0 if worst == 1.0 else 1


# ----------------------------------------------------------------------------------------------- layers


class NoCapture(Capture):
    def put(self, scope, op, d, ts):
        pass


def layers(a):
    dev = torch.device(a.device)
    lo, hi = (int(x) for x in a.layers.split(":"))
    cfg = ref.text_config(a.model_dir, 0)
    model = ref.Model(a.model_dir, cfg, dev, "full")
    o = safe_open(a.oracle, framework="pt")
    ids = o.get_tensor("ids")
    T = ids.shape[0]
    if o.get_tensor("rows").shape[0] != T:
        raise SystemExit("the oracle keeps sampled rows of this prompt: run inkling2_ref.py --all-rows")
    rows_of = lambda n: o.get_tensor(n).to(dev)  # noqa: E731

    def state(L):
        h = rows_of("embed") if L == 0 else rows_of(f"h.{L - 1}")
        return h, [rows_of("embed") if b == 0 else rows_of(f"h.{b - 1}") for b in range(0, L, 6)]

    h, blocks = state(lo)
    cap = NoCapture(dev)
    out = {"ids": ids, "rows": torch.arange(T)}
    for L in range(lo, hi):
        if a.teacher and L > lo:
            h, blocks = state(L)
        t0 = time.time()
        ly = model.layer(L)
        t1 = time.time()
        h, blocks, r = layer_chain(cap, Ops(cfg, ly, index=L), L, h, blocks, cfg)
        torch.cuda.synchronize()
        print(f"layer {L}: load {t1 - t0:.1f} s, ops {time.time() - t1:.1f} s; against the oracle:", flush=True)
        equal_report(f"h.{L}", h, rows_of(f"h.{L}"))
        equal_report(f"route.ids.{L}", r["ids"], rows_of(f"route.ids.{L}"))
        out[f"h.{L}"] = h.cpu()
        for k, n in (("ids", "route.ids"), ("w", "route.w"), ("logits", "route.logits")):
            out[f"{n}.{L}"] = r[k].cpu()
        del ly
        torch.cuda.empty_cache()
    if a.head:
        if hi != cfg.num_hidden_layers:
            raise SystemExit("--head needs the last layer")
        if a.teacher:
            h, blocks = state(hi)
        head = model.head()
        ops = Ops(cfg, head=head)
        ins = {f"block{i}": b for i, b in enumerate(blocks)} | {"cand": h}
        with torch.no_grad():
            y = ops.run("head.final_norm", {"x": ops.run("res.final_input", ins, 0)["x"]}, 0)["y"]
        equal_report("final", y, rows_of("final"))
        lr = o.get_tensor("logits.rows").to(dev)
        src = rows_of("final") if a.teacher else y
        with torch.no_grad():
            lg = ops.run("head.unembed", {"y": src[lr]}, 0)["logits"]
            lo_ = ops.run("head.softcap", {"logits": lg}, 0)["out"]
        equal_report("logits.full", lo_, rows_of("logits.full"))
        out |= {"final": y.cpu(), "logits.rows": lr.cpu(), "logits.full": lo_.cpu()}
    meta = {"tool": "inkling2_op_ref.py layers", "prompt": os.path.basename(a.oracle), "layers": a.layers,
            "rows": str(T), "rows_done": str(T), "status": "complete", "chunk": str(T),
            "inputs": "oracle at every layer" if a.teacher else "oracle at the first layer, then chained",
            "bf16_reduced_precision_reduction": str(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)}
    save_file({k: v.contiguous() for k, v in out.items()}, a.out, metadata=meta)
    print(f"{len(out)} tensors to {a.out}")
    return 0


# ----------------------------------------------------------------------------------------------- check


def check(a):
    from inkling2_compare import Store, line, op_parts, stats

    dev = torch.device(a.device)
    dump = Store(a.dump)
    f = safe_open(a.dump, framework="pt")
    T = int(dump.meta.get("rows", "0"))
    chunk = int(dump.meta.get("chunk", "0")) or T
    want = set(a.ops.split(",")) if a.ops else None
    by = {}
    for k in dump.keys:
        p = op_parts(k)
        if p is not None:
            by.setdefault((p[0], p[1]), {"in": [], "out": []})[p[2]].append(p[3])
    cfg = ref.text_config(a.model_dir, 0)
    model = ref.Model(a.model_dir, cfg, dev, "full")
    bad, n = [], 0
    layers = sorted({s for s, _ in by if s != "model"}, key=int)
    for scope in layers + (["model"] if any(s == "model" for s, _ in by) else []):
        mods = Ops(cfg, layer=model.layer(int(scope)), index=int(scope)) if scope != "model" else Ops(cfg, head=model.head())
        for (s, op), names in sorted(by.items()):
            if s != scope or not names["out"] or not names["in"] or (want and op not in want):
                continue
            selected = op in ("head.unembed", "head.softcap")  # the head's selected rows, in one piece
            n_rows = f.get_slice(f"op.{s}.{op}.out.{names['out'][0]}").get_shape()[0] if selected else T
            for start in range(0, n_rows, n_rows if selected else chunk):
                R = n_rows if selected else min(chunk, T - start)
                ins = {}
                for t in names["in"]:
                    v = f.get_tensor(f"op.{s}.{op}.in.{t}").to(dev)
                    if "@" in t:
                        if t.endswith(f"@{start}"):
                            ins[t] = v
                    else:
                        ins[t] = v[start:start + R]
                with torch.no_grad():
                    got = mods.run(op, ins, start)
                for t in names["out"]:
                    ours = f.get_tensor(f"op.{s}.{op}.out.{t}")[start:start + R]
                    ref_t = got[t].cpu()
                    if ref_t.dtype == BF:
                        st = stats(ours.view(torch.int16).numpy().view("uint16"), ref_t.view(torch.int16).numpy().view("uint16"), "bf16")
                    else:
                        st = stats(ours.numpy(), ref_t.numpy(), str(ref_t.dtype).replace("torch.", ""))
                    n += 1
                    ok = st["equal_frac"] == 1.0 if "rel" not in st else st["rel"] <= a.tol_rel
                    print(line(f"op.{s}.{op}.{t}@{start}", st) + ("" if ok else "  FAIL"))
                    if not ok:
                        bad.append(f"{s}.{op}.{t}")
        torch.cuda.empty_cache()
    if n == 0:
        print("nothing to check: run the runner with --dump-ops")
        return 2
    print(f"{n} op outputs recomputed; " + (f"FAIL {len(bad)}: {', '.join(bad[:12])}" if bad else "all within tolerance"))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("model_dir")
    c.add_argument("oracle")
    c.add_argument("--layers", required=True, help="A:B")
    c.add_argument("--out", required=True)
    c.add_argument("--head", action="store_true")
    c.add_argument("--fp32-reduction", action="store_true", help="bf16 matmuls reduced in fp32")
    c.add_argument("--device", default="cuda:0")
    y = sub.add_parser("layers")
    y.add_argument("model_dir")
    y.add_argument("oracle")
    y.add_argument("--layers", required=True, help="A:B")
    y.add_argument("--out", required=True)
    y.add_argument("--head", action="store_true")
    y.add_argument("--teacher", action="store_true", help="every layer from the oracle's input")
    y.add_argument("--fp32-reduction", action="store_true", help="bf16 matmuls reduced in fp32")
    y.add_argument("--device", default="cuda:0")
    k = sub.add_parser("check")
    k.add_argument("model_dir")
    k.add_argument("dump")
    k.add_argument("--tol-rel", type=float, default=1e-2)
    k.add_argument("--ops", default="", help="only these ops (comma-separated, e.g. conv.kv,attn.heads)")
    k.add_argument("--fp32-reduction", action="store_true", help="bf16 matmuls reduced in fp32")
    k.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if getattr(a, "fp32_reduction", False):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return {"capture": capture, "check": check, "layers": layers}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
