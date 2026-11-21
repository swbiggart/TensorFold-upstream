"""Inkling-2 vision tower gates V2-V5: the native tower's tables, ops, output rows and pinned bits against the image
oracle's files (tools/zig/inkling2_vision_ref.py). Every command prints one JSON verdict on stdout (progress goes to
stderr), carries the counts of the tables, ops, rows, layers and fixtures it looked at, and exits 1 on any FAIL. A
verdict over zero items is EMPTY, which counts as a failure: PASS is printed only over a non-empty, complete set.

USAGE = python tools/zig/inkling2_vision_check.py v2 TABLES OURS [--strict-nan] [--json OUT]
        python tools/zig/inkling2_vision_check.py v3 --model DIR --torch FIXTURE.sdpa.ops.safetensors
                   --ours OURS.ops.safetensors [--ours-from-torch] [--grid R,C] [--threads N] [--json OUT]
        python tools/zig/inkling2_vision_check.py v4 --forms-dir DIR --fixture NAME --ours OURS.safetensors
                   [--bar-forms sdpa,sdpa+red32,...] [--rows-only] [--json OUT]
        python tools/zig/inkling2_vision_check.py v5 FILE... [--key rows] [--variants a,b,...] [--json OUT]

  v2  Bit-equal tables. TABLES is the oracle's `tables` output (tables.safetensors); OURS uses the same key names:
        gelu.in, gelu.out          all 65,536 bf16 inputs and torch's erf GELU of each (on the GPU model of the run)
        rope.inv_freq [16], rope.cos / rope.sin [416, 16]   fp32, angles pos * inv_freq for pos 0..415
        pos.idx.<r>x<c> [4rc, 4], pos.w.<r>x<c> [4rc, 4], pos.ids.<r>x<c> [4rc, 2]
                                   bilinear taps, fp32 weights and (row, col) ids of a token grid, patches in block
                                   order (Transformers' vision_utils); ids without the model's +1
      Each table key of TABLES must be in OURS with the same shape and the same bits (floats compared as their bit
      patterns; integer tables by value, any integer width). Two NaNs count as equal whatever their payloads unless
      --strict-nan (the count is reported). Keys only OURS has are listed and not judged.

  v3  Per op, on torch default's recorded inputs (the oracle's `tower --ops` file of form sdpa; one image). Each op
      is recomputed here in float64 from TORCH'S RECORDED INPUTS with the checkpoint's weights (bf16 -> f64), and
      both torch default's output and ours are measured against it: normwise rel error (inkling2_f64.rel) and max
      |diff|; also ours' bit-equal share with torch, the share of ours equal to bf16(fp64), and the rounding floor
      rel(bf16(fp64)) with torch's distance over it. PASS per op iff rel(ours) <= rel(torch default) (with --tie-rel
      TIE: <= rel(torch) * (1 + TIE); the strict verdict is kept beside it as verdict_strict). OURS is the native
      tool's dump made from the same inputs, with these keys (the same shapes as torch's; [S, ...] rows with row 0
      the class token, patches in block order):
        E2.patch.out        from E2.patch.in
        E4.embeddings.out   from torch's E2.patch.out + the position sum (taps and fp32 weights from vision_utils as
                            the model computes them) and the class row class_embedding + cls_position_embedding
        E5.ln_pre.out       from E5.ln_pre.in
        L<i>.L1.ln_1.out, L<i>.L2.{q,k,v}_proj.out    from L<i>.L1.ln_1.in / L<i>.L2.{q,k,v}_proj.in
        L<i>.rot.{q,k}_out  from L<i>.rot.{q,k}_in [S, 16, 64]
        L<i>.core.out       [1, 16, S, 64] from L<i>.core.{q,k,v}: softmax over all keys, scale 0.125, query blocks
        L<i>.L2.out_proj.out
        L<i>.res1.out       layer.in + lambda_1 * out_proj.out (torch's L<i>.L7.ln_2.in)
        L<i>.L7.ln_2.out
        L<i>.L8.gelu.out    gelu(fc1(L<i>.L8.fc1.in)), erf form (torch's L<i>.L8.gelu.out)
        L<i>.L8.fc2.out
        L<i>.res2.out       L7.ln_2.in + lambda_2 * fc2.out (torch's L<i>.layer.out)
        M2.linear_1.out, M2.gelu.out (from M2.gelu.in), M2.linear_2.out
      for every layer i the torch file recorded. fp64 rotary (a choice, the same as the oracle's fp64 form): inv_freq
      = 1 / theta^(2j/32) in float64, angles = position * inv_freq in float64 (positions: patches (row, col) + 1,
      the class row (0, 0); column angles in the first 16 pairs), cos/sin and the rotation in float64. The fp64 cos
      is checked against torch's recorded fp32 cos/sin (a layout check; it stops on a mismatch above 1e-4).
      --ours-from-torch reads OURS as another oracle ops file and maps torch's keys to ours' (the self-test; that
      file's ops ran on its own inputs, so past the first op its numbers are not a same-input comparison).
      --grid R,C, else the grid of the sibling rows file (FIXTURE.sdpa.safetensors), else OURS' `grid`.

  v4  Free-running output. DIR holds the oracle's tower files <NAME>.<form>.safetensors (rows [T, 4608] and the
      sampled layer rows layer.<i> at layer.rows); <NAME>.fp64.safetensors is the reference. OURS has `rows` and the
      23 `layer.<i>` either at the reference's `layer.rows` (then OURS' own layer.rows must equal it) or at every
      tower row (then those rows are taken). Each of rows and every layer: rel vs fp64; PASS iff ours <= the largest
      rel among torch's own forms present (--bar-forms, default sdpa, sdpa+red32, eager, eager+red32, sdpa-math;
      f32core and any other form in DIR are reported, never a bar). --rows-only judges the output rows alone (the
      verdict records it).

  v5  Pinned bits. FILE...: engine output files of the same fixtures under different variants (row block, query
      block, slice, run, GPU). Fixture and variant come from the file's safetensors metadata (`fixture`, `variant`)
      or else from its name, <fixture>.<variant>.safetensors with <fixture> ending in an image extension (.png,
      .jpg, .jpeg, .webp; without one, the first dot splits). sha256 of KEY's bytes (bf16 as int16, the oracle's
      rows_sha256) per file; PASS per fixture iff every file holds KEY nonempty, all have one shape, one dtype and one
      sha256, and at least 2 distinct variants are present (and every --variants name, when given). KEY `rows` (the
      default, the tower's output rows) must also be bf16 [T, 4608].

Runs on the CPU (torch threads --threads, default 16); v3 imports Transformers' Inkling2 code for the config, the
position taps and the position ids. Transformers (Apache-2.0) is imported at run time; this file copies none of it.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkling2_f64 import rel  # noqa: E402  (the normwise relative error every gate uses)

BF = torch.bfloat16
TORCH_FORMS = ["sdpa", "sdpa+red32", "eager", "eager+red32", "sdpa-math"]
TOWER_WIDTH = 4608  # the tower's output rows: bf16 [T, 4608]
IMAGE_EXT = re.compile(r"^(?P<fixture>.+?\.(?:png|jpe?g|webp))\.(?P<variant>.+)$", re.IGNORECASE)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    """sha256 of the tensor's bytes (bf16 as its int16 bits): the oracle's rows_sha256."""
    t = t.detach().contiguous().cpu()
    if t.dtype == BF:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


BITS = {BF: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(BITS[t.dtype])


def share_equal(a: torch.Tensor, b: torch.Tensor) -> float:
    """Share of elements with the same bits (0 for different dtypes or shapes)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return 0.0
    if not a.numel():
        return 1.0
    if a.is_floating_point():
        return float((bits(a) == bits(b)).double().mean())
    return float((a == b).double().mean())


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0


def finish(report: dict, a) -> int:
    report["tool_sha256"] = file_sha(os.path.abspath(__file__))
    report["torch"] = torch.__version__
    report["argv"] = sys.argv[1:]
    report["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = json.dumps(report, indent=1)
    if getattr(a, "json", ""):
        with open(a.json, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0 if report["verdict"] == "PASS" else 1


def overall(verdicts: list) -> str:
    if not verdicts:
        return "EMPTY"
    return "PASS" if all(v == "PASS" for v in verdicts) else "FAIL"


# ----------------------------------------------------------------------------------------------------- v2


def cmd_v2(a) -> int:
    ft, fo = safe_open(a.tables, framework="pt"), safe_open(a.ours, framework="pt")
    tkeys, okeys = sorted(ft.keys()), set(fo.keys())
    items = []
    for k in tkeys:
        want = ft.get_tensor(k)
        it = {"key": k, "shape": list(want.shape), "dtype": str(want.dtype).replace("torch.", ""),
              "elements": want.numel()}
        if k not in okeys:
            items.append(it | {"verdict": "FAIL", "note": "missing in OURS"})
            continue
        got = fo.get_tensor(k)
        it["ours_dtype"] = str(got.dtype).replace("torch.", "")
        if tuple(got.shape) != tuple(want.shape):
            items.append(it | {"verdict": "FAIL", "note": f"shape {list(got.shape)}"})
            continue
        if want.is_floating_point():
            if got.dtype != want.dtype:
                items.append(it | {"verdict": "FAIL", "note": "dtype differs"})
                continue
            same = bits(got) == bits(want)
            both_nan = torch.isnan(got) & torch.isnan(want)
            nan_payload = int((both_nan & ~same).sum())
            ok = same if a.strict_nan else (same | both_nan)
            it |= {"equal": int(ok.sum()), "nan_payload_differs": nan_payload,
                   "max_abs_finite": max_abs(got[(fin := torch.isfinite(want) & torch.isfinite(got))], want[fin])}
        else:
            if got.is_floating_point():
                items.append(it | {"verdict": "FAIL", "note": "float against an integer table"})
                continue
            ok = got.to(torch.int64) == want.to(torch.int64)
            it |= {"equal": int(ok.sum())}
        it["differ"] = it["elements"] - it["equal"]
        it["verdict"] = "PASS" if it["elements"] > 0 and it["differ"] == 0 else ("EMPTY" if not it["elements"] else
                                                                                  "FAIL")
        if it["differ"]:
            bad = (~ok).reshape(-1).nonzero()[:5].reshape(-1).tolist()
            it["first_differ_at"] = bad
        items.append(it)
        log(f"v2 {k:22s} {it['verdict']} {it['equal']}/{it['elements']}")
    v = overall([i["verdict"] for i in items])
    rep = {"cmd": "v2", "verdict": v,
           "counts": {"tables": len(items), "tables_pass": sum(i["verdict"] == "PASS" for i in items),
                      "elements": sum(i["elements"] for i in items),
                      "elements_differ": sum(i.get("differ", i["elements"]) for i in items),
                      "ours_only_keys": sorted(okeys - set(tkeys))},
           "strict_nan": a.strict_nan, "inputs": {"tables": a.tables, "tables_sha256": file_sha(a.tables),
                                                  "ours": a.ours, "ours_sha256": file_sha(a.ours)},
           "items": items}
    return finish(rep, a)


# ----------------------------------------------------------------------------------------------------- v3


class Weights:
    """The checkpoint's model.visual.* tensors, bf16 -> float64 on the CPU, read on demand."""

    def __init__(self, folder: str):
        self.folder = folder
        idx = os.path.join(folder, "model.safetensors.index.json")
        self.map = json.load(open(idx))["weight_map"]
        self.files = {}
        self.dtypes = set()

    def __call__(self, name: str) -> torch.Tensor:
        full = "model.visual." + name
        shard = self.map[full]
        if shard not in self.files:
            self.files[shard] = safe_open(os.path.join(self.folder, shard), framework="pt")
        t = self.files[shard].get_tensor(full)
        self.dtypes.add(str(t.dtype))
        if t.dtype != BF:
            raise SystemExit(f"{full}: {t.dtype}, expected bf16")
        return t.double()


def ours_adapter_key(op: str) -> str:
    """Torch's key holding the value of ours' key `op` (for --ours-from-torch and torch's own output)."""
    m = re.match(r"^(L\d+)\.(res1|res2)\.out$", op)
    if m:
        return m.group(1) + (".L7.ln_2.in" if m.group(2) == "res1" else ".layer.out")
    return op


def f64_core(q, k, v, scale: float, log_every: float = 30.0):
    """softmax(q.k^T * scale) . v in float64 over all keys, in query blocks. q, k, v: [1, H, S, D]."""
    q, k, v = q.double(), k.double(), v.double()
    S = q.shape[2]
    blk = max(16, min(S, int(2 ** 30 // (8 * q.shape[1] * max(S, 1)))))
    out = torch.empty_like(q)
    kt = k.transpose(2, 3)
    t0, last = time.time(), time.time()
    for a in range(0, S, blk):
        s = torch.matmul(q[:, :, a:a + blk], kt) * scale
        out[:, :, a:a + blk] = torch.matmul(torch.softmax(s, dim=-1), v)
        if time.time() - last > log_every:
            last = time.time()
            log(f"  core {a + blk}/{S} rows, {time.time() - t0:.0f} s")
    return out


def cmd_v3(a) -> int:
    from transformers.models.inkling2 import modeling_inkling2 as M
    from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig

    torch.set_num_threads(a.threads)
    raw = json.load(open(os.path.join(a.model, "config.json")))["vision_config"]
    cfg = Inkling2VisionConfig(**raw)
    eps = float(cfg.layer_norm_eps)
    theta = float(cfg.rope_parameters["rope_theta"])
    heads, hidden = cfg.num_attention_heads, cfg.hidden_size
    hd = hidden // heads
    merge = cfg.spatial_merge_size
    if cfg.hidden_act != "gelu":
        raise SystemExit(f"hidden_act {cfg.hidden_act}: this check computes the erf GELU")
    W = Weights(a.model)
    ft = safe_open(a.torch, framework="pt")
    fo = safe_open(a.ours, framework="pt")
    tkeys, okeys = set(ft.keys()), set(fo.keys())
    T_ = ft.get_tensor  # torch's recorded tensors

    # grid: --grid, the sibling rows file, or ours' grid
    grid, grid_from = None, None
    if a.grid:
        grid, grid_from = [int(x) for x in a.grid.split(",")], "--grid"
    else:
        sib = re.sub(r"\.ops\.safetensors$", ".safetensors", a.torch)
        if sib != a.torch and os.path.exists(sib):
            g = safe_open(sib, framework="pt").get_tensor("grid").reshape(-1).tolist()
            grid, grid_from = g[-2:], os.path.basename(sib)
        elif "grid" in okeys:
            grid, grid_from = fo.get_tensor("grid").reshape(-1).tolist()[-2:], "OURS grid"
    if grid is None:
        raise SystemExit("no grid: pass --grid R,C")
    rows_, cols_ = grid
    P = 4 * rows_ * cols_
    S = P + 1
    if T_("E2.patch.in").shape[0] != P or T_("E5.ln_pre.in").shape[0] != S:
        raise SystemExit(f"grid {rows_}x{cols_} (P {P}) does not match the torch file "
                         f"(E2.patch.in {list(T_('E2.patch.in').shape)}, one image only)")
    layers = sorted(int(m.group(1)) for k in tkeys if (m := re.match(r"^L(\d+)\.layer\.in$", k)))
    thw = torch.tensor([[1, rows_ * merge, cols_ * merge]])
    idx, wts = M.get_vision_interpolation_indices_and_weights(thw, num_grid_per_side=cfg.posemb_grid_size,
                                                              mode="bilinear", align_corners=False,
                                                              spatial_merge_size=merge, kwargs={})
    pid = M.get_vision_position_ids(thw, merge, kwargs={}).to(torch.int64)
    pos = torch.zeros(S, 2, dtype=torch.int64)
    pos[1:] = pid + 1
    inv64 = 1.0 / (theta ** (torch.arange(0, hd // 2, 2, dtype=torch.float64) / (hd // 2)))
    fr = pos[..., None].double() * inv64  # [S, 2, 16]: column 0 = row, 1 = col

    def recompose(x):
        return torch.cat([x[:, 1], x[:, 0]], dim=-1).repeat_interleave(2, dim=-1)  # column angles first

    cos64, sin64 = recompose(fr.cos()), recompose(fr.sin())
    table_check = {}
    for i in layers:
        if f"L{i}.rot.cos" in tkeys:
            table_check[f"L{i}"] = {"cos_max_abs_vs_torch_fp32": max_abs(cos64, T_(f"L{i}.rot.cos")),
                                    "sin_max_abs_vs_torch_fp32": max_abs(sin64, T_(f"L{i}.rot.sin"))}
            if max(table_check[f"L{i}"].values()) > 1e-4:
                raise SystemExit(f"fp64 rotary tables disagree with torch's recorded ones: {table_check}")

    def rot64(x):
        x = x.double()
        x2 = x.view(*x.shape[:-1], -1, 2)
        r = torch.stack((-x2[..., 1], x2[..., 0]), dim=-1).flatten(start_dim=-2)
        return x * cos64[:, None, :] + r * sin64[:, None, :]

    def lin(x, w, b=None):
        return F.linear(x.double(), w, b)

    def ln(x, p):
        return F.layer_norm(x.double(), (hidden,), W(p + ".weight"), W(p + ".bias"), eps)

    def bb(n):
        return "backbone." + n

    # (ours' key, fp64 from torch's recorded inputs)
    plan = [("E2.patch.out", lambda: lin(T_("E2.patch.in"), W(bb("embeddings.patch_embedding.weight"))))]

    def e4():
        table = W(bb("embeddings.position_embedding"))
        posum = (table[idx] * wts.double()[:, :, None]).sum(1)
        out = torch.empty(S, hidden, dtype=torch.float64)
        out[0] = W(bb("embeddings.class_embedding")) + W(bb("embeddings.cls_position_embedding"))
        out[1:] = T_("E2.patch.out").double() + posum
        return out

    plan += [("E4.embeddings.out", e4), ("E5.ln_pre.out", lambda: ln(T_("E5.ln_pre.in"), bb("ln_pre")))]
    for i in layers:
        L, p = f"L{i}.", bb(f"layers.{i}.")
        plan.append((L + "L1.ln_1.out", lambda L=L, p=p: ln(T_(L + "L1.ln_1.in"), p + "ln_1")))
        for n in ("q", "k", "v"):
            plan.append((L + f"L2.{n}_proj.out", lambda L=L, p=p, n=n: lin(
                T_(L + f"L2.{n}_proj.in"), W(p + f"self_attn.{n}_proj.weight"), W(p + f"self_attn.{n}_proj.bias"))))
        for n in ("q", "k"):
            plan.append((L + f"rot.{n}_out", lambda L=L, n=n: rot64(T_(L + f"rot.{n}_in"))))
        plan.append((L + "core.out", lambda L=L: f64_core(T_(L + "core.q"), T_(L + "core.k"), T_(L + "core.v"),
                                                           hd ** -0.5)))
        plan.append((L + "L2.out_proj.out", lambda L=L, p=p: lin(
            T_(L + "L2.out_proj.in"), W(p + "self_attn.out_proj.weight"), W(p + "self_attn.out_proj.bias"))))
        plan.append((L + "res1.out", lambda L=L, p=p: T_(L + "layer.in").double()
                     + W(p + "ls_1.gamma") * T_(L + "L2.out_proj.out").double()))
        plan.append((L + "L7.ln_2.out", lambda L=L, p=p: ln(T_(L + "L7.ln_2.in"), p + "ln_2")))
        plan.append((L + "L8.gelu.out", lambda L=L, p=p: F.gelu(lin(
            T_(L + "L8.fc1.in"), W(p + "mlp.c_fc.weight"), W(p + "mlp.c_fc.bias")))))
        plan.append((L + "L8.fc2.out", lambda L=L, p=p: lin(
            T_(L + "L8.fc2.in"), W(p + "mlp.c_proj.weight"), W(p + "mlp.c_proj.bias"))))
        plan.append((L + "res2.out", lambda L=L, p=p: T_(L + "L7.ln_2.in").double()
                     + W(p + "ls_2.gamma") * T_(L + "L8.fc2.out").double()))
    plan += [("M2.linear_1.out", lambda: lin(T_("M2.linear_1.in"), W("proj_in.weight"))),
             ("M2.gelu.out", lambda: F.gelu(T_("M2.gelu.in").double())),
             ("M2.linear_2.out", lambda: lin(T_("M2.linear_2.in"), W("proj_out.weight")))]

    items = []
    for op, f64 in plan:
        tk = ours_adapter_key(op)
        ok_key = tk if a.ours_from_torch else op
        it = {"op": op, "torch_key": tk}
        t0 = time.time()
        if tk not in tkeys:
            items.append(it | {"verdict": "FAIL", "note": "missing in the torch file"})
            continue
        tv = T_(tk)
        it |= {"shape": list(tv.shape), "rows": int(tv.shape[2] if ".core." in op else tv.shape[0]),
               "elements": tv.numel()}
        with torch.no_grad():
            ref = f64().reshape(tv.shape)
        rb = ref.to(tv.dtype)
        it |= {"rel_torch": rel(tv, ref), "max_abs_torch": max_abs(tv, ref),
               "torch_eq_bf16_fp64": share_equal(tv, rb), "rel_floor_bf16_fp64": rel(rb, ref)}
        it["torch_over_floor"] = it["rel_torch"] / it["rel_floor_bf16_fp64"] if it["rel_floor_bf16_fp64"] else None
        if ok_key not in okeys:
            items.append(it | {"verdict": "FAIL", "note": f"{ok_key} missing in OURS"})
            log(f"v3 {op:22s} MISSING in OURS")
            continue
        ov = fo.get_tensor(ok_key)
        if ov.numel() != tv.numel():
            items.append(it | {"verdict": "FAIL", "note": f"OURS shape {list(ov.shape)}"})
            continue
        if list(ov.shape) != list(tv.shape):
            it["note"] = f"OURS shape {list(ov.shape)} read as {list(tv.shape)}"
            ov = ov.reshape(tv.shape)
        if ov.dtype != tv.dtype:
            items.append(it | {"verdict": "FAIL", "note": f"OURS dtype {ov.dtype}, torch's {tv.dtype}"})
            continue
        it |= {"rel_ours": rel(ov, ref), "max_abs_ours": max_abs(ov, ref), "bit_equal_ours_torch": share_equal(ov, tv),
               "ours_eq_bf16_fp64": share_equal(ov, rb)}
        it["ratio"] = it["rel_ours"] / it["rel_torch"] if it["rel_torch"] > 0 else (
            0.0 if it["rel_ours"] == 0 else math.inf)
        it["verdict_strict"] = "PASS" if it["rel_ours"] <= it["rel_torch"] else "FAIL"
        it["verdict"] = "PASS" if it["rel_ours"] <= it["rel_torch"] * (1 + a.tie_rel) else "FAIL"
        it["seconds"] = round(time.time() - t0, 2)
        items.append(it)
        log(f"v3 {op:22s} torch {it['rel_torch']:.3e} ours {it['rel_ours']:.3e} ratio {it['ratio']:.3f} "
            f"eq(o,t) {it['bit_equal_ours_torch']:.4f} {it['verdict']} ({it['seconds']} s)")
        del ref, rb, tv, ov
    v = overall([i["verdict"] for i in items])
    rep = {"cmd": "v3", "verdict": v,
           "counts": {"ops": len(items), "ops_pass": sum(i["verdict"] == "PASS" for i in items),
                      "ops_fail": sum(i["verdict"] != "PASS" for i in items), "layers": layers,
                      "tower_rows": S, "tokens": rows_ * cols_, "fixtures": 1},
           "tie_rel": a.tie_rel, "ops_fail_strict": sum(i.get("verdict_strict", "FAIL") != "PASS" for i in items),
           "grid": [rows_, cols_], "grid_from": grid_from, "ours_from_torch": a.ours_from_torch,
           "fp64_rotary": "inv_freq = 1/theta^(2j/32) and angles = pos * inv_freq in float64",
           "rotary_table_check": table_check, "weights_dtypes": sorted(W.dtypes),
           "inputs": {"model": a.model, "torch": a.torch, "torch_sha256": file_sha(a.torch), "ours": a.ours,
                      "ours_sha256": file_sha(a.ours)},
           "items": items}
    return finish(rep, a)


# ----------------------------------------------------------------------------------------------------- v4


def dist(got: torch.Tensor, ref: torch.Tensor) -> dict:
    return {"rel": rel(got, ref), "max_abs": max_abs(got, ref),
            "bit_equal_to_rounded_ref": share_equal(got, ref.to(got.dtype)) if got.dtype != ref.dtype else 1.0}


def form_path(d: str, name: str, form: str):
    for n in (name, name + ".png"):
        p = os.path.join(d, f"{n}.{form}.safetensors")
        if os.path.exists(p):
            return p
    return None


def load_tower(path: str) -> dict:
    f = safe_open(path, framework="pt")
    return {k: f.get_tensor(k) for k in f.keys()}


def cmd_v4(a) -> int:
    torch.set_num_threads(a.threads)
    ref_path = form_path(a.forms_dir, a.fixture, "fp64")
    if not ref_path:
        raise SystemExit(f"no {a.fixture}.fp64.safetensors in {a.forms_dir}")
    ref = load_tower(ref_path)
    name = os.path.basename(ref_path)[:-len(".fp64.safetensors")]
    lkeys = sorted((k for k in ref if re.match(r"^layer\.\d+$", k)), key=lambda k: int(k.split(".")[1]))
    sel = ref.get("layer.rows")
    bar_forms = a.bar_forms.split(",")
    others = sorted(re.sub(r"\.safetensors$", "", os.path.basename(p)[len(name) + 1:])
                    for p in os.listdir(a.forms_dir) if p.startswith(name + ".") and p.endswith(".safetensors")
                    and not p.endswith(".ops.safetensors") and ".packed." not in p)
    others = [f for f in others if f not in bar_forms and f != "fp64"]
    forms = {}
    for f in bar_forms + others:
        p = form_path(a.forms_dir, name, f)
        if not p:
            continue
        g = load_tower(p)
        d = {"bar": f in bar_forms, "rows": dist(g["rows"], ref["rows"]), "rows_sha256": tensor_sha(g["rows"]),
             "layers": {k: rel(g[k], ref[k]) for k in lkeys if k in g}}
        forms[f] = d
        log(f"v4 {f:14s} rows rel {d['rows']['rel']:.4e}{'' if d['bar'] else '  (not a bar)'}")
    ours = load_tower(a.ours)
    items = []
    bars = [f for f in forms if forms[f]["bar"]]
    # rows
    it = {"item": "rows", "rows": int(ref["rows"].shape[0]), "bar_forms": bars}
    if "rows" not in ours or ours["rows"].shape != ref["rows"].shape:
        it |= {"verdict": "FAIL", "note": f"OURS rows {list(ours['rows'].shape) if 'rows' in ours else None}"}
    elif not bars:
        it |= {"verdict": "EMPTY", "note": "no torch form present"}
    else:
        bar = max(forms[f]["rows"]["rel"] for f in bars)
        d = dist(ours["rows"], ref["rows"])
        it |= {"rel_ours": d["rel"], "max_abs_ours": d["max_abs"], "ours_eq_bf16_fp64": d["bit_equal_to_rounded_ref"],
               "bar": bar, "bar_form": max(bars, key=lambda f: forms[f]["rows"]["rel"]),
               "rel_sdpa": forms.get("sdpa", {}).get("rows", {}).get("rel"),
               "verdict": "PASS" if d["rel"] <= bar else "FAIL"}
    if "grid" in ours and "grid" in ref and ours["grid"].reshape(-1).tolist() != ref["grid"].reshape(-1).tolist():
        it |= {"verdict": "FAIL", "note": f"grid {ours['grid'].tolist()} vs {ref['grid'].tolist()}"}
    items.append(it)
    # layers
    S = None
    if "rows" in ref:
        S = 4 * int(ref["rows"].shape[0]) + 1
    ours_rows_ok = "layer.rows" not in ours or (sel is not None and torch.equal(ours["layer.rows"], sel))
    for k in lkeys:
        it = {"item": k, "rows": int(ref[k].shape[0])}
        lb = [f for f in bars if k in forms[f]["layers"]]
        if k not in ours:
            if a.rows_only:
                continue
            items.append(it | {"verdict": "FAIL", "note": "missing in OURS"})
            continue
        o = ours[k]
        if "layer.rows" in ours:
            if not ours_rows_ok:
                items.append(it | {"verdict": "FAIL", "note": "OURS layer.rows differ from the reference's"})
                continue
        elif S is not None and sel is not None and o.shape[0] == S:
            o = o[sel]
        if o.shape != ref[k].shape:
            items.append(it | {"verdict": "FAIL", "note": f"OURS {k} shape {list(o.shape)}"})
            continue
        if not lb:
            items.append(it | {"verdict": "EMPTY", "note": "no torch form has this layer"})
            continue
        bar = max(forms[f]["layers"][k] for f in lb)
        r = rel(o, ref[k])
        items.append(it | {"rel_ours": r, "bar": bar, "bar_form": max(lb, key=lambda f: forms[f]["layers"][k]),
                           "rel_sdpa": forms.get("sdpa", {}).get("layers", {}).get(k),
                           "verdict": "PASS" if r <= bar else "FAIL"})
    for i in items:
        if "rel_ours" in i:
            log(f"v4 ours {i['item']:9s} rel {i['rel_ours']:.4e} bar {i['bar']:.4e} ({i['bar_form']}) {i['verdict']}")
    v = overall([i["verdict"] for i in items])
    nl = sum(1 for i in items if i["item"] != "rows")
    rep = {"cmd": "v4", "verdict": v, "fixture": name,
           "counts": {"fixtures": 1, "items": len(items), "items_pass": sum(i["verdict"] == "PASS" for i in items),
                      "output_rows": int(ref["rows"].shape[0]), "layers": nl,
                      "layers_in_reference": len(lkeys), "sampled_layer_rows": int(sel.numel()) if sel is not None
                      else 0, "bar_forms": len(bars)},
           "rows_only": a.rows_only, "forms": forms,
           "inputs": {"forms_dir": a.forms_dir, "reference": os.path.basename(ref_path),
                      "reference_sha256": file_sha(ref_path), "ours": a.ours, "ours_sha256": file_sha(a.ours),
                      "ours_rows_sha256": tensor_sha(ours["rows"]) if "rows" in ours else None},
           "items": items}
    return finish(rep, a)


# ----------------------------------------------------------------------------------------------------- v5


def label(path: str):
    f = safe_open(path, framework="pt")
    md = f.metadata() or {}
    if md.get("fixture") and md.get("variant"):
        return md["fixture"], md["variant"], "metadata"
    base = os.path.basename(path)
    base = base[:-len(".safetensors")] if base.endswith(".safetensors") else base
    m = IMAGE_EXT.match(base)
    if m:
        return m.group("fixture"), m.group("variant"), "name"
    if "." in base:
        fx, var = base.split(".", 1)
        return fx, var, "name"
    return base, "", "name"


def v5_problems(t: torch.Tensor, key: str) -> list:
    """Why one file's tensor cannot pin bits: empty, or (KEY `rows`, the tower's output) not bf16 [T, 4608]."""
    out = []
    if t.numel() == 0:
        out.append(f"empty ({list(t.shape)})")
    if key == "rows":
        if t.dtype != BF:
            out.append(f"dtype {str(t.dtype).replace('torch.', '')}, the tower's rows are bfloat16")
        if t.dim() != 2 or t.shape[-1] != TOWER_WIDTH:
            out.append(f"shape {list(t.shape)}, the tower's rows are [T, {TOWER_WIDTH}]")
    return out


def cmd_v5(a) -> int:
    groups = {}
    for p in a.files:
        fx, var, how = label(p)
        f = safe_open(p, framework="pt")
        if a.key not in f.keys():
            groups.setdefault(fx, []).append({"file": p, "variant": var, "labelled_by": how, "sha256": None,
                                              "note": f"no {a.key}", "problems": [f"no {a.key}"]})
            continue
        t = f.get_tensor(a.key)
        groups.setdefault(fx, []).append({"file": p, "variant": var, "labelled_by": how, "sha256": tensor_sha(t),
                                          "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                                          "numel": t.numel(), "problems": v5_problems(t, a.key)})
    need = [x for x in a.variants.split(",") if x] if a.variants else []
    items = []
    for fx, fs in sorted(groups.items()):
        shas = {}
        for x in fs:
            shas.setdefault(x["sha256"], []).append(x["variant"])
        variants = sorted({x["variant"] for x in fs})
        missing = [x for x in need if x not in variants]
        shapes = sorted({tuple(x["shape"]) for x in fs if "shape" in x})
        dtypes = sorted({x["dtype"] for x in fs if "dtype" in x})
        problems = [f"{x['variant']}: {q}" for x in fs for q in x["problems"]]
        if len(shapes) > 1:
            problems.append(f"shapes differ: {[list(s) for s in shapes]}")
        if len(dtypes) > 1:
            problems.append(f"dtypes differ: {dtypes}")
        ok = (len(shas) == 1 and None not in shas and len(variants) >= 2 and not missing and not problems)
        it = {"fixture": fx, "files": len(fs), "variants": variants, "distinct_sha256": len(shas),
              "groups": [{"sha256": s, "variants": v} for s, v in shas.items()], "missing_variants": missing,
              "shapes": [list(s) for s in shapes], "dtypes": dtypes, "problems": problems,
              "verdict": "PASS" if ok else "FAIL"}
        if len(variants) < 2:
            it["note"] = "fewer than 2 variants"
        items.append(it)
        log(f"v5 {fx:28s} {len(fs)} files, {len(variants)} variants, {len(shas)} distinct sha256: {it['verdict']}"
            + (f" ({'; '.join(problems)})" if problems else ""))
    v = overall([i["verdict"] for i in items])
    rep = {"cmd": "v5", "verdict": v, "key": a.key,
           "counts": {"fixtures": len(items), "fixtures_pass": sum(i["verdict"] == "PASS" for i in items),
                      "files": len(a.files)},
           "items": items, "files": [x for fs in groups.values() for x in fs]}
    return finish(rep, a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s2 = sub.add_parser("v2")
    s2.add_argument("tables")
    s2.add_argument("ours")
    s2.add_argument("--strict-nan", action="store_true")
    s3 = sub.add_parser("v3")
    s3.add_argument("--model", required=True)
    s3.add_argument("--torch", required=True)
    s3.add_argument("--ours", required=True)
    s3.add_argument("--ours-from-torch", action="store_true")
    s3.add_argument("--grid", default="")
    s3.add_argument("--tie-rel", type=float, default=0.0,
                    help="PASS iff rel(ours) <= rel(torch) * (1 + TIE); default 0, the rule as written")
    s4 = sub.add_parser("v4")
    s4.add_argument("--forms-dir", required=True)
    s4.add_argument("--fixture", required=True)
    s4.add_argument("--ours", required=True)
    s4.add_argument("--bar-forms", default=",".join(TORCH_FORMS))
    s4.add_argument("--rows-only", action="store_true")
    s5 = sub.add_parser("v5")
    s5.add_argument("files", nargs="+")
    s5.add_argument("--key", default="rows")
    s5.add_argument("--variants", default="")
    for s in (s2, s3, s4, s5):
        s.add_argument("--json", default="")
        s.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    return {"v2": cmd_v2, "v3": cmd_v3, "v4": cmd_v4, "v5": cmd_v5}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
