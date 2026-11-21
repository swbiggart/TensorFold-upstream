"""Inkling-2-Small's vision tower and image prompts through Transformers' own Inkling2 modules: the image oracle.

The tower (`model.visual.*`, 422 bf16 tensors, 0.61 GiB, all in the last shard) is small, so it is built whole on one
GPU as Transformers' `Inkling2VisionModel` with the checkpoint's tensors assigned under Transformers' own renames
(conversion_mapping.py, the `model.visual.` rules). The processor is Transformers' `Inkling2Processor` /
`Inkling2ImageProcessor` (torchvision backend). Image prompts reuse the text oracle (`inkling2_ref.py`, imported
unchanged): the embed rows at <|unused_200054|> (200054) are replaced, in order, by the tower's rows, exactly as
`Inkling2Model.forward` does (`masked_scatter` after the normed embedding), and only in the pass that was given the
image (a 200054 the model generates later is plain text).

USAGE = python tools/zig/inkling2_vision_ref.py CMD MODEL_DIR --out OUT_DIR [options]

  fixtures OUT                      the image set (deterministic PNG, JPEG, WebP; sizes 1x1 .. 4096x4096) + SHA256SUMS
  process  --fixtures DIR           Transformers' processor on each fixture by both routes (the opened upright PIL
                                    image: alpha onto the checkerboard; the base64 string: alpha dropped): RGB bytes,
                                    fp32 pixel values, grid; prompt ids for one user message per image
  tower    --fixtures DIR --images a,b --forms sdpa,eager,...   the tower's output rows by form; --ops dumps every
                                    op's inputs and outputs (layers --op-layers), --packed runs the images as one call
  tables                            rotary cos/sin, GELU over all bf16 inputs, position taps for a list of grids
  greedy   --fixtures DIR --set NAME --decode N   image prompts through the text oracle: greedy ids, margins
  probe    --fixtures DIR --set NAME             the largest prompts: the tower and the text oracle's prefill, with
                                                 each one's peak GPU memory (does one GPU hold it?)

Tower forms (--forms; the manifest names what ran):
  sdpa         Transformers' default for the tower (its `_supports_sdpa`): torch's scaled_dot_product_attention, the
               kernel torch picks; `sdpa-flash`, `sdpa-efficient`, `sdpa-cudnn`, `sdpa-math` force one kernel
  eager        Transformers' eager core (bf16 q.k, fp32 softmax rounded to bf16, bf16 P.V); refused when the scores
               of one image would not fit (16 S^2 x 8 bytes over --eager-max-gib)
  f32core      fp32 q.k, scale, softmax and P.V in query blocks, one bf16 rounding of each head's output (the
               native core's proposed rounding points), the rest as `sdpa`
  fp64         the whole module in float64 from the bf16 weights and the bf16-cast pixels; rotary angles, cos and sin
               in float64; attention in float64 query blocks; the position taps' indices and fp32 weights as
               Transformers computes them (they are the definition)
  Suffix `+red32` (e.g. `sdpa+red32`): bf16 matmuls reduced in fp32 (torch's
  allow_bf16_reduced_precision_reduction off); without it, torch's default (on).

Every run sets TF32 off and probes it (inkling2_ref.tf32_guard). Transformers must contain models/inkling2
(main from 109a7d19); its image processor needs torchvision (and Pillow for PIL inputs). On the oracle host the two
live in a --target folder on PYTHONPATH, not in the venv.

Output (OUT_DIR/<cmd>/...): safetensors per image and form, manifest.json with versions, the attention kernel that
ran, GPU, settings, sha256 of every input and output, wall times and peak memory.

Adapted parts keep their notices. Transformers (Apache-2.0) is imported at run time; the tensor renames below
(`VISUAL_RENAMES`) follow its conversion_mapping.py (Inkling2, `model.visual.` rules) at 109a7d19, Copyright 2026 the
HuggingFace Team, Apache License 2.0.
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling2_ref as R  # noqa: E402  (the text oracle, unchanged)

BF = torch.bfloat16
IMAGE_TOKEN = 200054
TOKEN_PX = 40

# Transformers' conversion_mapping.py (Inkling2, `model.visual.` rules, 109a7d19), applied in order.
VISUAL_RENAMES = [
    (r"^model\.visual\.backbone\.embeddings\.position_embedding$", "embeddings.position_embedding.weight"),
    (r"^model\.visual\.backbone\.embeddings\.", "embeddings."),
    (r"^model\.visual\.backbone\.ln_pre\.", "pre_layernorm."),
    (r"^model\.visual\.backbone\.layers\.", "layers."),
    (r"^model\.visual\.proj_in\.", "merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "merger.linear_2."),
    (r"^(layers\.\d+)\.ln_1\.", r"\1.layernorm_before."),
    (r"^(layers\.\d+)\.ln_2\.", r"\1.layernorm_after."),
    (r"^(layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1."),
    (r"^(layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2."),
    (r"^(layers\.\d+)\.ls_1\.gamma$", r"\1.lambda_1"),
    (r"^(layers\.\d+)\.ls_2\.gamma$", r"\1.lambda_2"),
]


def hf_visual(name: str) -> str:
    out = name
    for pat, rep in VISUAL_RENAMES:
        out = re.sub(pat, rep, out)
    return out


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    if t.dtype == BF:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------- fixtures

SIZES = [(1, 1), (39, 41), (40, 40), (448, 336), (1024, 768), (1024, 1024), (1920, 1080), (2048, 2048),
         (2560, 1440), (3840, 2160), (4096, 4096)]  # (width, height)


def _scene(w: int, h: int, seed: int, text: str = ""):
    """A deterministic picture with structure the model can describe: a sky-to-ground gradient, a sun, a house,
    a few coloured shapes, and optionally a sign with `text`. Everything is integer geometry; noise is seeded."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    g = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]
    sky = np.stack([90 + 80 * y + 0 * x, 150 + 60 * y + 0 * x, 235 - 40 * y + 0 * x], -1)
    ground = np.stack([70 + 40 * x + 0 * y, 140 - 30 * y + 0 * x, 60 + 0 * x + 0 * y], -1)
    img = np.where(y[..., None] < 0.62, sky, ground)
    img = img + g.normal(0, 3.0, img.shape)
    im = Image.fromarray(np.clip(np.rint(img), 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    s = min(w, h)
    if s >= 40:
        r = max(2, s // 12)
        d.ellipse([w - 3 * r, r, w - r, 3 * r], fill=(250, 210, 40))
        hx, hy, hw = w // 5, int(h * 0.62), max(4, w // 5)
        d.rectangle([hx, hy - hw * 3 // 4, hx + hw, hy], fill=(180, 60, 50))
        d.polygon([(hx - hw // 8, hy - hw * 3 // 4), (hx + hw // 2, hy - hw * 5 // 4),
                   (hx + hw + hw // 8, hy - hw * 3 // 4)], fill=(90, 50, 40))
        d.rectangle([hx + hw * 2 // 5, hy - hw * 2 // 5, hx + hw * 3 // 5, hy], fill=(60, 40, 30))
        for k in range(int(g.integers(2, 5))):
            cx, cy, rr = int(g.integers(w // 2, w)), int(g.integers(h * 2 // 3, h)), max(2, s // 20)
            col = [(30, 90, 200), (220, 120, 30), (240, 240, 240), (120, 30, 160)][k % 4]
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col)
        if text:
            fs = max(10, s // 10)
            try:
                font = ImageFont.load_default(size=fs)
            except TypeError:
                font = ImageFont.load_default()
            bx, by = w // 2 - fs * len(text) // 3, h // 3
            d.rectangle([bx - fs // 3, by - fs // 3, bx + fs * len(text) * 2 // 3 + fs // 3, by + fs + fs // 3],
                        fill=(250, 250, 245), outline=(20, 20, 20), width=max(1, fs // 12))
            d.text((bx, by), text, fill=(200, 20, 20), font=font)
    return im


def _chart(w: int, h: int, seed: int):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    g = np.random.default_rng(seed)
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    vals = g.integers(10, 100, 5)
    fs = max(10, min(w, h) // 20)
    try:
        font = ImageFont.load_default(size=fs)
    except TypeError:
        font = ImageFont.load_default()
    bw = w // 8
    for i, v in enumerate(vals):
        x0 = w // 10 + i * (bw + w // 40)
        y0 = int(h * 0.85) - int(v / 100 * h * 0.7)
        d.rectangle([x0, y0, x0 + bw, int(h * 0.85)], fill=[(66, 133, 244), (219, 68, 55), (244, 180, 0),
                                                             (15, 157, 88), (171, 71, 188)][i])
        d.text((x0, int(h * 0.87)), "ABCDE"[i], fill=(0, 0, 0), font=font)
        d.text((x0, y0 - fs - 2), str(int(v)), fill=(0, 0, 0), font=font)
    d.line([w // 12, int(h * 0.85), w - w // 20, int(h * 0.85)], fill=(0, 0, 0), width=2)
    return im, [int(v) for v in vals]


def make_fixtures(out: str) -> dict:
    """The set: scenes and charts at every size in SIZES as PNG; formats and layouts at 448x336 (RGBA with partial
    alpha, palette with tRNS, gray, gray+alpha, 16-bit, interlaced PNG; baseline, progressive, gray JPEG and EXIF
    orientations 1-8; lossy, lossless and alpha WebP)."""
    import numpy as np
    from PIL import Image

    os.makedirs(out, exist_ok=True)
    made = {}

    def save(im, name, **kw):
        p = os.path.join(out, name)
        im.save(p, **kw)
        made[name] = {"w": im.size[0], "h": im.size[1], "mode": im.mode}

    words = ["STOP", "OPEN", "EXIT", "SALE", "CAFE", "PARK", "BANK", "TAXI", "HOTEL", "MUSEUM", "BAKERY"]
    for i, (w, h) in enumerate(SIZES):
        save(_scene(w, h, 1000 + i, words[i % len(words)]), f"scene_{w}x{h}.png", compress_level=6)
    for i, (w, h) in enumerate([(448, 336), (1024, 768), (1920, 1080)]):
        im, vals = _chart(w, h, 2000 + i)
        save(im, f"chart_{w}x{h}.png", compress_level=6)
        made[f"chart_{w}x{h}.png"]["values"] = vals
    base = _scene(448, 336, 3000, "CAFE")
    a = np.array(base.convert("RGBA"))
    yy, xx = np.mgrid[0:336, 0:448]
    a[..., 3] = np.clip((xx * 255) // 447, 0, 255).astype(np.uint8)
    a[:40, :40, 3] = 0
    save(Image.fromarray(a, "RGBA"), "rgba_448x336.png", compress_level=6)
    pal = base.convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
    pal.info["transparency"] = 0
    save(pal, "palette_trns_448x336.png", transparency=0)
    save(base.convert("L"), "gray_448x336.png")
    la = Image.merge("LA", (base.convert("L"), Image.fromarray(a[..., 3])))
    save(la, "gray_alpha_448x336.png")
    i16 = Image.fromarray(np.array(base.convert("L")).astype(np.uint16) * 257)
    save(i16, "gray16_448x336.png")
    save(base, "baseline_448x336.jpg", quality=90)
    save(base, "progressive_448x336.jpg", quality=90, progressive=True)
    save(base.convert("L"), "gray_448x336.jpg", quality=90)
    for o in range(1, 9):
        ex = Image.Exif()
        ex[0x0112] = o
        save(base, f"exif{o}_448x336.jpg", quality=90, exif=ex.tobytes())
    save(base, "lossy_448x336.webp", quality=80, method=4)
    save(base, "lossless_448x336.webp", lossless=True)
    save(Image.fromarray(a, "RGBA"), "alpha_448x336.webp", lossless=True)
    sums = {}
    for name in sorted(made):
        sums[name] = R.sha256(os.path.join(out, name))
    with open(os.path.join(out, "SHA256SUMS"), "w") as f:
        for name, h in sums.items():
            f.write(f"{h}  {name}\n")
    with open(os.path.join(out, "fixtures.json"), "w") as f:
        json.dump({"fixtures": made, "note": "interlaced PNG: Pillow writes no Adam7; V-b supplies one if needed"},
                  f, indent=1)
    return made


def open_upright(path: str):
    """The opened image turned upright (EXIF orientation applied), mode kept: the PIL route of the processor."""
    from PIL import Image, ImageOps

    im = Image.open(path)
    im.load()
    return ImageOps.exif_transpose(im)


def data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[ext]
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


# ----------------------------------------------------------------------------------------------- the tower


def vision_config(folder: str, attn: str):
    from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig

    raw = json.load(open(os.path.join(folder, "config.json")))["vision_config"]
    cfg = Inkling2VisionConfig(**raw)
    cfg._attn_implementation = attn
    return cfg


def build_tower(folder: str, attn: str, dev, dtype=BF):
    from transformers.models.inkling2 import modeling_inkling2 as M

    cfg = vision_config(folder, attn)
    src = R.Source(folder)
    names = src.names("model.visual.")
    if len(names) != 422:
        raise SystemExit(f"model.visual.*: {len(names)} tensors, expected 422")
    with torch.device("meta"):
        tower = M.Inkling2VisionModel(cfg)
    sd = {}
    for n in names:
        t = src.get(n)
        if t.dtype != BF:
            raise SystemExit(f"{n}: {t.dtype}, expected bf16")
        sd[hf_visual(n)] = t.to(dev).to(dtype)
    missing = set(k for k, _ in tower.named_parameters()) - set(sd)
    extra = set(sd) - set(k for k, _ in tower.named_parameters())
    if missing or extra:
        raise SystemExit(f"tower names: missing {sorted(missing)[:5]} extra {sorted(extra)[:5]}")
    tower.load_state_dict(sd, strict=False, assign=True)
    # buffers (inv_freq) were built on meta: rebuild them on the device
    inv, _ = M.Inkling2VisionRotaryEmbedding.compute_axial_rope_parameters(cfg)
    tower.rotary_emb.inv_freq = inv.to(dev)
    tower.rotary_emb.original_inv_freq = inv.clone().to(dev)
    for n, t in list(tower.named_parameters()) + list(tower.named_buffers()):
        if t.device.type == "meta":
            raise RuntimeError(f"{n} was not loaded")
    weights_sha = hashlib.sha256()
    for n in names:
        weights_sha.update(n.encode())
        weights_sha.update(tensor_sha(src.get(n)).encode())
    return tower.eval(), cfg, weights_sha.hexdigest()


# --- attention cores registered with Transformers' AttentionInterface

CAPTURE = {"on": False, "layer": None, "layers": set(), "ops": {}}
BLOCK = {"q": 1024}


def _cap(name: str, t: torch.Tensor):
    if CAPTURE["on"] and CAPTURE["layer"] in CAPTURE["layers"]:
        CAPTURE["ops"].setdefault(f"L{CAPTURE['layer']}.{name}", []).append(t.detach().cpu())


def blocked_core(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    """softmax(q.k * scale) . v without a mask over one image, in query blocks. q, k, v: [1, H, S, D].
    In float64 for an fp64 module; for bf16 inputs (form f32core): fp32 scores, softmax and P.V, one bf16 rounding."""
    if attention_mask is not None:
        raise RuntimeError("the tower's attention has no mask")
    work = torch.float64 if query.dtype == torch.float64 else torch.float32
    q, k, v = query.to(work), key.to(work), value.to(work)
    S = q.shape[2]
    out = torch.empty_like(q)
    for a in range(0, S, BLOCK["q"]):
        s = torch.matmul(q[:, :, a:a + BLOCK["q"]], k.transpose(2, 3)) * scaling
        p = torch.softmax(s, dim=-1)
        out[:, :, a:a + BLOCK["q"]] = torch.matmul(p, v)
    out = out.to(query.dtype)
    _cap("core.q", query)
    _cap("core.k", key)
    _cap("core.v", value)
    _cap("core.out", out)
    return out.transpose(1, 2).contiguous(), None


def make_capturing(inner):
    def core(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        o, w = inner(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
        _cap("core.q", query)
        _cap("core.k", key)
        _cap("core.v", value)
        _cap("core.out", o.transpose(1, 2))
        return o, w
    return core


def register_cores():
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.inkling2 import modeling_inkling2 as M

    AttentionInterface.register("tf_blocked", blocked_core)
    AttentionInterface.register("tf_sdpa", make_capturing(ALL_ATTENTION_FUNCTIONS["sdpa"]))
    AttentionInterface.register("tf_eager", make_capturing(M.eager_attention_forward))


def patch_rotary_fp64():
    """For the fp64 form: rotary angles, cos and sin in float64 and the rotation in float64 (Transformers casts to
    fp32 inside apply_rotary_pos_emb_vision and computes the angles in fp32)."""
    from transformers.models.inkling2 import modeling_inkling2 as M

    def fwd(self, x, position_ids):
        inv = 1.0 / (self.config.rope_parameters["rope_theta"] ** (
            torch.arange(0, 32, 2, dtype=torch.float64, device=x.device) / 32))
        freqs = position_ids[..., None].to(torch.float64) * inv
        return self.recomposition_frequencies(freqs.cos()), self.recomposition_frequencies(freqs.sin())

    def rot(q, k, cos, sin):
        cos, sin = cos.unsqueeze(-2).to(q.dtype), sin.unsqueeze(-2).to(q.dtype)
        return (q * cos) + (M.rotate_pairwise(q) * sin), (k * cos) + (M.rotate_pairwise(k) * sin)

    M.Inkling2VisionRotaryEmbedding.forward = torch.no_grad()(fwd)
    M.apply_rotary_pos_emb_vision = rot


def patch_rotary_capture():
    from transformers.models.inkling2 import modeling_inkling2 as M

    inner = M.apply_rotary_pos_emb_vision

    def rot(q, k, cos, sin):
        _cap("rot.q_in", q)
        _cap("rot.k_in", k)
        _cap("rot.cos", cos)
        _cap("rot.sin", sin)
        qe, ke = inner(q, k, cos, sin)
        _cap("rot.q_out", qe)
        _cap("rot.k_out", ke)
        return qe, ke

    M.apply_rotary_pos_emb_vision = rot


def hook_ops(tower, layers: set):
    """Forward hooks that record each op's input and output for the chosen layers, the embedding and the merger."""
    hs = []

    def rec(tag):
        def h(m, a, o):
            if CAPTURE["on"]:
                CAPTURE["ops"].setdefault(tag + ".in", []).append(a[0].detach().cpu())
                CAPTURE["ops"].setdefault(tag + ".out", []).append((o[0] if isinstance(o, tuple) else o).detach().cpu())
        return h

    e = tower.embeddings
    hs.append(e.patch_embedding.register_forward_hook(rec("E2.patch")))
    hs.append(e.register_forward_hook(lambda m, a, o: CAPTURE["ops"].setdefault("E4.embeddings.out", []).append(
        o.detach().cpu()) if CAPTURE["on"] else None))
    hs.append(tower.pre_layernorm.register_forward_hook(rec("E5.ln_pre")))
    for i, L in enumerate(tower.layers):
        def pre(m, a, i=i):
            CAPTURE["layer"] = i
        hs.append(L.register_forward_pre_hook(pre))
        if i not in layers:
            continue
        p = f"L{i}."
        hs.append(L.register_forward_hook(rec(p + "layer")))
        hs.append(L.layernorm_before.register_forward_hook(rec(p + "L1.ln_1")))
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            hs.append(getattr(L.self_attn, n).register_forward_hook(rec(p + "L2." + n)))
        hs.append(L.layernorm_after.register_forward_hook(rec(p + "L7.ln_2")))
        hs.append(L.mlp.fc1.register_forward_hook(rec(p + "L8.fc1")))
        hs.append(L.mlp.activation_fn.register_forward_hook(rec(p + "L8.gelu")))
        hs.append(L.mlp.fc2.register_forward_hook(rec(p + "L8.fc2")))
    m = tower.merger
    hs.append(m.linear_1.register_forward_hook(rec("M2.linear_1")))
    hs.append(m.act.register_forward_hook(rec("M2.gelu")))
    hs.append(m.linear_2.register_forward_hook(rec("M2.linear_2")))
    return hs


def sample_tower_rows(S: int) -> list[int]:
    rows = set(range(min(S, 33))) | set(range(max(0, S - 32), S)) | set(range(0, S, 64))
    return sorted(rows)


def run_tower(tower, pixel_values: torch.Tensor, grid: torch.Tensor, dtype=BF, layer_rows: bool = True):
    """pooler_output [T, 4608] and each layer's output at sampled rows (incl. the class row 0)."""
    feeds = {}
    hooks = []
    if layer_rows:
        S = int((grid.prod(-1) * 4 + 1).sum())
        keep = torch.tensor(sample_tower_rows(S))
        for i, L in enumerate(tower.layers):
            hooks.append(L.register_forward_hook(
                lambda m, a, o, i=i: feeds.__setitem__(f"layer.{i}", (o[0] if isinstance(o, tuple) else o)[
                    keep.to(o.device if not isinstance(o, tuple) else o[0].device)].detach().cpu())))
        feeds["layer.rows"] = keep
    pv = pixel_values.to(BF).to(dtype)  # the model's own cast (bf16), then the form's dtype
    try:
        with torch.no_grad():
            out = tower(pixel_values=pv, image_grid_thw=grid)
    finally:
        for h in hooks:
            h.remove()
    return out.pooler_output, feeds


SDPA_BACKENDS = {"flash": "FLASH_ATTENTION", "efficient": "EFFICIENT_ATTENTION", "cudnn": "CUDNN_ATTENTION",
                 "math": "MATH"}


def form_context(form: str):
    """(attn implementation name, dtype, a context manager factory or None)."""
    base = form.split("+")[0]
    if base == "fp64":
        return "tf_blocked", torch.float64, None
    if base == "f32core":
        return "tf_blocked", BF, None
    if base == "eager":
        return "tf_eager", BF, None
    if base == "sdpa":
        return "tf_sdpa", BF, None
    if base.startswith("sdpa-"):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        b = getattr(SDPBackend, SDPA_BACKENDS[base[5:]])
        return "tf_sdpa", BF, (lambda: sdpa_kernel([b]))
    raise SystemExit(f"unknown form {form}")


def distances(got: torch.Tensor, ref64: torch.Tensor) -> dict:
    """got (any dtype) against an fp64 reference: max and mean |diff|, relative norm, and the share bit-equal to
    the reference rounded to got's dtype."""
    g, r = got.double(), ref64.double()
    d = (g - r).abs()
    rr = ref64.to(got.dtype) if got.dtype != torch.float64 else ref64
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel_norm": float((g - r).norm() / max(float(r.norm()), 1e-300)),
            "bit_equal_to_rounded_ref": float((got == rr).float().mean())}


# ----------------------------------------------------------------------------------------------- commands


def cmd_process(a, dev):
    from transformers import AutoProcessor
    from transformers.image_utils import load_image  # noqa: F401  (the string route goes through it)

    proc = AutoProcessor.from_pretrained(a.model_dir)
    fx = json.load(open(os.path.join(a.fixtures, "fixtures.json")))["fixtures"]
    names = a.images.split(",") if a.images else sorted(fx)
    out = os.path.join(a.out, "process")
    os.makedirs(out, exist_ok=True)
    man = {"fixtures_sha256": {}, "routes": {}, "files_out_sha256": {}}
    for n in names:
        p = os.path.join(a.fixtures, n)
        man["fixtures_sha256"][n] = R.sha256(p)
        rec = {}
        for route, img in (("pil", open_upright(p)), ("string", data_url(p))):
            bf = proc.image_processor(images=[img], return_tensors="pt")
            pv, grid = bf["pixel_values"], bf["image_grid_thw"]
            rec[f"{route}.pixel_values"] = pv.float().contiguous()
            rec[f"{route}.grid"] = grid.to(torch.int64)
            man["routes"].setdefault(n, {})[route] = {"grid": grid[0].tolist(), "tokens": int(grid.prod()),
                                                      "pixel_values_sha256": tensor_sha(pv.float())}
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, reasoning_effort="none")
        enc = proc(text=[text], images=[open_upright(p)], return_tensors="pt")
        rec["prompt.ids"] = enc["input_ids"][0].to(torch.int64)
        man["routes"][n]["prompt_tokens"] = int(enc["input_ids"].shape[1])
        man["routes"][n]["prompt_ids_sha256"] = tensor_sha(enc["input_ids"][0].to(torch.int64))
        path = os.path.join(out, n + ".safetensors")
        save_file(rec, path)
        man["files_out_sha256"][os.path.basename(path)] = R.sha256(path)
        print(f"{n}: grid {man['routes'][n]['pil']['grid']}, tokens {man['routes'][n]['pil']['tokens']}, "
              f"pil==string {man['routes'][n]['pil']['pixel_values_sha256'] == man['routes'][n]['string']['pixel_values_sha256']}",
              flush=True)
    return out, man


def processed(proc, path: str):
    bf = proc.image_processor(images=[open_upright(path)], return_tensors="pt")
    return bf["pixel_values"], bf["image_grid_thw"]


def cmd_tower(a, dev):
    from transformers import AutoProcessor

    register_cores()
    forms = a.forms.split(",")
    if any(f.split("+")[0] == "fp64" for f in forms) and len(set(f.split("+")[0] == "fp64" for f in forms)) > 1:
        raise SystemExit("run fp64 in its own invocation (it patches the rotary code)")
    if forms[0].split("+")[0] == "fp64":
        patch_rotary_fp64()
    if a.ops:
        patch_rotary_capture()
    proc = AutoProcessor.from_pretrained(a.model_dir)
    names = a.images.split(",")
    out = os.path.join(a.out, "tower")
    os.makedirs(out, exist_ok=True)
    man_path = os.path.join(out, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {"runs": {}}
    inputs = []
    for n in names:
        pv, grid = processed(proc, os.path.join(a.fixtures, n))
        inputs.append((n, pv, grid))
    for form in forms:
        attn, dtype, ctxf = form_context(form)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = "+red32" not in form
        tower, cfg, wsha = build_tower(a.model_dir, attn, dev, dtype)
        hooks = hook_ops(tower, set(int(x) for x in a.op_layers.split(","))) if a.ops else []
        CAPTURE["layers"] = set(int(x) for x in a.op_layers.split(",")) if a.ops else set()
        groups = [inputs] if a.packed else [[x] for x in inputs]
        for grp in groups:
            pv = torch.cat([g[1] for g in grp]).to(dev)
            grid = torch.cat([g[2] for g in grp])
            S = int((grid.prod(-1) * 4 + 1).max())
            if form.split("+")[0] == "eager" and 16 * S * S * 8 / 2**30 > a.eager_max_gib:
                print(f"{form} {'+'.join(g[0] for g in grp)}: S {S} needs ~{16 * S * S * 8 / 2**30:.0f} GiB of "
                      f"scores; refused (--eager-max-gib {a.eager_max_gib})", flush=True)
                continue
            CAPTURE["ops"], CAPTURE["on"] = {}, bool(a.ops)
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
            t0 = time.time()
            if ctxf:
                with ctxf():
                    rows, feeds = run_tower(tower, pv, grid, dtype)
            else:
                rows, feeds = run_tower(tower, pv, grid, dtype)
            torch.cuda.synchronize(dev)
            wall = time.time() - t0
            CAPTURE["on"] = False
            key = ("packed:" if a.packed else "") + "+".join(g[0] for g in grp)
            splits = rows.split(grid.prod(-1).tolist())
            for (n, _, g), r in zip(grp, splits):
                rec = {"rows": r.contiguous().cpu(), "grid": g.to(torch.int64)}
                if len(grp) == 1:
                    rec.update(feeds)
                tag = f"{n}.{form}{'.packed' if a.packed else ''}"
                path = os.path.join(out, tag + ".safetensors")
                save_file({k: v.contiguous() for k, v in rec.items()}, path)
                man["runs"][tag] = {"image": n, "form": form, "attn_impl": attn, "dtype": str(dtype),
                                    "bf16_reduced_precision_reduction":
                                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                    "packed_with": [x[0] for x in grp], "grid": g[0].tolist(),
                                    "tokens": int(g.prod()), "tower_rows": int(g.prod()) * 4 + 1,
                                    "rows_sha256": tensor_sha(r), "file_sha256": R.sha256(path),
                                    "wall_s_group": round(wall, 3),
                                    "peak_gib_group": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
                                    "weights_sha256": wsha}
                print(f"{tag}: rows {tuple(r.shape)} sha {man['runs'][tag]['rows_sha256'][:16]} "
                      f"{wall:.2f} s peak {man['runs'][tag]['peak_gib_group']} GiB", flush=True)
            if a.ops and len(grp) == 1:
                ops = {}
                for k, v in CAPTURE["ops"].items():
                    ops[k] = v[0] if len(v) == 1 else torch.cat([x.reshape(-1, x.shape[-1]) for x in v])
                opath = os.path.join(out, f"{grp[0][0]}.{form}.ops.safetensors")
                save_file({k: v.contiguous() for k, v in ops.items()}, opath)
                man["runs"][f"{grp[0][0]}.{form}"]["ops_file_sha256"] = R.sha256(opath)
        for h in hooks:
            h.remove()
        del tower
        torch.cuda.empty_cache()
    if forms and forms[0].split("+")[0].startswith("sdpa"):
        man["sdpa_flags"] = {"flash": torch.backends.cuda.flash_sdp_enabled(),
                             "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                             "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                             "math": torch.backends.cuda.math_sdp_enabled()}
    return out, man


def cmd_tables(a, dev):
    from transformers.models.inkling2 import modeling_inkling2 as M

    out = os.path.join(a.out, "tables")
    os.makedirs(out, exist_ok=True)
    rec = {}
    allbits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(BF).to(dev)
    rec["gelu.in"] = allbits.cpu()
    rec["gelu.out"] = F.gelu(allbits).cpu()  # ACT2FN["gelu"]: GELUActivation, the erf form
    cfg = vision_config(a.model_dir, "eager")
    inv, _ = M.Inkling2VisionRotaryEmbedding.compute_axial_rope_parameters(cfg)
    pos = torch.arange(0, 416, device=dev)
    freqs = pos[:, None].float() * inv.to(dev)[None]
    rec["rope.inv_freq"] = inv.cpu()
    rec["rope.cos"] = freqs.cos().cpu()
    rec["rope.sin"] = freqs.sin().cpu()
    grids = [(1, r, c) for r, c in ((1, 1), (9, 12), (20, 26), (26, 26), (27, 48), (52, 52), (54, 96), (103, 103))]
    for g in grids:
        thw = torch.tensor([g]) * torch.tensor([1, 2, 2])
        idx, w = M.get_vision_interpolation_indices_and_weights(thw, num_grid_per_side=cfg.posemb_grid_size,
                                                                 mode="bilinear", align_corners=False,
                                                                 spatial_merge_size=2, kwargs={})
        rec[f"pos.idx.{g[1]}x{g[2]}"] = idx.to(torch.int64).cpu()
        rec[f"pos.w.{g[1]}x{g[2]}"] = w.float().cpu()
        pid = M.get_vision_position_ids(thw, 2, kwargs={})
        rec[f"pos.ids.{g[1]}x{g[2]}"] = pid.to(torch.int64).cpu()
    path = os.path.join(out, "tables.safetensors")
    save_file({k: v.contiguous() for k, v in rec.items()}, path)
    return out, {"files_out_sha256": {"tables.safetensors": R.sha256(path)}, "grids": grids}


# --- image prompts through the text oracle

SETS = {
    # (name, effort, [(fixture, ...)], question): one user message, the images first, then the question
    "g0": [("scene_448x336", "none", ["scene_448x336.png"], "Describe this image in two sentences.")],
    "probe": [("four_2048", "none", ["scene_2048x2048.png"] * 4, "What do these four pictures have in common?"),
              ("one_4096", "none", ["scene_4096x4096.png"], "Describe this image.")],
    "top1": [
        ("t01", "none", ["scene_40x40.png"], "What colour is this image mostly?"),
        ("t02", "none", ["scene_448x336.png"], "Describe this image in two sentences."),
        ("t03", "none", ["chart_448x336.png"], "Which bar is the tallest, and what is its value?"),
        ("t04", "none", ["scene_1024x768.png"], "What does the sign say?"),
        ("t05", "none", ["chart_1024x768.png"], "List the five bar values from left to right."),
        ("t06", "none", ["scene_1024x1024.png"], "Is there a house in this picture? Describe it."),
        ("t07", "none", ["scene_1920x1080.png"], "What time of day does this look like, and why?"),
        ("t08", "none", ["chart_1920x1080.png"], "Summarise this chart."),
        ("t09", "none", ["rgba_448x336.png"], "Describe what you see, including the background."),
        ("t10", "none", ["scene_448x336.png", "chart_448x336.png"], "Compare the two images."),
        ("t11", "low", ["scene_2048x2048.png"], "Count the round shapes in this picture."),
        ("t12", "none", ["scene_2560x1440.png"], "Write a one-line caption for this picture."),
    ],
}


def image_specs(proc, fixtures: str, set_name: str):
    from PIL import Image  # noqa: F401

    specs = []
    for name, effort, files, q in SETS[set_name]:
        content = [{"type": "image"} for _ in files] + [{"type": "text", "text": q}]
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
        imgs = [open_upright(os.path.join(fixtures, f)) for f in files]
        enc = proc(text=[text], images=imgs, return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        specs.append({"name": name, "effort": effort, "messages": msgs, "text": text, "ids": ids, "files": files,
                      "pixel_values": enc["pixel_values"], "grid": enc["image_grid_thw"],
                      "image_tokens": int(sum(1 for t in ids if t == IMAGE_TOKEN))})
    return specs


class SplicedEmbed(torch.nn.Module):
    """The normed embedding, then the tower's rows over the 200054 rows of a prompt that was given its images
    (`Inkling2Model.forward`: masked_scatter after the embedding). Other calls (decode steps) are plain text."""

    def __init__(self, emb, rows_by_prompt: dict):
        super().__init__()
        self.emb, self.rows = emb, rows_by_prompt

    def forward(self, ids):
        e = self.emb(ids)
        if ids.shape[1] > 1:
            key = tuple(ids[0].tolist())
            if key in self.rows:
                r = self.rows[key].to(e.device, e.dtype)
                mask = (ids == IMAGE_TOKEN)[..., None].expand_as(e)
                if int((ids == IMAGE_TOKEN).sum()) != r.shape[0]:
                    raise RuntimeError("image rows and placeholders differ")
                e = e.masked_scatter(mask, r)
        return e


def tower_rows_for(specs, a, dev):
    """Each prompt's tower rows (all its images in one call, as get_image_features does), in --tower-form."""
    register_cores()
    attn, dtype, ctxf = form_context(a.tower_form)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = "+red32" not in a.tower_form
    tower, cfg, wsha = build_tower(a.model_dir, attn, dev, dtype)
    rows, info = {}, {}
    for s in specs:
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        pv, grid = s["pixel_values"].to(dev), s["grid"]
        if ctxf:
            with ctxf():
                r, _ = run_tower(tower, pv, grid, dtype, layer_rows=False)
        else:
            r, _ = run_tower(tower, pv, grid, dtype, layer_rows=False)
        torch.cuda.synchronize(dev)
        rows[tuple(s["ids"])] = r.to(BF).cpu()
        info[s["name"]] = {"tower_s": round(time.time() - t0, 3), "rows_sha256": tensor_sha(r.to(BF)),
                           "tower_peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
                           "grid": grid.tolist(), "image_tokens": s["image_tokens"]}
        print(f"{s['name']}: tower {info[s['name']]['tower_s']} s, rows {tuple(r.shape)}, "
              f"peak {info[s['name']]['tower_peak_gib']} GiB", flush=True)
    del tower
    torch.cuda.empty_cache()
    return rows, info, wsha


def cmd_greedy(a, dev, probe: bool = False):
    from transformers import AutoProcessor

    R.ATTN["form"] = a.attn
    proc = AutoProcessor.from_pretrained(a.model_dir)
    specs = image_specs(proc, a.fixtures, a.set)
    if a.only:
        specs = [s for s in specs if s["name"] in a.only.split(",")]
    out = os.path.join(a.out, ("probe" if probe else "greedy") + f"-{a.set}" + (f"-{a.only}" if a.only else ""),
                       a.mode)
    os.makedirs(out, exist_ok=True)
    rows, info, wsha = tower_rows_for(specs, a, dev)
    # the text oracle's setting (inkling2_ref.main): --attn fp32 reduces bf16 matmuls in fp32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = a.attn != "fp32"
    cfg = R.text_config(a.model_dir, a.layers)
    model = R.Model(a.model_dir, cfg, dev, a.mode)
    orig = model.embed
    model.embed = lambda: SplicedEmbed(orig(), rows)
    man = R.provenance(a.model_dir)
    man.update({"set": a.set, "mode": a.mode, "attn": a.attn, "tower_form": a.tower_form, "tower_weights": wsha,
                "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "prompts": [{k: s[k] for k in ("name", "effort", "messages", "text", "files", "image_tokens")}
                            | {"tokens": len(s["ids"]), "ids_sha": R.ids_sha(s["ids"])} for s in specs],
                "tower": info, "pythonpath": os.environ.get("PYTHONPATH", "")})
    import PIL
    import torchvision
    man.update({"pillow": PIL.__version__, "torchvision": torchvision.__version__})
    t0 = time.time()
    if probe:
        torch.cuda.reset_peak_memory_stats(dev)
        files, times = R.teacher_forced(model, specs, out, cfg.num_hidden_layers)
        man.update({"files_out_sha256": files, "layer_times_s": times, "prefill_wall_s": round(time.time() - t0, 1),
                    "prefill_peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)})
    else:
        margins = [[] for _ in specs]
        outs, report = R.greedy_decode(model, specs, a.decode, torch.device(a.expert_device), a.expert_cache, None,
                                       margins=margins)
        tok = proc.tokenizer
        res = {}
        for s, o, mg in zip(specs, outs, margins):
            res[s["name"]] = {"prompt_ids_sha": R.ids_sha(s["ids"]), "ids": o, "ids_sha": R.ids_sha(o),
                              "text": tok.decode(o, skip_special_tokens=False), **report[s["name"]], "margins": mg}
            print(f"--- {s['name']}\n{res[s['name']]['text']}", flush=True)
        with open(os.path.join(out, "decode.json"), "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        man.update({"decode_wall_s": round(time.time() - t0, 1),
                    "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
                    "decode_sha256": R.sha256(os.path.join(out, "decode.json"))})
    for s in specs:  # the prompts' ids, for the engine side
        save_file({"ids": torch.tensor(s["ids"], dtype=torch.int64), "rows": rows[tuple(s["ids"])],
                   "pixel_values": s["pixel_values"].float().contiguous(), "grid": s["grid"].to(torch.int64)},
                  os.path.join(out, s["name"] + ".prompt.safetensors"))
    return out, man


def cmd_export(a, dev):
    """The engine's inputs per fixture: the processor's RGB (alpha composited, upright; recovered exactly from its
    fp32 pixel values, which are checked to round-trip bit for bit), its size, grid and pixel values."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(a.model_dir)
    out = os.path.join(a.out, "inputs")
    os.makedirs(out, exist_ok=True)
    man = {"files_out_sha256": {}, "inputs": {}}
    for n in a.images.split(","):
        im = open_upright(os.path.join(a.fixtures, n))
        W_, H_ = im.size
        pv, grid = processed(proc, os.path.join(a.fixtures, n))
        rows, cols = int(grid[0, 1]), int(grid[0, 2])
        full = pv.reshape(rows, cols, TOKEN_PX, TOKEN_PX, 3).permute(0, 2, 1, 3, 4).reshape(rows * TOKEN_PX,
                                                                                            cols * TOKEN_PX, 3)
        x = torch.round(full[:H_, :W_].double() * 127.5 + 127.5)
        if x.min() < 0 or x.max() > 255:
            raise SystemExit(f"{n}: pixel values outside 0..255")
        rgb = x.to(torch.uint8)
        back = (rgb.float() - 127.5) / 127.5
        if not torch.equal(back, full[:H_, :W_]):
            raise SystemExit(f"{n}: RGB does not round-trip to the processor's pixel values")
        pad = full.clone()
        pad[:H_, :W_] = 0
        if H_ < rows * TOKEN_PX or W_ < cols * TOKEN_PX:
            padv = full[H_:].flatten().tolist()[:1] + full[:, W_:].flatten().tolist()[:1]
            if any(v != float((torch.tensor(-1.0) - 127.5) / 127.5) for v in padv):  # fp32, as the processor
                raise SystemExit(f"{n}: unexpected padding value {padv}")
        rec = {"rgb": rgb.contiguous(), "hw": torch.tensor([H_, W_], dtype=torch.int32),
               "grid": grid[0].to(torch.int64), "pixel_values": pv.float().contiguous()}
        path = os.path.join(out, n + ".input.safetensors")
        save_file(rec, path)
        man["files_out_sha256"][os.path.basename(path)] = R.sha256(path)
        man["inputs"][n] = {"h": H_, "w": W_, "grid": [rows, cols], "rgb_sha256": tensor_sha(rgb),
                            "pixel_values_sha256": tensor_sha(pv.float())}
        print(f"{n}: {H_}x{W_} grid {rows}x{cols} rgb {man['inputs'][n]['rgb_sha256'][:16]}", flush=True)
    return out, man


def cmd_compare(a, dev):
    """Each tower form's output rows (and sampled layer rows) against the fp64 form's, per image: the distances of
    `distances`, and which forms are bit-equal to each other."""
    from safetensors.torch import load_file

    src = os.path.join(a.out, "tower")
    man = json.load(open(os.path.join(src, "manifest.json")))
    res = {}
    for img in sorted({r["image"] for r in man["runs"].values()}):
        tags = {r["form"] + (".packed" if len(r["packed_with"]) > 1 else ""): t for t, r in man["runs"].items()
                if r["image"] == img}
        if "fp64" not in tags:
            continue
        ref = load_file(os.path.join(src, tags["fp64"] + ".safetensors"))
        out = {}
        for f, t in sorted(tags.items()):
            got = load_file(os.path.join(src, t + ".safetensors"))
            d = {"rows": distances(got["rows"], ref["rows"]), "rows_sha256": man["runs"][t]["rows_sha256"]}
            layers = {}
            for k in got:
                if k.startswith("layer.") and k != "layer.rows" and k in ref:
                    layers[k] = distances(got[k], ref[k])["rel_norm"]
            d["layer_rel_norm"] = layers
            out[f] = d
        shas = {}
        for f, d in out.items():
            shas.setdefault(d["rows_sha256"], []).append(f)
        res[img] = {"forms": out, "bit_equal_groups": [g for g in shas.values()]}
        print(f"{img}:")
        for f, d in out.items():
            r = d["rows"]
            print(f"  {f:22s} rel {r['rel_norm']:.3e} max {r['max_abs']:.3e} mean {r['mean_abs']:.3e} "
                  f"bit=ref {r['bit_equal_to_rounded_ref']:.3f}", flush=True)
    dst = os.path.join(a.out, "compare")
    os.makedirs(dst, exist_ok=True)
    json.dump(res, open(os.path.join(dst, "compare.json"), "w"), indent=1)
    return dst, {"tower_manifest_sha256": R.sha256(os.path.join(src, "manifest.json"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["fixtures", "process", "tower", "tables", "greedy", "probe", "compare", "export"])
    ap.add_argument("model_dir", help="checkpoint folder (fixtures: the output folder)")
    ap.add_argument("--out", default="")
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--images", default="", help="fixture file names, comma-separated")
    ap.add_argument("--forms", default="sdpa")
    ap.add_argument("--packed", action="store_true", help="tower: all --images in one call (as one request)")
    ap.add_argument("--ops", action="store_true", help="tower: record every op's input and output")
    ap.add_argument("--op-layers", default="0,11,22")
    ap.add_argument("--eager-max-gib", type=float, default=40.0)
    ap.add_argument("--q-block", type=int, default=1024, help="query block of the f32core and fp64 cores")
    ap.add_argument("--set", default="g0")
    ap.add_argument("--mode", choices=["full", "checkpoint"], default="checkpoint")
    ap.add_argument("--attn", choices=["eager", "fp32"], default="fp32", help="the text oracle's core (v0.2: fp32)")
    ap.add_argument("--tower-form", default="sdpa+red32", help="greedy/probe: the tower form whose rows enter the prompt")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--only", default="", help="greedy/probe: only these prompts of --set")
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--expert-device", default="cuda:0")
    ap.add_argument("--expert-cache", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    BLOCK["q"] = a.q_block
    if a.cmd == "fixtures":
        made = make_fixtures(a.model_dir)
        print(f"{len(made)} fixtures in {a.model_dir}")
        return 0
    dev = torch.device(a.device)
    tf32 = R.tf32_guard(dev)
    t0 = time.time()
    if a.cmd == "process":
        out, man = cmd_process(a, dev)
    elif a.cmd == "tower":
        out, man = cmd_tower(a, dev)
    elif a.cmd == "tables":
        out, man = cmd_tables(a, dev)
    elif a.cmd == "compare":
        out, man = cmd_compare(a, dev)
    elif a.cmd == "export":
        out, man = cmd_export(a, dev)
    else:
        out, man = cmd_greedy(a, dev, probe=a.cmd == "probe")
    import transformers
    man.update({"cmd": a.cmd, "argv": sys.argv[1:], "tf32": tf32, "torch": torch.__version__,
                "cuda": torch.version.cuda, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None, "wall_s": round(time.time() - t0, 1),
                "tool_sha256": R.sha256(os.path.abspath(__file__)),
                "text_oracle_sha256": R.sha256(os.path.abspath(R.__file__))})
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    print(f"done: {out} ({man['wall_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
