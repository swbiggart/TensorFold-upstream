"""Inkling-2 image fixtures and what Transformers' Inkling2ImageProcessor makes of them (gate V1's oracle).

Each case is opened with Pillow, turned upright (ImageOps.exif_transpose) and handed to the processor as the opened
image, the route that composites alpha onto the processor's checkerboard (design decision D1). For each case the
output records the upright size, the RGB bytes the processor's convert_to_rgb gives (sha256), the grid and the fp32
pixel values (sha256 of their little-endian bytes, [rows * cols, 40, 40, 3]).

  small: python -I inkling2_image_fixtures.py --out zig/tests/fixtures/inkling2/images.json
  large: python -I inkling2_image_fixtures.py --large --out DIR/images.json   (pictures written beside it)
  check: python -I inkling2_image_fixtures.py --check JSON   (the oracle here, on that json's own pictures)

--check is gate V1's oracle run: encoders differ by platform, so the oracle is rerun on the committed bytes and every
recorded field must come out equal (exit 1 on any difference, 2 on an empty set).

The Zig side is zig/src/families/inkling2/image_test.zig; TF_INK2_IMAGE_FIXTURES names a large set's json.
Needs Pillow 12.3.0 (libjpeg-turbo 3.1.4.1, libwebp 1.6.0: the decoders TensorFold pins), torch, torchvision and
Transformers with models/inkling2 (the oracle's 109a7d19).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import platform
import sys

import numpy as np
from PIL import Image, ImageOps, features

SIDE = 4096  # the recipe's limit a side (design D2)
PIXELS = 16 << 20  # TensorFold's decoded-pixel bound


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def gradient(w: int, h: int, channels: int, seed: int) -> np.ndarray:
    """Smooth ramps plus a noisy channel: every byte value appears, and the files stay small."""
    y, x = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(seed)
    a = np.empty((h, w, 4), np.uint8)
    a[..., 0] = (x * 255 // max(w - 1, 1)).astype(np.uint8)
    a[..., 1] = (y * 255 // max(h - 1, 1)).astype(np.uint8)
    a[..., 2] = ((x * 7 + y * 13) % 256).astype(np.uint8)
    # alpha: every value 0..255 along a diagonal, with fully clear and fully opaque runs
    a[..., 3] = np.clip(((x + 2 * y) % 320) - 32, 0, 255).astype(np.uint8)
    if min(w, h) <= 64:
        a[..., 2] = rng.integers(0, 256, (h, w), dtype=np.uint8)
    return a[..., :channels]


def enc(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def exif_with(o: int) -> bytes:
    e = Image.Exif()
    e[0x0112] = o
    return e.tobytes()


def small_cases() -> list[tuple[str, bytes]]:
    rgb = lambda w, h, s=1: Image.fromarray(gradient(w, h, 3, s))
    rgba = lambda w, h, s=1: Image.fromarray(gradient(w, h, 4, s))
    out: list[tuple[str, bytes]] = []
    # PNG layouts (TensorFold's own decoder)
    out.append(("png_rgb_1x1", enc(rgb(1, 1), "PNG")))
    out.append(("png_rgb_39x41", enc(rgb(39, 41), "PNG")))
    out.append(("png_rgb_40x40", enc(rgb(40, 40), "PNG")))
    out.append(("png_rgb_41x39", enc(rgb(41, 39), "PNG")))
    out.append(("png_rgb_448x336", enc(rgb(448, 336), "PNG")))
    out.append(("png_rgba_partial_97x61", enc(rgba(97, 61), "PNG")))
    out.append(("png_rgba_opaque_50x30", enc(Image.fromarray(np.dstack([gradient(50, 30, 3, 2), np.full((30, 50), 255, np.uint8)])), "PNG")))
    out.append(("png_rgba_clear_17x9", enc(Image.fromarray(np.dstack([gradient(17, 9, 3, 3), np.zeros((9, 17), np.uint8)])), "PNG")))
    out.append(("png_interlaced_rgba_83x45", interlaced(rgba(83, 45, 4))))
    pal = rgb(64, 48).quantize(colors=200)
    out.append(("png_palette_64x48", enc(pal, "PNG")))
    out.append(("png_palette_trns_64x48", enc(pal, "PNG", transparency=bytes((i * 37) % 256 for i in range(200)))))
    g = Image.fromarray(gradient(70, 33, 3, 5)[..., 0])
    out.append(("png_gray_70x33", enc(g, "PNG")))
    out.append(("png_gray_trns_70x33", enc(g, "PNG", transparency=128)))
    la = Image.fromarray(np.dstack([gradient(45, 50, 3, 6)[..., 1], gradient(45, 50, 4, 6)[..., 3]]), "LA")
    out.append(("png_gray_alpha_45x50", enc(la, "PNG")))
    out.append(("png_rgb_trns_33x21", enc(rgb(33, 21, 7), "PNG", transparency=(0, 0, 0))))
    g16 = Image.fromarray((gradient(60, 20, 3, 8)[..., 0].astype(np.uint16) * 257 + np.arange(60, dtype=np.uint16)[None, :]).astype(np.uint16), "I;16")
    out.append(("png_gray16_60x20", enc(g16, "PNG")))
    out.append(("png_1bit_37x29", enc(rgb(37, 29).convert("1"), "PNG")))
    for o in (3, 6, 8):
        out.append((f"png_exif{o}_rgba_43x27", enc(rgba(43, 27, 9), "PNG", exif=exif_with(o))))
    # JPEG (libjpeg-turbo 3.1.4.1)
    out.append(("jpeg_baseline_448x336", enc(rgb(448, 336), "JPEG", quality=85)))
    out.append(("jpeg_progressive_123x77", enc(rgb(123, 77, 10), "JPEG", quality=90, progressive=True)))
    out.append(("jpeg_444_61x39", enc(rgb(61, 39, 11), "JPEG", quality=92, subsampling=0)))
    out.append(("jpeg_422_61x39", enc(rgb(61, 39, 11), "JPEG", quality=92, subsampling=1)))
    out.append(("jpeg_420_odd_41x39", enc(rgb(41, 39, 12), "JPEG", quality=75, subsampling=2)))
    out.append(("jpeg_gray_88x40", enc(rgb(88, 40, 13).convert("L"), "JPEG", quality=80)))
    out.append(("jpeg_1x1", enc(rgb(1, 1), "JPEG")))
    for o in range(1, 9):
        out.append((f"jpeg_exif{o}_57x31", enc(rgb(57, 31, 14), "JPEG", quality=88, exif=exif_with(o))))
    out.append(("jpeg_cmyk_16x16", enc(rgb(16, 16).convert("CMYK"), "JPEG")))
    # a multi-picture JPEG: Pillow opens an MpoImageFile of two frames, which the processor's route refuses
    out.append(("jpeg_mpo_two_48x32", enc(rgb(48, 32, 22), "MPO", save_all=True, append_images=[rgb(48, 32, 23)])))
    # WebP (libwebp 1.6.0)
    out.append(("webp_lossy_157x93", enc(rgb(157, 93, 15), "WEBP", quality=80)))
    out.append(("webp_lossless_64x48", enc(rgb(64, 48, 16), "WEBP", lossless=True)))
    out.append(("webp_lossy_alpha_57x41", enc(rgba(57, 41, 17), "WEBP", quality=70)))
    out.append(("webp_lossless_alpha_40x80", enc(rgba(40, 80, 18), "WEBP", lossless=True)))
    out.append(("webp_exif6_alpha_30x20", enc(rgba(30, 20, 19), "WEBP", lossless=True, exif=exif_with(6))))
    # refusals before any pixel is allocated
    out.append(("png_side_4097x1", enc(rgb(4097, 1), "PNG")))
    out.append(("png_side_1x4097", enc(rgb(1, 4097), "PNG")))
    frames = [rgb(16, 16, s) for s in (20, 21)]
    out.append(("png_animated_16x16", enc(frames[0], "PNG", save_all=True, append_images=frames[1:])))
    out.append(("gif_16x16", enc(rgb(16, 16), "GIF")))
    return out


def interlaced(im: Image.Image) -> bytes:
    """Pillow cannot write Adam7; build it from the rows with zlib (filter 0 per pass row)."""
    import struct
    import zlib

    a = np.asarray(im.convert("RGBA"))
    h, w = a.shape[:2]
    raw = bytearray()
    for x0, y0, dx, dy in ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)):
        sub = a[y0::dy, x0::dx]
        if sub.shape[0] == 0 or sub.shape[1] == 0:
            continue
        for row in sub:
            raw += b"\x00" + row.tobytes()

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 1)) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")


def large_cases() -> list[tuple[str, bytes]]:
    rgb = lambda w, h, s=1: Image.fromarray(gradient(w, h, 3, s))
    rgba = lambda w, h, s=1: Image.fromarray(gradient(w, h, 4, s))
    out: list[tuple[str, bytes]] = []
    for w, h in ((1024, 768), (1920, 1080), (2048, 2048), (3840, 2160), (4096, 4096)):
        out.append((f"png_rgb_{w}x{h}", enc(rgb(w, h, w + h), "PNG")))
        out.append((f"jpeg_{w}x{h}", enc(rgb(w, h, w + h), "JPEG", quality=85)))
    out.append(("png_rgba_1920x1080", enc(rgba(1920, 1080, 31), "PNG")))
    out.append(("webp_lossy_1920x1080", enc(rgb(1920, 1080, 32), "WEBP", quality=80)))
    out.append(("webp_alpha_1024x768", enc(rgba(1024, 768, 33), "WEBP", quality=75)))
    out.append(("jpeg_exif6_4032x3024", enc(rgb(4032, 3024, 34), "JPEG", quality=85, exif=exif_with(6))))
    out.append(("png_4096x4097", enc(rgb(4096, 4097, 35), "PNG")))
    return out


def processor():
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor

    # processor_config.json's image_processor block (its values are the class defaults)
    return Inkling2ImageProcessor(do_convert_rgb=True, do_normalize=True, do_rescale=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], rescale_factor=0.00392156862745098, size={"height": 40, "width": 40})


def oracle(proc, data: bytes) -> dict:
    """The processor's outcome for one picture, or the refusal the server must give."""
    import torch
    from torchvision.transforms.v2 import functional as tvF

    im = Image.open(io.BytesIO(data))
    if im.format not in ("PNG", "JPEG", "MPO", "WEBP"):  # MPO: Pillow's name for a multi-picture JPEG
        return {"problem": "format"}
    if getattr(im, "n_frames", 1) != 1:  # animated PNG or WebP, multi-picture JPEG: one frame is served
        return {"problem": "animated"}
    if im.mode == "CMYK":
        return {"problem": "cmyk"}
    if max(im.size) > SIDE or im.size[0] * im.size[1] > PIXELS:
        return {"problem": "side", "width": im.size[0], "height": im.size[1]}
    im = ImageOps.exif_transpose(im)
    # the RGB the processor composites (process_image, then convert_to_rgb), recorded for the decode check
    t = tvF.pil_to_tensor(im.convert("RGBA" if im.has_transparency_data else "RGB"))
    rgb = proc.convert_to_rgb(t) if t.shape[0] == 4 else t
    assert rgb.dtype == torch.uint8 and rgb.shape[0] == 3, (rgb.dtype, rgb.shape)
    hwc = rgb.permute(1, 2, 0).contiguous().numpy()
    out = proc(images=[im], return_tensors="pt")
    pv = out["pixel_values"]
    thw = out["image_grid_thw"].tolist()[0]
    assert pv.dtype == torch.float32 and tuple(pv.shape[1:]) == (40, 40, 3), (pv.dtype, pv.shape)
    vals = pv.contiguous().numpy().astype("<f4", copy=False)
    return {
        "width": im.width,
        "height": im.height,
        "grid": [thw[1], thw[2]],
        "rgb": sha(hwc.tobytes()),
        "values": sha(vals.tobytes()),
        "mode": im.mode,
    }


def check(proc, path: str) -> None:
    with open(path) as f:
        want = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    equal, differ = 0, []
    for c in want["cases"]:
        data = base64.b64decode(c["b64"]) if "b64" in c else open(os.path.join(base, c["file"]), "rb").read()
        got = {"name": c["name"], "bytes": sha(data)}
        got.update(oracle(proc, data))
        keys = [k for k in c if k not in ("b64", "file")]
        bad = [k for k in sorted(set(keys) | set(got)) if c.get(k) != got.get(k)]
        if bad:
            differ.append((c["name"], {k: (c.get(k), got.get(k)) for k in bad}))
        else:
            equal += 1
    print(f"CHECK {path}: {equal}/{len(want['cases'])} cases equal on every field ({sum('problem' in c for c in want['cases'])} refusals)")
    for name, d in differ:
        print("  DIFFER", name, d)
    if not want["cases"]:
        sys.exit(2)
    if differ:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--large", action="store_true", help="the large set: pictures written beside --out, not inlined")
    ap.add_argument("--check", help="a fixtures json: rerun the oracle on its pictures and compare every field")
    args = ap.parse_args()
    if not args.out and not args.check:
        ap.error("--out or --check")
    import PIL
    import torch
    import torchvision
    import transformers

    assert PIL.__version__ == "12.3.0", PIL.__version__
    assert features.version_feature("libjpeg_turbo") == "3.1.4.1", features.version_feature("libjpeg_turbo")
    assert features.version("webp") == "1.6.0", features.version("webp")
    proc = processor()
    if args.check:
        check(proc, args.check)
        return
    cases = []
    for name, data in large_cases() if args.large else small_cases():
        c = {"name": name, "bytes": sha(data)}
        if args.large:
            fn = name + {"\x89P": ".png", "\xff\xd8": ".jpg", "RI": ".webp"}.get(data[:2].decode("latin-1"), ".bin")
            with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), fn), "wb") as f:
                f.write(data)
            c["file"] = fn
        else:
            c["b64"] = base64.b64encode(data).decode()
        c.update(oracle(proc, data))
        cases.append(c)
        print(name, c.get("grid", c.get("problem")), file=sys.stderr)
    source = {
        "route": "Image.open, ImageOps.exif_transpose, Inkling2ImageProcessor(images=[opened image]) (alpha onto its checkerboard)",
        "pillow": PIL.__version__,
        "libjpeg_turbo": features.version_feature("libjpeg_turbo"),
        "libwebp": features.version("webp"),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "machine": platform.machine(),
    }
    with open(args.out, "w") as f:
        json.dump({"source": source, "cases": cases}, f, indent=0)
        f.write("\n")


if __name__ == "__main__":
    main()
