"""Chain tf-inkling2-cuda's layer ranges into one full-depth run: the runner holds about 12 layers a card, so a
48-layer chained forward runs as ranges, each fed the previous ranges' outputs through a stitched prompt file.

USAGE = python tools/zig/inkling2_chain.py stitch BASE OUT [DUMP ...]
        python tools/zig/inkling2_chain.py head MODEL_DIR REF NAME=FILE [NAME=FILE ...] [--json OUT]

  stitch  OUT = BASE (an oracle prompt file with every row: inkling2_ref.py --all-rows, or a short prompt) with every
          h.<L>, route.*.<L>, final and logits.full that a DUMP holds replaced by the dump's. Run range [a, b) of
          tf-inkling2-cuda on OUT and its layer a input and block states are our own earlier layers' outputs; the
          last OUT, stitched from every range's dump, is our whole chained forward in the oracle's format, ready for
          inkling2_compare.py as a dump (metadata: status complete, every row done).
  head    The head's predictions at every row, not only the oracle's few logits.rows: torch's unembed and softcap
          (fp32 reductions) on each file's `final` rows, the same arithmetic for all, so the only difference is the
          final hidden rows. Against REF: top-1 agreement, REF's top-1 within the other's top 5, and KL(REF || NAME)
          per row (mean, median, 99th percentile, max). --logits-rows: only REF's logits.rows.
"""

import argparse
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def stitch(a):
    b = safe_open(a.base, framework="pt")
    out = {k: b.get_tensor(k) for k in b.keys()}
    T = out["ids"].shape[0]
    if out["rows"].shape[0] != T:
        raise SystemExit("BASE keeps sampled rows: use an oracle made with --all-rows")
    took = []
    for p in a.dumps:
        d = safe_open(p, framework="pt")
        meta = d.metadata() or {}
        if meta.get("status") != "complete" or int(meta.get("rows_done", T)) != T:
            raise SystemExit(f"{p}: the run did not finish ({meta.get('status')})")
        n = 0
        for k in d.keys():
            if k in out and (k.startswith("h.") or k.startswith("route.") or k in ("final", "logits.full")):
                x = d.get_tensor(k)
                if x.shape != out[k].shape or (x.dtype != out[k].dtype and x.is_floating_point()):
                    raise SystemExit(f"{p}: {k} is {tuple(x.shape)} {x.dtype}, BASE's {tuple(out[k].shape)} {out[k].dtype}")
                out[k] = x.to(out[k].dtype)  # expert ids: int32 in a dump, the oracle's integer type in BASE
                n += 1
        took.append(f"{os.path.basename(p)} ({meta.get('layers', '?')}, chunk {meta.get('chunk', '?')}): {n}")
    meta = {"tool": "inkling2_chain.py stitch", "base": os.path.basename(a.base), "prompt": os.path.basename(a.base),
            "rows": str(T), "rows_done": str(T), "status": "complete", "chunk": "see sources",
            "sources": "; ".join(took) or "none", "layers": "stitched"}
    save_file({k: v.contiguous() for k, v in out.items()}, a.out, metadata=meta)
    print(f"{a.out}: {len(out)} tensors; from the dumps: " + ("; ".join(took) or "none"))
    return 0


def head(a):
    import json

    import inkling2_ref as ref

    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cfg = ref.text_config(a.model_dir, 0)
    hd = ref.Model(a.model_dir, cfg, dev, "full").head()

    def logits(path, lo, hi):
        y = safe_open(path, framework="pt").get_slice("final")[lo:hi].to(dev)
        return ref.logits_of(hd, cfg, y).double()

    T = safe_open(a.ref, framework="pt").get_slice("final").get_shape()[0]
    want = None
    if a.logits_rows:  # only the reference's logits.rows (the oracle's own head rows)
        want = safe_open(a.ref, framework="pt").get_tensor("logits.rows").to(torch.int64)
        if bool((want >= T).any()):
            raise SystemExit("REF's logits.rows reach past its final rows")
    out = {}
    for spec in a.files:
        name, path = spec.split("=", 1)
        top1 = in5 = 0
        kls = []
        for lo in range(0, T, 128):
            hi = min(T, lo + 128)
            lr, lc = logits(a.ref, lo, hi), logits(path, lo, hi)
            if want is not None:
                sel = want[(want >= lo) & (want < hi)] - lo
                if sel.numel() == 0:
                    continue
                lr, lc = lr[sel.to(lr.device)], lc[sel.to(lc.device)]
            pr, pc = torch.log_softmax(lr, -1), torch.log_softmax(lc, -1)
            kls.append((pr.exp() * (pr - pc)).sum(-1).cpu())
            ar, ac = lr.argmax(-1), lc.argmax(-1)
            top1 += int((ar == ac).sum())
            in5 += int((lc.topk(5, -1).indices == ar[:, None]).any(-1).sum())
        kl = torch.cat(kls)
        n = kl.numel()
        out[name] = {"rows": n, "top1_equal_frac": top1 / n, "ref_top1_in_top5_frac": in5 / n,
                     "kl_mean": float(kl.mean()), "kl_median": float(kl.median()), "kl_max": float(kl.max()),
                     "kl_p99": float(kl.quantile(0.99))}
        r = out[name]
        print(f"{name}: {n} rows; top-1 equal {r['top1_equal_frac']:.4f} ({top1}), reference's top-1 in its top 5"
              f" {r['ref_top1_in_top5_frac']:.4f}; KL(ref || {name}) mean {r['kl_mean']:.3e} median {r['kl_median']:.3e}"
              f" p99 {r['kl_p99']:.3e} max {r['kl_max']:.3e}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stitch")
    s.add_argument("base")
    s.add_argument("out")
    s.add_argument("dumps", nargs="*")
    h = sub.add_parser("head")
    h.add_argument("model_dir")
    h.add_argument("ref")
    h.add_argument("files", nargs="+", help="NAME=FILE: files with every row's final")
    h.add_argument("--json", default="")
    h.add_argument("--logits-rows", action="store_true", help="only REF's logits.rows (the oracle's head rows)")
    h.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    return {"stitch": stitch, "head": head}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
