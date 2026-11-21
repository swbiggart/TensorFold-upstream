"""Compare a tf-inkling2-cuda dump with the oracle (tools/zig/inkling2_ref.py), layer by layer, or op by op.

USAGE = python tools/zig/inkling2_compare.py DUMP ORACLE_PROMPT [--ops CAPTURE] [--tol-rel 0.01]
            [--tol-route 0.0] [--model MODEL_DIR] [--tol-kl 0.01] [--json OUT]

  DUMP           the runner's output safetensors (h.<L>, route.*.<L>, final, logits.full, op.* with --dump-ops)
  ORACLE_PROMPT  the oracle's <prompt>.safetensors (full mode) the run started from
  --ops CAPTURE  also compare every op output the dump holds (op.<scope>.<group>.<op>.out.<t>) with the same
                 tensor of an inkling2_op_ref.py capture; ops the dump marks as fed stubs are skipped
  --model DIR    the checkpoint, for the router's selection bias: every row whose picks differ is explained by
                 the oracle's own scores (sigmoid(logit) + bias): the gap between the expert only the oracle picked
                 and the one only we picked, against how far our scores of that row are from the oracle's. A swap
                 whose oracle gap is within that distance is a tie within rounding.

Per layer: max |a - b|, the relative norm ||a - b|| / ||b|| (fp32), the bf16 ulp histogram (0, 1, 2, 3-4, 5-8,
9-16, >16 ulps; ordered bit distance), and for routes the fraction of rows whose 8 experts equal the oracle's
(as sets) with the largest weight difference on equal rows; route logits by max abs and relative norm.
A layer passes when its relative norm is within --tol-rel and its route-mismatch fraction within --tol-route
(with --model, swaps that are ties within rounding do not count). A layer with differing picks also prints its
hidden rows' distance over the rows whose picks are equal; with --model, when every differing pick is a tie, that
distance (not the whole layer's, which the swapped rows dominate) is the one held to --tol-rel.
Logits: relative norm, top-1 agreement, and KL(oracle || ours) per row over the softmax of the softcapped logits
(mean and max); they pass when the relative norm is within --tol-rel and the largest KL within --tol-kl.

Exit: 0 every compared tensor within tolerance and the run complete; 1 a tolerance exceeded (the line says
which); 3 within tolerance but the dump's status is not "complete" (missing, stopped early, or a failed rank
check): not a result; 2 nothing to compare.
"""

import argparse
import json
import sys

import numpy as np
from safetensors import safe_open

BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 1 << 30)]


class Store:
    """Tensors by name; bf16 read through torch as uint16 bits."""

    def __init__(self, path):
        import torch  # noqa: F401  (safetensors' bf16 needs the torch framework)

        self.path = path
        self.f = safe_open(path, framework="pt")
        self.meta = self.f.metadata() or {}
        self.keys = set(self.f.keys())

    def has(self, name):
        return name in self.keys

    def get(self, name):
        import torch

        t = self.f.get_tensor(name)
        if t.dtype == torch.bfloat16:
            return t.view(torch.int16).numpy().view(np.uint16), "bf16"
        return t.numpy(), str(t.dtype).replace("torch.", "")


def op_parts(name):
    """op.<scope>.<group>.<op>.<in|out>.<tensor> -> (scope, "group.op", dir, tensor), else None."""
    p = name.split(".")
    if len(p) < 6 or p[0] != "op":
        return None
    return p[1], f"{p[2]}.{p[3]}", p[4], ".".join(p[5:])


def op_key(name):
    p = op_parts(name)
    if p is None:
        return (2, 0, name)
    return (1 if p[0] == "model" else 0, int(p[0]) if p[0].isdigit() else 0, name)


def bf16_float(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def ordered(bits):
    """bf16 bits to integers whose difference counts representable steps (sign-magnitude to two's complement)."""
    b = bits.astype(np.int32)
    return np.where(b & 0x8000, -(b & 0x7FFF), b & 0x7FFF)


def stats(a, b, kind):
    """a: ours, b: reference, same shape. bf16 inputs as uint16 bits."""
    out = {"shape": list(a.shape)}
    if kind == "bf16":
        fa, fb = bf16_float(a), bf16_float(b)
        ulp = np.abs(ordered(a) - ordered(b))
        out["ulp_hist"] = [int(((ulp >= lo) & (ulp <= hi)).sum()) for lo, hi in BUCKETS]
        out["equal_frac"] = float((a == b).mean()) if a.size else 1.0
        out["differ"] = int((a != b).sum())
        out["max_ulp"] = int(ulp.max()) if a.size else 0
    elif kind in ("int32", "int64"):
        out["equal_frac"] = float((a == b).mean()) if a.size else 1.0
        out["differ"] = int((a != b).sum())
        return out
    else:
        fa, fb = a.astype(np.float32), b.astype(np.float32)
        out["equal_frac"] = float((fa == fb).mean()) if a.size else 1.0
        out["differ"] = int((fa != fb).sum())
    d = fa.astype(np.float64) - fb.astype(np.float64)
    nb = float(np.linalg.norm(fb.astype(np.float64)))
    out["max_abs"] = float(np.abs(d).max()) if d.size else 0.0
    out["rel"] = float(np.linalg.norm(d) / nb) if nb > 0 else float(np.linalg.norm(d))
    out["nonfinite"] = int((~np.isfinite(fa)).sum())
    return out


def routes(ids_a, ids_b, w_a, w_b):
    same = np.array([set(x) == set(y) for x, y in zip(ids_a.tolist(), ids_b.tolist())])
    exact = (ids_a == ids_b).all(axis=1)
    wd = float(np.abs(w_a[exact] - w_b[exact]).max()) if exact.any() else 0.0
    return {"rows": int(len(same)), "set_equal_frac": float(same.mean()), "mismatch_rows": int((~same).sum()),
            "order_equal_frac": float(exact.mean()), "max_w_diff": wd}


def router_bias(model_dir, layer):
    """The router's selection bias of `layer` (fp32 [384]), from the checkpoint's index."""
    import os

    import torch

    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    name = f"model.llm.layers.{layer}.mlp.gate.bias"
    with safe_open(os.path.join(model_dir, idx[name]), framework="pt") as f:
        return f.get_tensor(name).to(torch.float32).numpy()


def ties(ia, ib, la, lb, bias, rows_kept, tie_gap):
    """Rows whose pick sets differ: the oracle's score gap of each swap against our score error on that row.
    A swap is a tie only when the oracle's own gap is within tie_gap, a fixed bound that does not grow with our error,
    and our score error on the experts involved is large enough to explain it.
    la, lb: router logits of the rows in rows_kept (ours, oracle); returns a list of dicts."""
    pos = {int(r): j for j, r in enumerate(rows_kept)}
    sig = lambda x: 1.0 / (1.0 + np.exp(-x.astype(np.float64)))  # noqa: E731
    out = []
    for r in np.nonzero([set(x) != set(y) for x, y in zip(ia.tolist(), ib.tolist())])[0]:
        a, b = set(ia[r].tolist()), set(ib[r].tolist())
        d = {"row": int(r), "only_ours": sorted(a - b), "only_oracle": sorted(b - a)}
        if r in pos:
            so = sig(lb[pos[r]]) + bias
            sa = sig(la[pos[r]]) + bias
            # a swap needs both scores to move: at most twice our largest error on the experts involved
            err = 2 * float(np.abs(sa - so)[d["only_ours"] + d["only_oracle"]].max())
            # the oracle's score of its own lowest pick that we dropped against our extra pick's
            gap = max(float(so[e_o] - so[e_a]) for e_o in d["only_oracle"] for e_a in d["only_ours"])
            d.update({"oracle_gap": gap, "score_err": err, "tie": gap <= tie_gap and gap <= err,
                      "oracle_8th": float(np.sort(so)[-8]), "oracle_9th": float(np.sort(so)[-9])})
        out.append(d)
    return out


def kl_rows(la, lb):
    """KL(softmax(lb) || softmax(la)) per row, fp64."""
    a = la.astype(np.float64)
    b = lb.astype(np.float64)
    la_ = a - a.max(-1, keepdims=True)
    lb_ = b - b.max(-1, keepdims=True)
    la_ -= np.log(np.exp(la_).sum(-1, keepdims=True))
    lb_ -= np.log(np.exp(lb_).sum(-1, keepdims=True))
    return (np.exp(lb_) * (lb_ - la_)).sum(-1)


def line(name, s):
    if "rel" not in s:
        return f"{name:34s} differ {s['differ']} of {int(np.prod(s['shape']))}"
    t = f"{name:34s} rel {s['rel']:.3e}  max_abs {s['max_abs']:.3e}  differ {s['differ']} of {int(np.prod(s['shape']))}"
    if "ulp_hist" in s:
        t += "  ulps[0,1,2,3-4,5-8,9-16,>16] " + ",".join(str(x) for x in s["ulp_hist"])
    if s.get("nonfinite"):
        t += f"  NONFINITE {s['nonfinite']}"
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("oracle")
    ap.add_argument("--ops", default="", help="an inkling2_op_ref.py capture to compare op outputs with")
    ap.add_argument("--tol-rel", type=float, default=1e-2)
    ap.add_argument("--tol-route", type=float, default=0.0, help="allowed fraction of rows whose picks differ")
    ap.add_argument("--tol-kl", type=float, default=1e-2, help="largest allowed KL(oracle || ours) of a logits row")
    ap.add_argument("--model", default="", help="checkpoint dir: explain differing picks by the router's scores")
    ap.add_argument("--tie-gap", type=float, default=5e-3,
                    help="largest oracle score gap (sigmoid + bias) a differing pick may have and count as a tie")
    ap.add_argument("--tol-tied", type=float, default=0.02, help="allowed fraction of rows whose picks differ by ties")
    ap.add_argument("--tol-router-rel", type=float, default=1e-2, help="largest relative error of the router logits")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    dump, ora = Store(a.dump), Store(a.oracle)
    meta = dump.meta
    print(f"dump {a.dump}: {meta.get('prompt', '?')}, layers {meta.get('layers', '?')}, chunk {meta.get('chunk', '?')},"
          f" status: {meta.get('status', '?')}")
    if meta.get("fed_stubs"):
        byop = {}
        for x in meta["fed_stubs"].split(","):
            byop.setdefault(x.split(".", 1)[1], []).append(x.split(".", 1)[0])
        print("ops passed with torch's outputs (no kernel): " + ", ".join(f"{k} (x{len(v)})" for k, v in byop.items()))
    rows_all, _ = ora.get("rows")
    T = int(ora.get("ids")[0].shape[0])
    # a run that stopped early holds rows [0, rows_done) of its layers (and [0, embed_rows) of a computed embed)
    done = int(meta.get("rows_done", T))
    rows, keep = rows_all[rows_all < done], rows_all < done
    if done < T:
        print(f"comparing the {done} rows of the forwards that finished ({int(keep.sum())} of the oracle's kept rows)")
    report = {"meta": meta, "layers": {}, "ops": {}, "tol_rel": a.tol_rel, "tol_route": a.tol_route, "tol_kl": a.tol_kl}
    bad, compared = [], 0
    print(f"tolerance: rel <= {a.tol_rel:g}, route-mismatch rows <= {a.tol_route:g} of rows")
    for L in range(64):
        if not dump.has(f"h.{L}"):
            continue
        h, kind = dump.get(f"h.{L}")
        ref, _ = ora.get(f"h.{L}")
        s = stats(h[rows], ref[keep], kind)
        r = {"h": s}
        ok = s["rel"] <= a.tol_rel and not s.get("nonfinite")
        if dump.has(f"route.ids.{L}") and ora.has(f"route.ids.{L}"):
            ia, _ = dump.get(f"route.ids.{L}")
            ib, _ = ora.get(f"route.ids.{L}")
            wa, _ = dump.get(f"route.w.{L}")
            wb, _ = ora.get(f"route.w.{L}")
            r["route"] = routes(ia[:done], ib[:done], wa[:done], wb[:done])
            la, _ = dump.get(f"route.logits.{L}")
            lb, _ = ora.get(f"route.logits.{L}")
            r["route_logits"] = stats(la[rows], lb[keep], "float32")
            untied = r["route"]["mismatch_rows"]
            if a.model and r["route"]["mismatch_rows"]:
                r["route"]["swaps"] = ties(ia[:done], ib[:done], la[rows], lb[keep], router_bias(a.model, L),
                                           rows_all[keep], a.tie_gap)
                # with the router's scores at hand, a swap that is a tie within rounding is not a mismatch
                untied = sum(1 for sw in r["route"]["swaps"] if not sw.get("tie", False))
            r["route"]["untied_rows"] = untied
            tied = r["route"]["mismatch_rows"] - untied
            r["route"]["tied_rows"] = tied
            rl = r["route_logits"]
            router_ok = not rl.get("nonfinite") and rl.get("rel", 0.0) <= a.tol_router_rel
            if not router_ok:
                print(f"layer {L} router logits rel {rl.get('rel', float('nan')):.3e} past {a.tol_router_rel:g}")
            if tied > a.tol_tied * done:
                print(f"layer {L}: {tied} tied rows past {a.tol_tied:g} of {done}")
            ok = router_ok and untied <= a.tol_route * done and tied <= a.tol_tied * done
            ok_h = s["rel"] <= a.tol_rel and not s.get("nonfinite")
            if r["route"]["mismatch_rows"]:
                # the hidden rows again without the rows whose picks differ (a swapped expert moves a whole row)
                same = np.array([set(x) == set(y) for x, y in zip(ia[:done].tolist(), ib[:done].tolist())])
                sk = same[rows]
                r["h_same_routes"] = stats(h[rows][sk], ref[keep][sk], kind)
                print(line(f"layer {L} h (equal-pick rows)", r["h_same_routes"]))
                if a.model and untied == 0:
                    # every swap is a tie within rounding: the rows it moved say nothing about our arithmetic, so
                    # the distance is judged on the rows with equal picks (non-finite values count everywhere)
                    ok_h = r["h_same_routes"]["rel"] <= a.tol_rel and not s.get("nonfinite")
            ok = ok and ok_h
        r["pass"] = bool(ok)
        report["layers"][L] = r
        compared += 1
        rt = r.get("route", {})
        print(line(f"layer {L} h", s) + (f"  routes equal {rt['set_equal_frac']:.4f} ({rt['mismatch_rows']} rows differ),"
                                          f" order {rt['order_equal_frac']:.4f}, max_w_diff {rt['max_w_diff']:.2e}"
                                          if rt else "") + ("" if ok else "  FAIL"))
        for sw in rt.get("swaps", []):
            if "tie" in sw:
                print(f"    row {sw['row']}: ours {sw['only_ours']} oracle {sw['only_oracle']}: oracle gap"
                      f" {sw['oracle_gap']:.3e}, our score error {sw['score_err']:.3e}"
                      f" ({'tie within rounding' if sw['tie'] else 'NOT a tie'}); oracle 8th/9th"
                      f" {sw['oracle_8th']:.6f}/{sw['oracle_9th']:.6f}")
            else:
                print(f"    row {sw['row']}: ours {sw['only_ours']} oracle {sw['only_oracle']} (logits not kept)")
        if not ok:
            bad.append(f"layer {L}")
    if dump.has("embed") and not ora.has("embed"):
        print("embed: the reference holds no embed rows; not compared")
    if dump.has("embed") and ora.has("embed"):  # --embed: the computed layer-0 input against the oracle's
        ek = rows_all < int(meta.get("embed_rows", T))
        s = stats(dump.get("embed")[0][rows_all[ek]], ora.get("embed")[0][ek], "bf16")
        report["layers"]["embed"] = s
        compared += 1
        ok = s["rel"] <= a.tol_rel
        print(line("embed", s) + ("" if ok else "  FAIL"))
        if not ok:
            bad.append("embed")
    for name in ("final",):
        if dump.has(name):
            s = stats(dump.get(name)[0][rows], ora.get(name)[0][keep], "bf16")
            report["layers"][name] = s
            compared += 1
            ok = s["rel"] <= a.tol_rel
            print(line(name, s) + ("" if ok else "  FAIL"))
            if not ok:
                bad.append(name)
    if dump.has("logits.full") and done == T:
        la, _ = dump.get("logits.full")
        lb, _ = ora.get("logits.full")
        s = stats(la, lb, "float32")
        top = float((la.argmax(-1) == lb.argmax(-1)).mean())
        s["top1_equal_frac"] = top
        kl = kl_rows(la, lb)
        s["kl_mean"], s["kl_max"] = float(kl.mean()), float(kl.max())
        report["layers"]["logits"] = s
        compared += 1
        ok = s["rel"] <= a.tol_rel and s["kl_max"] <= a.tol_kl
        print(line("logits.full", s) + f"  top1 equal {top:.4f}  KL mean {s['kl_mean']:.3e} max {s['kl_max']:.3e}"
              + ("" if ok else "  FAIL"))
        if not ok:
            bad.append("logits")
    if a.ops:
        cap = Store(a.ops)
        stubs = set(x for x in meta.get("fed_stubs", "").split(",") if x)
        for name in sorted(dump.keys, key=op_key):
            part = op_parts(name)
            if part is None or part[2] != "out" or "@" in name:
                continue
            if f"{part[0]}.{part[1]}" in stubs or not cap.has(name):
                continue
            x, kind = dump.get(name)
            y, _ = cap.get(name)
            if x.shape != y.shape:
                print(f"{name}: shape {x.shape} against the capture's {y.shape}  FAIL")
                bad.append(name)
                continue
            s = stats(x, y, kind)
            report["ops"][name] = s
            compared += 1
            ok = s["equal_frac"] == 1.0 if "rel" not in s else (s["rel"] <= a.tol_rel and not s.get("nonfinite"))
            print(line(name, s) + ("" if ok else "  FAIL"))
            if not ok:
                bad.append(name)
    if a.ops:
        # the scaffold's state steps: what an op read from the stream's state (a conv's 3 rows, the cache's rows)
        # must be the earlier rows of the op output that fed it (the capture's, under --feed)
        src = {"conv.kv.in.state_k": "attn.project.out.k", "conv.kv.in.state_v": "attn.project.out.v",
               "conv.attn.in.state": "attn.out.out.ao", "conv.mlp.in.state": "moe.experts.out.y",
               "attn.core.in.k_ctx": "attn.heads.out.k", "attn.core.in.v_ctx": "attn.heads.out.v"}
        seams = 0
        for name in sorted(dump.keys, key=op_key):
            if "@" not in name:
                continue
            base, at = name.split("@")
            scope, rest = base[3:].split(".", 1)
            if rest not in src or not cap.has(f"op.{scope}.{src[rest]}"):
                continue
            at = int(at)
            x, _ = dump.get(name)
            y, _ = cap.get(f"op.{scope}.{src[rest]}")
            n = x.shape[0]
            if rest.startswith("attn.core"):
                R = min(int(meta.get("chunk", T)), T - at)
                want = y[at + R - n:at + R]
            else:
                want = np.zeros_like(x)
                lo = max(0, at - n)
                if at > 0:
                    want[n - (at - lo):] = y[lo:at]
            seams += 1
            if not np.array_equal(x, want):
                print(f"{name}: not the rows it should hold  FAIL")
                bad.append(name)
        if seams:
            compared += 1
            print(f"state seams: {seams} conv states and cache windows read back as the rows written"
                  + ("" if not any("@" in b for b in bad) else " (some FAIL)"))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
    if compared == 0:
        print("nothing to compare: the dump holds no layer output (see its status)")
        return 2
    if bad:
        if meta.get("status") != "complete":
            print(f"the run's status is not complete: {meta.get('status')}")
        print(f"FAIL: {len(bad)} past tolerance: {', '.join(bad[:12])}")
        return 1
    if meta.get("status") != "complete":
        print(f"REJECTED: within tolerance, but the run's status is not complete: {meta.get('status')}")
        return 3
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
