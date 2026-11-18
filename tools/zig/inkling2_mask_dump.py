#!/usr/bin/env python3
"""inkling2_mask_dump.py DUMP [ROWS_NPY IDS_JSON] [--at POS,...] (numpy only).
Reads TF_INK2_MASK_DUMP records (cuda_lanes.zig dumpRow: u32 magic "TFMD", version, position u64, prompt length u64,
device argmax, pick, width U, mask words W, flags (1 refused argmax, 2 sampled, 4 first token, 8 late window), 0; W mask
words; U fp32 logits) and checks, for every greedy masked draw, that the pick is the first largest logit among the
allowed tokens of the row the draw read, and that the device argmax is the first largest of the whole row (the row
fetched is the head's). With ROWS_NPY/IDS_JSON (the Python checkpoint oracle's fp32 rows of the same
server reply, teacher-forced): per record the engine-vs-Python row difference, both masked argmaxes and their gaps.
--at: print every allowed token's engine and Python logit at these reply positions (j = position - prompt length)."""
import json, struct, sys
import numpy as np

HEAD = struct.Struct("<12I")


def records(path):
    data = open(path, "rb").read()
    at = 0
    while at < len(data):
        h = HEAD.unpack_from(data, at)
        assert h[0] == 0x444D4654 and h[1] == 1, f"bad record at {at}"
        pos = h[2] | h[3] << 32
        plen = h[4] | h[5] << 32
        dev, pick, U, W, flags = h[6], h[7], h[8], h[9], h[10]
        at += HEAD.size
        mask = np.frombuffer(data, dtype="<u4", count=W, offset=at)
        at += 4 * W
        row = np.frombuffer(data, dtype="<f4", count=U, offset=at)
        at += 4 * U
        yield dict(pos=pos, plen=plen, dev=None if dev == 0xFFFFFFFF else dev, pick=pick, flags=flags, mask=mask, row=row)


def allowed(mask, U):
    bits = ((mask[:, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(bool).reshape(-1)
    return np.nonzero(bits[:U])[0]


def masked_argmax(row, ids):
    v = row[ids]
    return int(ids[int(np.argmax(v))])  # np.argmax: the first largest, ids ascending


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    at = []
    if "--at" in sys.argv:
        at = [int(x) for x in sys.argv[sys.argv.index("--at") + 1].split(",")]
        a = [x for x in a if x != sys.argv[sys.argv.index("--at") + 1]]
    recs = list(records(a[0]))
    # the committed draw at a position is its last one (a draft row past a window's accepted path is drawn again)
    last = {}
    for i, r in enumerate(recs):
        last[(r["plen"], r["pos"])] = i
    py = ids = None
    if len(a) >= 3:
        py = np.load(a[1]); ids = json.load(open(a[2]))
    n = greedy = pick_ok = dev_ok = devn = refused = lowest = 0
    bad = []
    diffs = []; disagree = []
    for i, r in enumerate(recs):
        n += 1
        final = last[(r["plen"], r["pos"])] == i
        U = len(r["row"]); al = allowed(r["mask"], U)
        if r["flags"] & 1:
            refused += 1
        if r["dev"] is not None:
            devn += 1
            dev_ok += int(np.argmax(r["row"])) == r["dev"]
        if r["flags"] & 2:
            continue
        greedy += 1
        host = masked_argmax(r["row"], al)
        pick_ok += host == r["pick"]
        lowest += r["pick"] == int(al[0]) and len(al) > 1
        if host != r["pick"]:
            bad.append((r["pos"], r["pick"], host))
        if py is not None and final:
            j = r["pos"] - r["plen"]
            if 0 <= j < len(py):
                p = py[j]
                if j < len(ids["reply"]) and ids["reply"][j] != r["pick"]:
                    print(f"  note: position {r['pos']} (j {j}): engine pick {r['pick']} but the Python file's reply has {ids['reply'][j]}")
                d = np.abs(p - r["row"]); top = np.argsort(-p)[:5]
                pm = masked_argmax(p, al)
                diffs.append((j, float(d.max()), float(d[top].max())))
                if pm != r["pick"]:
                    disagree.append(dict(j=j, pos=r["pos"], allowed=len(al), engine_pick=r["pick"], python_pick=pm,
                                         engine_gap=float(r["row"][r["pick"]] - r["row"][pm]),
                                         python_gap=float(p[pm] - p[r["pick"]]),
                                         engine=[float(r["row"][r["pick"]]), float(r["row"][pm])],
                                         python=[float(p[r["pick"]]), float(p[pm])], refused=bool(r["flags"] & 1)))
        if final and (r["pos"] - r["plen"]) in at:
            j = r["pos"] - r["plen"]
            print(f"-- position {r['pos']} (reply {j}): pick {r['pick']}, device argmax {r['dev']} (row max {r['row'].max():.4f} at {int(np.argmax(r['row']))}), flags {r['flags']}, {len(al)} allowed")
            for t in al[:64]:
                extra = f"  python {py[j][t]:.4f}" if py is not None and 0 <= j < len(py) else ""
                print(f"   allowed {int(t):6d}  engine {r['row'][t]:.4f}{extra}")
            if py is not None and 0 <= j < len(py):
                te = np.argsort(-r["row"])[:8]; tp = np.argsort(-py[j])[:8]
                print("   engine top8 " + ", ".join(f"{int(t)}:{r['row'][t]:.3f}" for t in te))
                print("   python top8 " + ", ".join(f"{int(t)}:{py[j][t]:.3f}" for t in tp))
                print(f"   row |engine - python| max {np.abs(py[j] - r['row']).max():.4f}, mean {np.abs(py[j] - r['row']).mean():.4f}")
    print(f"MASKDUMP {a[0]}: {n} masked draws, {greedy} greedy; pick = first largest allowed logit of the row read: "
          f"{pick_ok}/{greedy}; device argmax = row argmax: {dev_ok}/{devn}; refused-argmax rows {refused}; "
          f"picks that are the lowest allowed id (of 2+): {lowest}")
    for b in bad:
        print(f"  MISMATCH position {b[0]}: pick {b[1]}, row's masked argmax {b[2]}")
    if py is not None and diffs:
        m = np.array([x[1] for x in diffs]); t = np.array([x[2] for x in diffs])
        print(f"vs Python fp32 rows (each position's committed draw): {len(diffs)} rows; max |engine - python| over a row: median {np.median(m):.3f}, max {m.max():.3f}; "
              f"over the Python top-5: median {np.median(t):.3f}, max {t.max():.3f}; committed masked picks that differ: {len(disagree)}")
        for d in disagree:
            print("  DIFFER " + json.dumps(d))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
