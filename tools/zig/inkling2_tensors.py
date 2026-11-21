"""Inkling-2-Small's safetensors inventory: every tensor's name, dtype, shape, shard and absolute byte range.

USAGE = python tools/zig/inkling2_tensors.py MODEL_DIR --out tensors.json [--hf-tree tree.json]

Reads only the header at the start of each shard (`model-*.safetensors` and `mtp.safetensors`), so it also works on a
download in progress: with --hf-tree (the Hub's `api/models/<repo>/tree/<rev>?recursive=true` listing, which carries each
LFS file's sha256) a shard that is still an `.incomplete` file under MODEL_DIR/.cache/huggingface/download is found by its
sha256 and its header read from there when those bytes have arrived. Shards whose header cannot be read yet are listed
under "missing".

Output (JSON):
  files:   {shard: {"header_bytes": 8 + header length, "size": bytes on the Hub or on disk, "complete": bool}}
  tensors: [{"name", "dtype", "shape", "shard", "offset", "nbytes"}], sorted by name; `offset` is the absolute byte
           offset of the tensor's data in its shard file (8 + header length + data_offsets[0])
  missing: shards whose header was not readable
"""

import argparse
import glob
import json
import os
import struct
import sys


def read_header(path: str):
    """(header_bytes, tensors) from a safetensors file, or None when the header bytes are not there (yet)."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return None
            n = struct.unpack("<Q", raw)[0]
            if n == 0 or n > 100 << 20:
                return None
            body = f.read(n)
            if len(body) < n:
                return None
        h = json.loads(body)
    except (OSError, ValueError):
        return None
    h.pop("__metadata__", None)
    return 8 + n, h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-tree", help="Hub tree listing (JSON) to find in-progress shards by their sha256")
    a = ap.parse_args()

    expected = {}  # shard -> (size, sha256)
    if a.hf_tree:
        for f in json.load(open(a.hf_tree)):
            p = f.get("path", "")
            if p.endswith(".safetensors"):
                expected[p] = (f.get("size"), (f.get("lfs") or {}).get("oid"))
    for p in glob.glob(os.path.join(a.model_dir, "*.safetensors")):
        expected.setdefault(os.path.basename(p), (os.path.getsize(p), None))

    partial = {}
    for p in glob.glob(os.path.join(a.model_dir, ".cache", "huggingface", "download", "*.incomplete")):
        parts = os.path.basename(p).split(".")
        if len(parts) >= 3:
            partial[parts[1]] = p

    files, tensors, missing = {}, [], []
    for shard in sorted(expected):
        size, sha = expected[shard]
        final = os.path.join(a.model_dir, shard)
        complete = os.path.exists(final) and (size is None or os.path.getsize(final) == size)
        path = final if os.path.exists(final) else partial.get(sha or "")
        got = read_header(path) if path else None
        if got is None:
            missing.append(shard)
            continue
        hb, h = got
        files[shard] = {"header_bytes": hb, "size": size, "complete": complete}
        for name, info in h.items():
            s, e = info["data_offsets"]
            tensors.append({"name": name, "dtype": info["dtype"], "shape": info["shape"], "shard": shard,
                            "offset": hb + s, "nbytes": e - s})
    tensors.sort(key=lambda t: (t["shard"], t["name"]))
    with open(a.out, "w") as f:
        json.dump({"files": files, "tensors": tensors, "missing": missing}, f, indent=1)
    print(f"{len(tensors)} tensors from {len(files)} shards; missing headers: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
