"""Inkling-2-Small's safetensors inventory as a small text file: every tensor's file, dtype, shape and byte range.

The Zig family's slice plan is tested against this inventory (zig/src/families/inkling2/fixtures/inventory.tsv).
It reads the shard headers from a local checkpoint folder (only the first bytes of each file are read, so a folder
whose shards are still downloading works once each header is on disk) or, with --hub, from the Hub by range request.

  python tools/zig/inkling2_inventory.py MODEL_DIR > inventory.tsv
  python tools/zig/inkling2_inventory.py --hub thinkingmachines/Inkling-2-Small --revision aac97412 > inventory.tsv

Format, one record a line, tab-separated:
  file <index> <name> <header_len>           every shard, in index order (data starts at 8 + header_len)
  t <file index> <name> <dtype> <d0,d1,..> <begin> <end>   begin and end count from the file's data start
"""

import argparse
import json
import os
import struct
import sys
import urllib.request

FILES = [f"model-{i:05d}-of-00024.safetensors" for i in range(1, 25)] + ["mtp.safetensors"]


def local(folder, name):
    with open(os.path.join(folder, name), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = f.read(n)
    if len(head) != n:
        raise SystemExit(f"{name}: header is {n} bytes, {len(head)} on disk")
    return n, head


def hub(repo, revision, name):
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"

    def get(a, b):
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    n = struct.unpack("<Q", get(0, 7))[0]
    return n, get(8, 8 + n - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?")
    p.add_argument("--hub")
    p.add_argument("--revision", default="main")
    a = p.parse_args()
    if (a.dir is None) == (a.hub is None):
        raise SystemExit("give a checkpoint folder or --hub REPO")
    out = sys.stdout
    out.write("# Inkling-2-Small safetensors inventory (tools/zig/inkling2_inventory.py)\n")
    for i, name in enumerate(FILES):
        n, head = local(a.dir, name) if a.dir else hub(a.hub, a.revision, name)
        out.write(f"file\t{i}\t{name}\t{n}\n")
        h = json.loads(head)
        rows = [(v["data_offsets"][0], k, v) for k, v in h.items() if k != "__metadata__"]
        for _, k, v in sorted(rows):
            shape = ",".join(str(d) for d in v["shape"])
            b, e = v["data_offsets"]
            out.write(f"t\t{i}\t{k}\t{v['dtype']}\t{shape}\t{b}\t{e}\n")


if __name__ == "__main__":
    main()
