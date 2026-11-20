#!/usr/bin/env python3
"""Build a code corpus from local Python sources (no network).

Scans stdlib, venv site-packages and repo scripts for .py files, dedups by
content hash, caps per-file size, writes ~1MB text shards to data/corpus_code/.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "corpus_code")
SHARD_BYTES = 1 << 20
PER_FILE_CAP = 200_000

SOURCES = [
    "/usr/lib/python3.10",
    "/usr/lib/python3.11",
    "/usr/lib/python3.12",
    os.path.expanduser("~/.venvs/ncm/lib"),
    os.path.join(ROOT, "scripts"),
    os.path.join(ROOT, "vendor", "ncm-upstream"),
]


def iter_py(root):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def main():
    os.makedirs(OUT, exist_ok=True)
    seen = set()
    shard, shard_sz, idx = [], 0, 0
    n_files = n_dup = n_skip = 0
    for src in SOURCES:
        if not os.path.isdir(src):
            continue
        for path in iter_py(src):
            try:
                data = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                n_skip += 1
                continue
            if len(data) < 200:
                n_skip += 1
                continue
            h = hashlib.md5(data.encode()).hexdigest()
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)
            blob = data[:PER_FILE_CAP] + "\n\n"
            if shard_sz + len(blob) > SHARD_BYTES and shard:
                with open(os.path.join(OUT, f"shard_{idx:04d}.txt"), "w") as fo:
                    fo.write("".join(shard))
                idx += 1
                shard, shard_sz = [], 0
            shard.append(blob)
            shard_sz += len(blob)
            n_files += 1
    if shard:
        with open(os.path.join(OUT, f"shard_{idx:04d}.txt"), "w") as fo:
            fo.write("".join(shard))
        idx += 1
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    stats = {"files": n_files, "dupes": n_dup, "skipped": n_skip,
             "shards": idx, "bytes": total}
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
