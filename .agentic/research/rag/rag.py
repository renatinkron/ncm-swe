#!/usr/bin/env python3
"""BM25 retriever over .agentic/research/rag/papers/*.txt + project docs.
Usage: python rag.py "query" [topk] — chunks ~150 words, BM25 k1=1.5 b=0.75."""
import glob, math, re, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
FILES = (glob.glob(f"{BASE}/papers/*.txt")
         + [f"{BASE}/../../../artifacts/report.md",
            f"{BASE}/../../../vendor/ncm-upstream/README.md",
            f"{BASE}/../../../vendor/REVIEW.md"])

STOP = set("the a an and or of to in is are was were for on with as at by that this it be from we our you your".split())

def chunks(path, w=150):
    try:
        words = open(path, encoding="utf8", errors="ignore").read().split()
    except OSError:
        return
    for i in range(0, max(len(words) - 60, 1), w):
        yield os.path.basename(path), " ".join(words[i:i + w])

def tok(s):
    return [t for t in re.findall(r"[a-z0-9_]{3,}", s.lower()) if t not in STOP]

docs, names = [], []
for f in FILES:
    for name, c in chunks(f) or []:
        docs.append(tok(c)); names.append(f"{name}#{len(docs)}")

N = len(docs)
df = {}
for d in docs:
    for t in set(d):
        df[t] = df.get(t, 0) + 1
avg = sum(map(len, docs)) / max(N, 1)

def score(q, d):
    return sum((df.get(t, 0) and math.log((N - df[t] + .5) / (df[t] + .5) + 1))
               * (d.count(t) * 2.5) / (d.count(t) + 1.5 * (1 - .75 + .75 * len(d) / avg))
               for t in q)

if __name__ == "__main__":
    q = tok(sys.argv[1]); k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    ranked = sorted(range(N), key=lambda i: -score(q, docs[i]))[:k]
    print(f"[{N} chunks indexed]")
    for i in ranked:
        s = score(q, docs[i])
        print(f"\n=== {names[i]} (score {s:.2f}) ===")
        # reprint the source chunk
        src = glob.glob(f"{BASE}/papers/*.txt") + FILES[-3:]
        for f in src:
            ws = open(f, encoding="utf8", errors="ignore").read().split()
            for j in range(0, max(len(ws) - 60, 1), 150):
                pass
        # simpler: cache chunks again
    # second pass to print chunk text
    allc = []
    for f in FILES:
        for name, c in chunks(f) or []:
            allc.append((name, c))
    for i in ranked:
        print(f"\n### {allc[i][0]}\n{allc[i][1][:600]}")
