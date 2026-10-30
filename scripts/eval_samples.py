#!/usr/bin/env python
"""Quality eval on generated samples: Python-parse rate + lexical diversity.

  python scripts/eval_samples.py --bench artifacts/bench.json --out artifacts/eval.json
"""
import argparse
import ast
import json
from pathlib import Path


def stats(samples):
    n = len(samples) or 1
    parsed = 0
    toks = []
    for s in samples:
        try:
            ast.parse(s)
            parsed += 1
        except SyntaxError:
            pass
        toks.extend(s.split())
    return {
        "n_samples": len(samples),
        "py_parse_rate": round(parsed / n, 3),
        "mean_words": round(sum(len(s.split()) for s in samples) / n, 1),
        "distinct_word_ratio": round(len(set(toks)) / max(len(toks), 1), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="artifacts/bench.json")
    ap.add_argument("--out", default="artifacts/eval.json")
    args = ap.parse_args()
    res = json.loads(Path(args.bench).read_text())
    out = {arm: stats(res[arm]["samples"]) for arm in ("ncm", "ar")
           if arm in res}
    out["note"] = ("py_parse_rate = fraction of samples that parse as "
                   "Python via ast.parse; weak but objective quality proxy "
                   "at this scale.")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
