#!/usr/bin/env python
"""Sample do gen_ncm_probe → decode via VQ+BPE → ast.parse.

  python scripts/_gen_probe_eval.py --gen artifacts/gen_ncm_probe/ckpt.pt \
      --vqvae artifacts/tokenizer_probe.pt --n 32 --out /tmp/probe_eval.json
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncm_swe.tokenizer.bpe import CorpusBPE
from ncm_swe.vq.model import load_vqvae
from bench_generate import load_gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--vqvae", default="artifacts/tokenizer_probe.pt")
    ap.add_argument("--bpe", default="artifacts/bpe.json")
    ap.add_argument("--l0-unigram", default="")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    vqvae, _, _ = load_vqvae(args.vqvae)
    vqvae.eval()
    bpe = CorpusBPE.load(args.bpe)
    gen, _ = load_gen(args.gen, vqvae)

    l0 = None
    if args.l0_unigram and Path(args.l0_unigram).exists():
        p = torch.tensor(np.load(args.l0_unigram, allow_pickle=False),
                         dtype=torch.float64)
        l0 = torch.multinomial(p.float(), args.n * gen.levels[0],
                               replacement=True).view(args.n, gen.levels[0])
    torch.manual_seed(args.seed)
    t0 = time.time()
    with torch.no_grad():
        idx, _ = gen.generate(batch_size=args.n, temperature=args.temperature,
                              top_k=args.top_k, return_per_level=True, l0=l0)
    dt = time.time() - t0

    texts = gen.decode_text(idx, bpe)
    n_parse = 0
    for t in texts:
        try:
            ast.parse(t)
            n_parse += 1
        except Exception:
            pass
    res = {"n": args.n, "parse": n_parse, "gen_s": round(dt, 1),
           "s_per_sample": round(dt / args.n, 2), "texts": texts}
    blob = json.dumps(res, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(blob)
    print(f"parse {n_parse}/{args.n} · {dt/args.n:.2f}s/amostra")
    for t in texts[:3]:
        print("───", t[:200])


if __name__ == "__main__":
    main()
