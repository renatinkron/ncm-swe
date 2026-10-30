#!/usr/bin/env python3
"""Encode the corpus into VQ code indices with the frozen tokenizer.

Output: artifacts/codes/{train,held}.npy (N × total_codes uint16) +
        artifacts/codes/meta.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ncm_swe.data.segments import build_segments
from ncm_swe.tokenizer.bpe import CorpusBPE
from ncm_swe.vq.model import load_vqvae


def encode(model, segs: np.ndarray, bs: int) -> np.ndarray:
    out = np.zeros((len(segs), sum(model.vq.levels)), dtype=np.int32)
    with torch.no_grad():
        for i in range(0, len(segs), bs):
            x = torch.from_numpy(segs[i:i + bs])
            idx = model.encode(x)
            out[i:i + bs] = idx.numpy()
    return out.astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(REPO / "artifacts/tokenizer.pt"))
    ap.add_argument("--bpe", default=str(REPO / "artifacts/bpe.json"))
    ap.add_argument("--corpus-dir", default=str(REPO / "artifacts/corpus"))
    ap.add_argument("--out-dir", default=str(REPO / "artifacts/codes"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    model, cfg, meta = load_vqvae(args.tokenizer)
    model.eval()
    bpe = CorpusBPE.load(args.bpe)

    tr, ho, seg_meta = build_segments(
        args.corpus_dir, bpe, int(cfg["context_length"]))
    if args.max_samples:
        tr, ho = tr[:args.max_samples], ho[: max(1, args.max_samples // 8)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr_codes = encode(model, tr, args.batch_size)
    ho_codes = encode(model, ho, args.batch_size)
    np.save(out_dir / "train.npy", tr_codes)
    np.save(out_dir / "held.npy", ho_codes)
    (out_dir / "meta.json").write_text(json.dumps({
        "levels": list(model.vq.levels),
        "vqvae_vocab_size": list(model.vq.vqvae_vocab_sizes),
        "context_length": cfg["context_length"],
        "train_shape": list(tr_codes.shape), "held_shape": list(ho_codes.shape),
        "segments": seg_meta,
        "wall_s": round(time.time() - t0, 1)}, indent=2))
    print(f"[codes] train={tr_codes.shape} held={ho_codes.shape} "
          f"in {time.time()-t0:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
