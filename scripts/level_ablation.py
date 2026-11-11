#!/usr/bin/env python3
"""RUN-015 T001: level-truncation ablation of the MiniMSRQ VQ-VAE.

For each k in 0..K: decode held segments using ONLY levels < k
(z_hat = sum of codebook contributions of levels 0..k-1) and measure
token-level reconstruction accuracy vs ground-truth BPE tokens.
This measures how much decodable information each hierarchy depth
contributes — and what a generator that only learned the first k
levels could produce.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncm_swe.vq.model import load_vqvae            # noqa: E402
from ncm_swe.tokenizer.bpe import CorpusBPE        # noqa: E402
from ncm_swe.data.segments import build_segments  # noqa: E402


@torch.no_grad()
def z_partial(vq, idx, k):
    """z_hat using only levels 0..k-1 (k=0 -> zeros)."""
    B = idx.shape[0]
    z = torch.zeros(B, vq.L_full, vq.embed_dim)
    ofs = 0
    for i, L_i in enumerate(vq.levels):
        if i < k:
            q = vq.embeddings[i].weight[
                idx[:, ofs:ofs + L_i].reshape(-1)].view(B, L_i, vq.embed_dim)
            z = z + vq._up_to_full(q, i)
        ofs += L_i
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqvae", default="artifacts/tokenizer_ts.pt")
    ap.add_argument("--bpe", default="artifacts/bpe_ts.json")
    ap.add_argument("--corpus-dir", default="data/corpus_ts_vq")
    ap.add_argument("--shards", type=int, default=3,
                    help="use only first N shards for speed")
    ap.add_argument("--n", type=int, default=512,
                    help="max held segments to evaluate")
    ap.add_argument("--out", default="artifacts/level_ablation.json")
    args = ap.parse_args()

    model, cfg, _ = load_vqvae(args.vqvae)
    model.eval()
    vq = model.vq
    bpe = CorpusBPE.load(args.bpe)

    # held segments: deterministic SHA-1 bucket <8, same split as training
    import tempfile, shutil
    if args.shards > 0:
        src = Path(args.corpus_dir)
        tmp = Path(tempfile.mkdtemp(prefix="abl_corpus_"))
        for s in sorted(src.glob("shard-*.jsonl"))[: args.shards]:
            shutil.copy(s, tmp / s.name)
        corpus_dir = tmp
    else:
        corpus_dir = Path(args.corpus_dir)
    tr, ho, meta = build_segments(corpus_dir, bpe, cfg["context_length"])
    print(f"[abl] held segments={ho.shape} meta={meta}", flush=True)

    n = min(args.n, ho.shape[0])
    toks = torch.from_numpy(ho[:n]).long()            # (n, 128) token ids
    idx = model.encode(toks)                          # (n, 249) codes
    levels = vq.levels
    K = len(levels)

    results = {}
    for k in range(K + 1):
        z = z_partial(vq, idx, k)
        logits = model.decoder(z)
        pred = logits.argmax(-1)
        acc = float((pred == toks).float().mean())
        active = sum(levels[:k])
        results[f"k{k}_levels{levels[:k]}"] = {
            "active_code_positions": active,
            "token_accuracy": round(acc, 4)}

    # qualitative samples for a few depths
    samples = {}
    for k in range(K + 1):
        z = z_partial(vq, idx[:2], k)
        pred = model.decoder(z).argmax(-1)
        samples[f"k{k}"] = [bpe.decode(p.tolist()) for p in pred]
    results["ground_truth"] = [bpe.decode(t.tolist()) for t in toks[:2]]

    Path(args.out).write_text(json.dumps(
        {"n_segments": n, "levels": levels, **results,
         "samples": samples}, ensure_ascii=False, indent=2))
    print(f"[abl] -> {args.out}")
    for k in range(K + 1):
        key = f"k{k}_levels{levels[:k]}"
        print(f"  k={k} active={results[key]['active_code_positions']:>3} "
              f"acc={results[key]['token_accuracy']:.4f}")


if __name__ == "__main__":
    main()
