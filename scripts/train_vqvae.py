#!/usr/bin/env python3
"""Train the NCM stage-1 VQ-VAE on the corpus (CPU).

Pipeline: corpus → corpus BPE (train once, cached) → 128-id segments →
VQ-VAE (recon CE + commitment loss). Resumable via atomic checkpoints.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ncm_swe.checkpoint import load_ckpt, save_ckpt
from ncm_swe.data.segments import build_segments
from ncm_swe.metrics import MetricLog
from ncm_swe.tokenizer.bpe import CorpusBPE
from ncm_swe.vq.model import build_vqvae, load_config, save_vqvae


def get_bpe(args, texts):
    bpe_path = Path(args.bpe_path)
    if bpe_path.exists():
        return CorpusBPE.load(bpe_path)
    bpe = CorpusBPE.train(texts, vocab_size=args.bpe_vocab, out_path=bpe_path)
    print(f"[bpe] trained vocab={bpe.vocab_size} -> {bpe_path}")
    return bpe


def batches(segs: np.ndarray, bs: int, rng: np.random.Generator):
    order = rng.permutation(len(segs))
    for i in range(0, len(order) - bs + 1, bs):
        yield torch.from_numpy(segs[order[i:i + bs]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/vqvae.yaml"))
    ap.add_argument("--corpus-dir", default=str(REPO / "artifacts/corpus"))
    ap.add_argument("--bpe-path", default=str(REPO / "artifacts/bpe.json"))
    ap.add_argument("--bpe-vocab", type=int, default=8192)
    ap.add_argument("--out", default=str(REPO / "artifacts/tokenizer.pt"))
    ap.add_argument("--ckpt", default=str(REPO / "artifacts/checkpoints/vqvae.pt"))
    ap.add_argument("--metrics", default=str(REPO / "artifacts/metrics/vqvae.jsonl"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="limit segments (smoke)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    from ncm_swe.tokenizer.bpe import corpus_texts
    texts = list(corpus_texts(args.corpus_dir))
    print(f"[data] {len(texts)} records")
    bpe = get_bpe(args, iter(texts))

    cfg = load_config(args.config)
    cfg["in_vocab_size"] = bpe.vocab_size
    tr, ho, meta = build_segments(args.corpus_dir, bpe,
                                int(cfg["context_length"]))
    if args.max_samples:
        tr, ho = tr[:args.max_samples], ho[: max(1, args.max_samples // 8)]
    print(f"[data] segments train={tr.shape} held={ho.shape} meta={meta}")

    model = build_vqvae(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                          betas=(0.9, 0.95), weight_decay=0.01)
    mlog = MetricLog(args.metrics)

    start_step = 0
    if Path(args.ckpt).exists():
        ck = load_ckpt(args.ckpt, model, opt)
        start_step = int(ck["step"])
        print(f"[resume] from step {start_step}")

    rng = np.random.default_rng(args.seed + start_step)
    model.train()
    t0 = time.time()
    step = start_step
    done = False
    while step < args.steps and not done:
        for x in batches(tr, args.batch_size, rng):
            logits, vq_loss, per_lvl, gutil, _ = model(x)
            recon = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), x.reshape(-1),
                ignore_index=bpe.pad_token_id)
            loss = recon + vq_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % 50 == 0:
                mlog.log(step=step, recon_ce=round(float(recon), 4),
                         vq_loss=round(float(vq_loss), 4),
                         global_util=round(gutil, 4),
                         per_lvl_util=[round(u, 3) for u in per_lvl],
                         tok_s=int(step * np.prod(x.shape)
                                   / (time.time() - t0)))
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    hx = torch.from_numpy(ho[: min(64, len(ho))])
                    hl, hv, hp, hg, _ = model(hx)
                    hce = F.cross_entropy(
                        hl.reshape(-1, hl.shape[-1]), hx.reshape(-1),
                        ignore_index=bpe.pad_token_id)
                model.train()
                mlog.log(step=step, eval="heldout", held_ce=round(float(hce), 4),
                         held_vq=round(float(hv), 4),
                         held_util=round(hg, 4))
            if step % args.save_every == 0 or step == args.steps:
                save_ckpt(args.ckpt, model, opt, step, cfg,
                          {"bpe": args.bpe_path, "meta": meta})
            if step >= args.steps:
                done = True
                break

    save_vqvae(model, args.out, cfg,
               {"bpe": args.bpe_path, "meta": meta, "final": True,
                "steps": step})
    print(f"[done] steps={step} wall={time.time()-t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
