#!/usr/bin/env python
"""Train the NCM level-generator or the AR baseline on the SAME data.

  python scripts/train_generator.py --model ncm --config configs/generator.yaml \
      --vqvae artifacts/tokenizer.pt --codes artifacts/codes \
      --out artifacts/gen_ncm [--warmstart] [--steps N] [--resume]

  python scripts/train_generator.py --model ar --config configs/generator.yaml \
      --bpe artifacts/bpe.json --corpus artifacts/corpus \
      --out artifacts/gen_ar [--steps N]

NCM consumes encoded multi-level codes (artifacts/codes/train.npy);
AR consumes the same segment stream (BPE ids, same context_length).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncm_swe.checkpoint import load_ckpt, save_ckpt
from ncm_swe.data.segments import build_segments
from ncm_swe.metrics import MetricLog
from ncm_swe.model.baseline_ar import ARBaseline
from ncm_swe.model.generator import LevelGenerator
from ncm_swe.model.warmstart import warmstart_blocks
from ncm_swe.tokenizer.bpe import CorpusBPE
from ncm_swe.vq.model import load_vqvae


def batches(X: torch.Tensor, batch: int, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    while True:
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - batch + 1, batch):
            yield X[perm[i:i + batch]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ncm", "ar"], required=True)
    ap.add_argument("--config", default="configs/generator.yaml")
    ap.add_argument("--vqvae", default="artifacts/tokenizer.pt")
    ap.add_argument("--codes", default="artifacts/codes")
    ap.add_argument("--corpus", default="artifacts/corpus")
    ap.add_argument("--bpe", default="artifacts/bpe.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmstart", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--intra", action="store_true",
                    help="intra-level chunked conditioning (MaskGIT-style)")
    ap.add_argument("--intra-passes", type=int, default=32,
                    help="max positions unmasked per intra pass")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["intra"] = args.intra
    cfg["intra_passes"] = args.intra_passes
    tc = cfg.get("train", {})
    steps = args.steps or tc.get("steps", 2000)
    batch = args.batch_size or tc.get("batch_size", 16)
    lr = tc.get("lr", 6e-4)
    save_every = tc.get("save_every", 250)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    if args.model == "ncm":
        vqvae, vcfg, _ = load_vqvae(args.vqvae)
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad_(False)
        model = LevelGenerator(
            vqvae, n_embd=cfg["n_embd"], n_layer=cfg["n_layer"],
            n_head_q=cfg["n_head_q"], n_head_kv=cfg["n_head_kv"],
            dropout=cfg.get("dropout", 0.0), mlp_hidden=cfg["mlp_hidden"],
            glu=cfg.get("glu", True), use_qk_norm=cfg.get("use_qk_norm", False),
            rope_base=cfg.get("rope_base", 1e6), bias=cfg.get("bias", True),
            shared_head=cfg.get("shared_head", True),
            intra=args.intra, intra_passes=args.intra_passes)
        X = torch.from_numpy(np.load(
            Path(args.codes) / "train.npy").astype(np.int64))
    else:
        bpe = CorpusBPE.load(args.bpe)
        ctx = cfg.get("context_length", 128)
        tr, _, seg_meta = build_segments(args.corpus, bpe, ctx)
        print(f"[gen] segments: {seg_meta}", flush=True)
        X = torch.from_numpy(tr)
        model = ARBaseline(
            vocab_size=bpe.vocab_size, n_embd=cfg["n_embd"],
            n_layer=cfg["n_layer"], n_head_q=cfg["n_head_q"],
            n_head_kv=cfg["n_head_kv"], dropout=cfg.get("dropout", 0.0),
            mlp_hidden=cfg["mlp_hidden"], glu=cfg.get("glu", True),
            use_qk_norm=cfg.get("use_qk_norm", False),
            rope_base=cfg.get("rope_base", 1e6), bias=cfg.get("bias", True),
            context_length=ctx)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[gen] model={args.model} params={n_params/1e6:.2f}M "
          f"data={tuple(X.shape)} steps={steps} batch={batch}", flush=True)

    if args.warmstart:
        ws = cfg.get("warmstart", {})
        rep = warmstart_blocks(model.blocks, ws["qwen_dir"],
                               ws.get("q_layers"))
        print(f"[gen] warmstart: {rep['tensors']} tensors, "
              f"{len(rep['skipped'])} skipped {rep['skipped'][:3]}",
              flush=True)
        (out / "warmstart_report.json").write_text(json.dumps(rep, indent=2))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    start_step = 0
    ckpt_path = out / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = load_ckpt(ckpt_path, model, opt)
        start_step = ck.get("step", 0)
        print(f"[gen] resumed at step {start_step}", flush=True)

    mlog = MetricLog(out / "metrics.jsonl")
    data = batches(X, batch)
    t0 = time.time()
    model.train()
    for step in range(start_step, steps):
        idx = next(data)
        if args.model == "ncm":
            loss, per = model.loss(idx)
        else:
            loss, _ = model(idx, targets=idx)
            per = {}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            mlog.log(step=step, loss=round(float(loss.detach()), 4),
                     **{k: v for k, v in per.items()})
            print(f"[gen] step {step}/{steps} loss={float(loss):.4f}",
                  flush=True)
        if (step + 1) % save_every == 0:
            save_ckpt(ckpt_path, model, opt, step + 1, cfg)

    save_ckpt(ckpt_path, model, opt, steps, cfg)
    torch.save({"model": model.state_dict(), "config": cfg,
                "model_type": args.model}, out / "final.pt")
    mlog.log(step=steps, done=1, wall_seconds=round(time.time() - t0, 1))
    print(f"[gen] done in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
