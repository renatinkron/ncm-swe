#!/usr/bin/env python3
"""Boost VQ decoder fidelity: continue emb+decoder tune from vq_code_tune.pt.

Encoder + VQ codebook stay frozen (codes_code.npy and gen_lessons2.pt depend
on the frozen code mapping). Trains on more rows, longer, evals recovery on
a held split every CKPT steps. Target: >=99% token recovery.
"""
import glob
import importlib
import json
import os
import pickle
import sys
import time
import types

import torch
import torch.nn.functional as F

hydra = types.ModuleType("hydra")


class _U:
    @staticmethod
    def get_class(p):
        m, _, n = p.rpartition(".")
        return getattr(importlib.import_module(m), n)


hydra.utils = _U
sys.modules["hydra"] = hydra
ROOT = "/repo/vendor/ncm-upstream"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
from tokenizers import Tokenizer as HFTokenizer

CK = "/work/artifacts/ncm_official"
OUTD = "/work/artifacts/ncm_code"
CORPUS = "/repo/data/corpus_code"
CTX = 128
N_ROWS = int(os.environ.get("N_ROWS", "8192"))
STEPS = int(os.environ.get("STEPS", "2500"))
BATCH = 32
LR = 1e-4
CKPT = 500
PAD_ID = 1


def recovery(vq, rows):
    vq.eval()
    hits = tot = 0
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            b = rows[i:i + 64]
            codes = vq.encode(b)
            _, zl = vq.vq.idx_to_zhats(codes, return_zlevels=True)
            rec = vq.decoder(zl[-1]).argmax(-1)
            pad = (b == PAD_ID)
            hits += ((rec == b) & ~pad).sum().item()
            tot += (~pad).sum().item()
    vq.eval()
    return hits / max(tot, 1)


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "14")))
    t0 = time.time()
    cfg = OmegaConf.load(ROOT + "/config/eval/tinystories.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(
        CK, cfg.checkpoints.tokenizer)
    cfg.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(
        CK, cfg.checkpoints.bpe_vocab)
    cfg.tokenizer.pre_tokenizer.merges_file_path = os.path.join(
        CK, cfg.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    vq = tk.vqvae
    tune = torch.load(OUTD + "/vq_code_tune.pt", weights_only=False)
    vq.encoder.wte.weight.data.copy_(
        tune["wte"].to(vq.encoder.wte.weight.dtype))
    if tune.get("decoder") is not None:
        vq.decoder.load_state_dict(tune["decoder"])
    print("resumed from vq_code_tune.pt", flush=True)

    fast = HFTokenizer.from_file(OUTD + "/bpe_code_tokenizer.json")
    ids_all = []
    for f in sorted(glob.glob(CORPUS + "/shard_*.txt")):
        ids_all.extend(fast.encode(open(f).read()).ids)
        if len(ids_all) > (N_ROWS + 512) * CTX:
            break
    n = min(len(ids_all) // CTX, N_ROWS + 512)
    rows = torch.tensor([ids_all[i * CTX:(i + 1) * CTX] for i in range(n)])
    train_rows, eval_rows = rows[512:], rows[:512]
    print(f"rows train={len(train_rows)} eval={len(eval_rows)}", flush=True)

    for p in vq.parameters():
        p.requires_grad_(False)
    emb = vq.encoder.wte
    emb.weight.requires_grad_(True)
    for p in vq.decoder.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(
        [{"params": [emb.weight], "lr": LR},
         {"params": vq.decoder.parameters(), "lr": LR}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    g = torch.Generator().manual_seed(0)
    vq.eval()  # eval-mode: dropout off; grads still flow to unfrozen params
    hist = []
    best = recovery(vq, eval_rows)
    print(f"resume recovery: {best:.4f}", flush=True)
    for step in range(STEPS):
        ix = torch.randint(0, len(train_rows), (BATCH,), generator=g)
        b = train_rows[ix]
        logits, vq_loss, *_ = vq(b)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), b.reshape(-1),
            ignore_index=PAD_ID)
        loss = ce + 0.25 * vq_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(vq.decoder.parameters()) + [emb.weight], 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step} ce={ce.item():.4f}", flush=True)
        if (step + 1) % CKPT == 0:
            rec = recovery(vq, eval_rows)
            hist.append({"step": step + 1, "rec": round(rec, 4)})
            print(f"  >> recovery@{step+1}: {rec:.4f} (best {best:.4f})",
                  flush=True)
            if rec > best:
                best = rec
                torch.save({"wte": emb.weight.detach().cpu(),
                            "decoder": vq.decoder.state_dict()},
                           OUTD + "/vq_code_tune.pt")
            json.dump(hist, open(OUTD + "/dec_boost_hist.json", "w"))

    res = {"steps": STEPS, "hist": hist,
           "final_rec": hist[-1]["rec"] if hist else None,
           "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(res, open(OUTD + "/dec_boost_metrics.json", "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
