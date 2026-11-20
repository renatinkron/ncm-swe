#!/usr/bin/env python3
"""Knowledge distillation via gradient selection ("ultra-condensed" set).

Hypothesis (user's): a small subset of code-rows chosen by what the model
does NOT know transfers more knowledge per step than random rows.

Protocol (no eval leakage):
  held-out = last 500 code rows (never trained on)
  pool     = first 4500 rows
  A) score each pool row by base-model loss -> top-K with l0 dedup = CONDENSED
  B) random K rows = CONTROL
  train two copies, same steps; report held-out CE delta A vs B.
"""
import importlib
import json
import os
import sys
import time
import types

import numpy as np
import torch

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
sys.path.insert(0, "/repo/src")
os.chdir(ROOT)

from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
from ncm_swe.model.generator import LevelGenerator

CK = "/work/artifacts/ncm_official"
OUTD = "/work/artifacts/ncm_code"
K = int(os.environ.get("K", "300"))
STEPS = int(os.environ.get("STEPS", "300"))
BATCH = 16
LR = 1e-4


def load_gen(vq):
    ck = torch.load("/work/artifacts/gen_ncm_v2/final.pt",
                    map_location="cpu", weights_only=False)
    c = ck.get("config", {})
    g = LevelGenerator(vq, n_embd=c.get("n_embd", 896),
                       n_layer=c.get("n_layer", 4), n_head_q=14,
                       n_head_kv=2, mlp_hidden=4864,
                       intra=c.get("intra", False),
                       corrupt=c.get("corrupt", 0.0))
    g.load_state_dict(ck["model"], strict=True)
    return g, ck


def heldout_ce(gen, held, bs=32):
    gen.eval()
    tot, nb = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(held), bs):
            out = gen.loss(held[i:i + bs])
            loss = out[0] if isinstance(out, (tuple, list)) else out
            tot += float(loss)
            nb += 1
    return tot / max(nb, 1)


def train(gen, rows, steps, seed):
    gen.train()
    opt = torch.optim.AdamW(gen.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    n = len(rows)
    for s in range(steps):
        ix = torch.randint(0, n, (BATCH,), generator=g)
        out = gen.loss(rows[ix])
        loss = out[0] if isinstance(out, (tuple, list)) else out
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        if s % 100 == 0:
            print(f"  step {s} loss={float(loss):.3f}", flush=True)


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    t0 = time.time()
    cfg = OmegaConf.load(ROOT + "/config/eval/tinystories.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(
        CK, cfg.checkpoints.tokenizer)
    cfg.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(
        CK, cfg.checkpoints.bpe_vocab)
    cfg.tokenizer.pre_tokenizer.merges_file_path = os.path.join(
        CK, cfg.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    vq = tk.vqvae.eval()
    for p in vq.parameters():
        p.requires_grad_(False)

    codes = np.load(OUTD + "/codes_code.npy").astype(np.int64)
    held = torch.tensor(codes[-500:])
    pool = torch.tensor(codes[:-500][:2500])
    print(f"pool {len(pool)} held {len(held)}", flush=True)

    base, ck = load_gen(vq)
    base_ce = heldout_ce(base, held)
    print(f"heldout CE base: {base_ce:.4f}", flush=True)

    # per-row loss on pool (batch, no grad) -> difficulty score
    base.eval()
    scores = np.empty(len(pool))
    with torch.no_grad():
        for i in range(0, len(pool), 32):
            b = pool[i:i + 32]
            out = base.loss(b)
            loss = out[0] if isinstance(out, (tuple, list)) else out
            scores[i:i + 32] = float(loss)  # batch-mean proxy per chunk
    # rank chunks, take rows from highest-loss chunks, dedup by l0 code
    order = np.argsort(-scores)
    seen, sel = set(), []
    for ci in order:
        lo = ci * 32
        for r in range(lo, min(lo + 32, len(pool))):
            l0 = int(pool[r][0])
            if l0 in seen:
                continue
            seen.add(l0)
            sel.append(r)
            if len(sel) == K:
                break
        if len(sel) == K:
            break
    print(f"condensed: {len(sel)} rows, {len(seen)} distinct l0", flush=True)
    rng = np.random.RandomState(0)
    rand_sel = rng.choice(len(pool), K, replace=False)

    res = {"k": K, "steps": STEPS, "heldout_ce_base": round(base_ce, 4)}
    for tag, rows in [("condensed", pool[sel]), ("random", pool[rand_sel])]:
        gen, _ = load_gen(vq)
        print(f"--- train {tag} ({len(rows)} rows)", flush=True)
        train(gen, rows, STEPS, seed=1)
        ce = heldout_ce(gen, held)
        res[f"heldout_ce_{tag}"] = round(ce, 4)
        print(f"heldout CE {tag}: {ce:.4f}", flush=True)
        del gen
    res["delta_cond_minus_rand"] = round(
        res["heldout_ce_random"] - res["heldout_ce_condensed"], 4)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(OUTD + "/kdistill_metrics.json", "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
