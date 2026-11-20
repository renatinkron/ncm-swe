#!/usr/bin/env python3
"""T3: can the official VQ-VAE carry code tokens?

(a) recovery with NEW-code-BPE ids through frozen VQ (emb mismatch baseline)
(b) fine-tune ONLY encoder.wte on code (enc/VQ/dec frozen) -> recovery
(c) fallback: also unfreeze decoder if (b) < gate

Rows: code corpus -> new BPE ids -> 128-token rows (EOS-padded).
Metric: token recovery accuracy on non-pad positions, held-out split.
"""
import glob
import importlib
import json
import os
import pickle
import sys
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
from tokenization.bpe_tokenizer import BPETokenizer
from tokenization.vqvae_tokenizer import VQVAETokenizer

CK = "/work/artifacts/ncm_official"
OUTD = "/work/artifacts/ncm_code"
CORPUS = "/repo/data/corpus_code"
CTX = 128
N_TRAIN = int(os.environ.get("N_TRAIN", "4096"))
N_EVAL = 512
STEPS = int(os.environ.get("STEPS", "600"))
BATCH = 32
LR = 3e-4
GATE = 0.85


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
    return hits / max(tot, 1)


def main():
    global PAD_ID
    cfg = OmegaConf.load(ROOT + "/config/eval/tinystories.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(
        CK, cfg.checkpoints.tokenizer)
    cfg.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(
        CK, cfg.checkpoints.bpe_vocab)
    cfg.tokenizer.pre_tokenizer.merges_file_path = os.path.join(
        CK, cfg.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    vq = tk.vqvae

    EOS = chr(60) + "|endoftext|>"
    PAD = chr(60) + "|pad|>"
    spec = {"eos": EOS, "pad": PAD}
    vocab = pickle.load(open(OUTD + "/bpe_code_vocab.pickle", "rb"))
    merges = pickle.load(open(OUTD + "/bpe_code_merges.pickle", "rb"))
    bpe = BPETokenizer(vocab, merges, spec)
    PAD_ID, EOS_ID = bpe.pad_token_id, bpe.eos_token_id

    # build rows: concatenate shard text, chunk into ctx ids
    ids_all = []
    for f in sorted(glob.glob(CORPUS + "/shard_*.txt")):
        ids_all.extend(bpe.encode(open(f).read()))
        if len(ids_all) > (N_TRAIN + N_EVAL + 64) * CTX:
            break
    n_rows = min(len(ids_all) // CTX, N_TRAIN + N_EVAL)
    rows = torch.tensor(
        [ids_all[i * CTX:(i + 1) * CTX] for i in range(n_rows)])
    train_rows, eval_rows = rows[N_EVAL:], rows[:N_EVAL]
    print(f"rows: train={len(train_rows)} eval={len(eval_rows)}", flush=True)

    res = {"n_train": len(train_rows), "n_eval": len(eval_rows)}
    res["recovery_mismatch"] = round(recovery(vq, eval_rows), 4)
    print("mismatch recovery:", res["recovery_mismatch"], flush=True)

    # (b) emb-table fine-tune, everything else frozen
    for p in vq.parameters():
        p.requires_grad_(False)
    emb = vq.encoder.wte
    emb.weight.requires_grad_(True)
    opt = torch.optim.AdamW([emb.weight], lr=LR)
    g = torch.Generator().manual_seed(0)
    vq.train()
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
        opt.step()
        if step % 100 == 0:
            print(f"step {step} ce={ce.item():.3f} vq={vq_loss.item():.3f}",
                  flush=True)
    res["recovery_emb_tuned"] = round(recovery(vq, eval_rows), 4)
    print("emb-tuned recovery:", res["recovery_emb_tuned"], flush=True)

    # (c) fallback: unfreeze decoder too if under gate
    tuned_decoder = False
    if res["recovery_emb_tuned"] < GATE:
        tuned_decoder = True
        for p in vq.decoder.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(
            [{"params": [emb.weight], "lr": LR},
             {"params": vq.decoder.parameters(), "lr": LR}])
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
            opt.step()
            if step % 100 == 0:
                print(f"[dec] step {step} ce={ce.item():.3f}", flush=True)
        res["recovery_dec_tuned"] = round(recovery(vq, eval_rows), 4)
        print("dec-tuned recovery:", res["recovery_dec_tuned"], flush=True)

    res["decoder_tuned"] = tuned_decoder
    res["gate"] = GATE
    res["pass"] = res.get("recovery_dec_tuned",
                         res["recovery_emb_tuned"]) >= GATE
    torch.save({"wte": emb.weight.detach().cpu(),
                "decoder": vq.decoder.state_dict() if tuned_decoder else None},
               OUTD + "/vq_code_tune.pt")
    json.dump(res, open(OUTD + "/recon_metrics.json", "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
