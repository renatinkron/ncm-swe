#!/usr/bin/env python3
"""RUN-038: NCM as student of the same knowledge pack (cross-arch proof).

pack_regex cards -> official BPE -> VQ encode -> code rows -> light FT of
LevelGenerator (in-memory copy; checkpoint never touched) -> measure whether
generated segments gain regex content vs pre-FT baseline. Container run.
"""
import importlib
import json
import os
import re
import sys
import time
import types
from pathlib import Path

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

CK = os.environ.get("CK", "/work/artifacts/ncm_official")
GEN = os.environ.get("GEN", "/work/artifacts/gen_ncm_v2/final.pt")
PACK = os.environ.get("PACK", "/work/artifacts/domcart/pack_regex.json")
OUT = Path(os.environ.get("OUT", "/repo/artifacts/ncm_packtrain"))
CTX = 128
N_SAMPLES = int(os.environ.get("N", 16))
EPOCHS = int(os.environ.get("EPOCHS", 30))
LR = float(os.environ.get("LR", 1e-4))
SEED = int(os.environ.get("SEED", 0))

REGEX_INDICATORS = ["regex", "re.", "\\d", "\\w", "\\s", "[", "^", "$",
                    "{1,", "{2,", "*", "+", "?"]
PAT_RE = re.compile(r"r\"([^\"\n]+)\"|r'([^'\n]+)'|re\.compile\(\s*[\"']([^\"']+)")


def pick_tokens(logits):
    return logits.argmax(-1)


def regex_score(text):
    hits = sum(1 for ind in REGEX_INDICATORS if ind in text)
    pat = None
    m = PAT_RE.search(text)
    if m:
        pat = next((g for g in m.groups() if g), None)
    comp = False
    if pat:
        try:
            re.compile(pat)
            comp = True
        except re.error:
            pass
    return hits, comp


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    OUT.mkdir(parents=True, exist_ok=True)
    CFG = OmegaConf.load("/repo/vendor/ncm-upstream/config/eval/tinystories.yaml")
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
    CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
    CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    vqvae = tk.vqvae.eval()
    EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id

    ck = torch.load(GEN, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = LevelGenerator(vqvae, n_embd=cfg.get("n_embd", 896),
                           n_layer=cfg.get("n_layer", 4), n_head_q=14,
                           n_head_kv=2, mlp_hidden=4864,
                           intra=cfg.get("intra", False),
                           corrupt=cfg.get("corrupt", 0.0))
    model.load_state_dict(ck["model"], strict=True)

    # 1) pack -> codes
    cards = json.loads(Path(PACK).read_text())["cards"]
    pad_row = [PAD] * CTX
    segs = []
    for c in cards:
        ids = tk.pre_tokenizer.encode(c["probe"] + "\n" + c["ideal_answer"])[:CTX]
        segs.append(ids + pad_row[len(ids):])
    with torch.no_grad():
        codes = vqvae.encode(torch.tensor(segs, dtype=torch.long))
    codes = codes.long()
    np.save(OUT / "codes.npy", codes.numpy().astype(np.uint16))
    print(f"pack codes: {codes.shape}", flush=True)

    def sample_and_score(tag):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model.eval()
        idx = model.generate(batch_size=N_SAMPLES, temperature=0.9,
                             top_k=32, top_p=1.0)
        with torch.no_grad():
            _, zl = vqvae.vq.idx_to_zhats(idx, return_zlevels=True)
            ids = pick_tokens(vqvae.decoder(zl[-1]))
        texts, hits, comps = [], [], 0
        for row in ids.tolist():
            if EOS in row:
                row = row[:row.index(EOS)]
            t = tk.pre_tokenizer.decode(
                [x for x in row if x not in (EOS, PAD)],
                skip_special_tokens=True)
            h, c = regex_score(t)
            texts.append(t)
            hits.append(h)
            comps += c
        (OUT / f"samples_{tag}.txt").write_text("\n---\n".join(texts))
        return {"mean_hits": round(sum(hits) / len(hits), 2),
                "compilable": comps, "n": len(texts)}

    # 2) baseline
    print("baseline sampling...", flush=True)
    pre = sample_and_score("pre")
    print(f"PRE {pre}", flush=True)

    # 3) light FT on pack codes
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ce_first = ce_last = None
    for ep in range(EPOCHS):
        out = model.loss(codes)
        total = out[0] if isinstance(out, (tuple, list)) else out
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        v = float(total.detach())
        if ep == 0:
            ce_first = v
        ce_last = v
        if ep % 10 == 0:
            print(f"ep{ep} loss={v:.4f}", flush=True)

    # 4) post-FT
    print("post-FT sampling...", flush=True)
    post = sample_and_score("post")
    print(f"POST {post}", flush=True)

    rec = {"n_cards": len(cards), "epochs": EPOCHS, "lr": LR,
           "train_ce_first": round(ce_first, 4), "train_ce_last": round(ce_last, 4),
           "regex_score_pre": pre["mean_hits"], "regex_score_post": post["mean_hits"],
           "compilable_pre": pre["compilable"], "compilable_post": post["compilable"],
           "n_samples": pre["n"]}
    (OUT / "metrics.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
