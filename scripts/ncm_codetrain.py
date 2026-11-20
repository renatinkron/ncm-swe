#!/usr/bin/env python3
"""T4: train the NCM LevelGenerator on code-domain VQ codes.

Pipeline: code corpus -> code-BPE ids -> 128-tok rows -> tuned VQ encode
-> code rows (N,641) -> FT a COPY of gen_ncm_v2 -> sample -> decode via
tuned VQ + code BPE -> ast.parse / compile metrics.

Original checkpoints are never modified (in-memory copies only).
Long run: designed for background execution with progress logging.
"""
import ast
import glob
import importlib
import json
import os
import pickle
import sys
import time
import types

import numpy as np
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

sys.path.insert(0, "/repo/src")
from ncm_swe.model.generator import LevelGenerator

CK = "/work/artifacts/ncm_official"
OUTD = "/work/artifacts/ncm_code"
CORPUS = "/repo/data/corpus_code"
CTX = 128
N_ROWS = int(os.environ.get("N_ROWS", "20000"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BATCH = 16
LR = float(os.environ.get("LR", "1e-4"))
N_SAMPLES = 32


def build_tk(vq_tune):
    cfg = OmegaConf.load(ROOT + "/config/eval/tinystories.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(
        CK, cfg.checkpoints.tokenizer)
    cfg.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(
        CK, cfg.checkpoints.bpe_vocab)
    cfg.tokenizer.pre_tokenizer.merges_file_path = os.path.join(
        CK, cfg.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    vq = tk.vqvae.eval()
    if vq_tune is not None:
        vq.encoder.wte.weight.data.copy_(vq_tune["wte"].to(
            vq.encoder.wte.weight.dtype))
        if vq_tune.get("decoder") is not None:
            vq.decoder.load_state_dict(vq_tune["decoder"])
    for p in vq.parameters():
        p.requires_grad_(False)
    return tk, vq


def encode_corpus(fast_tok, vq, n_rows):
    lessons_file = os.environ.get("LESSONS")
    if lessons_file:
        units = [u for u in open(lessons_file).read().split("\n\n")
                 if u.strip()]
        pad = b"\x01"
        rows_l = []
        for u in units:
            ids = fast_tok.encode(u).ids[:CTX - 1] + [0]
            ids += [1] * (CTX - len(ids))
            rows_l.append(ids)
        rows = torch.tensor(rows_l[:n_rows])
        out = np.empty((len(rows), 641), dtype=np.uint16)
        with torch.no_grad():
            for i in range(0, len(rows), 64):
                out[i:i + 64] = vq.encode(rows[i:i + 64]).cpu().numpy()
        return out
    ids_all = []
    for f in sorted(glob.glob(CORPUS + "/shard_*.txt")):
        ids_all.extend(fast_tok.encode(open(f).read()).ids)
        if len(ids_all) > n_rows * CTX:
            break
    n = min(len(ids_all) // CTX, n_rows)
    rows = torch.tensor([ids_all[i * CTX:(i + 1) * CTX] for i in range(n)])
    out = np.empty((n, 641), dtype=np.uint16)
    with torch.no_grad():
        for i in range(0, n, 64):
            codes = vq.encode(rows[i:i + 64])
            out[i:i + 64] = codes.cpu().numpy().astype(np.uint16)
        if i % 640 == 0:
            print(f"encode {i}/{n}", flush=True)
    return out


def decode_codes(vq, bpe, codes_row):
    with torch.no_grad():
        _, zl = vq.vq.idx_to_zhats(
            torch.tensor(codes_row).unsqueeze(0), return_zlevels=True)
        ids = vq.decoder(zl[-1]).argmax(-1)[0].tolist()
    if 0 in ids:
        ids = ids[:ids.index(0)]
    return bpe.decode(ids, skip_special_tokens=True)


def gen_samples(gen, n, seed):
    torch.manual_seed(seed)
    gen.eval()
    idx = gen.generate(batch_size=n, temperature=0.9, top_k=32, top_p=1.0)
    return [r.tolist() for r in idx]


def score(texts):
    n_parse = n_comp = 0
    for t in texts:
        try:
            ast.parse(t)
            n_parse += 1
        except Exception:
            pass
        try:
            compile(t, "<s>", "exec")
            n_comp += 1
        except Exception:
            pass
    return n_parse, n_comp


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "14")))
    t0 = time.time()
    tune = torch.load(OUTD + "/vq_code_tune.pt", weights_only=False)
    tk, vq = build_tk(tune)
    EOS = chr(60) + "|endoftext|>"
    PAD = chr(60) + "|pad|>"
    spec = {"eos": EOS, "pad": PAD}
    bpe = BPETokenizer(pickle.load(open(OUTD + "/bpe_code_vocab.pickle", "rb")),
                       pickle.load(open(OUTD + "/bpe_code_merges.pickle", "rb")),
                       spec)
    from tokenizers import Tokenizer as HFTokenizer
    fast_tok = HFTokenizer.from_file(OUTD + "/bpe_code_tokenizer.json")

    tag = os.environ.get("TAG") or ("lessons" if os.environ.get("LESSONS") else "code")
    codes_path = OUTD + f"/codes_{tag}.npy"
    if os.path.exists(codes_path):
        codes = np.load(codes_path)[:N_ROWS]
    else:
        codes = encode_corpus(fast_tok, vq, N_ROWS)
        np.save(codes_path, codes)
    print(f"codes {codes.shape}", flush=True)

    # generator copy (checkpoint untouched)
    init = os.environ.get("INIT", "/work/artifacts/gen_ncm_v2/final.pt")
    ck = torch.load(init, map_location="cpu", weights_only=False)
    gcfg = ck.get("config", {})
    gen = LevelGenerator(vq, n_embd=gcfg.get("n_embd", 896),
                         n_layer=gcfg.get("n_layer", 4), n_head_q=14,
                         n_head_kv=2, mlp_hidden=4864,
                         intra=gcfg.get("intra", False),
                         corrupt=gcfg.get("corrupt", 0.0))
    gen.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
    res = {"n_rows": int(codes.shape[0]), "epochs": EPOCHS, "lr": LR}

    # baseline samples
    pre = [decode_codes(vq, bpe, r) for r in gen_samples(gen, N_SAMPLES, 7)]
    res["parse_pre"], res["comp_pre"] = score(pre)
    json.dump(pre, open(OUTD + "/code_samples_pre.json", "w"), indent=1)
    print("pre:", res["parse_pre"], res["comp_pre"], flush=True)

    # fine-tune on code rows
    gen.train()
    loss_fn = gen.loss
    if os.environ.get("COMPILE"):
        torch._dynamo.config.suppress_errors = True
        gen.vqvae.vq.idx_to_ncm_input = torch._dynamo.disable(
            gen.vqvae.vq.idx_to_ncm_input)
        loss_fn = torch.compile(gen.loss)
    data = torch.tensor(codes.astype(np.int64))
    opt = torch.optim.AdamW(gen.parameters(), lr=LR)
    g = torch.Generator().manual_seed(0)
    first = last = None
    n = len(data)
    step_log = int(os.environ.get("STEP_LOG", "100"))
    gen_out = OUTD + "/" + os.environ.get("GEN_OUT", "gen_code.pt")
    max_t = float(os.environ.get("MAX_T", "0"))
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        tot = nb = 0
        for i in range(0, n - BATCH + 1, BATCH):
            if max_t and time.time() - t0 > max_t:
                print(f"max_t {max_t}s reached at ep{ep} step {nb}",
                      flush=True)
                break
            b = data[perm[i:i + BATCH]]
            out = loss_fn(b)
            loss = out[0] if isinstance(out, (tuple, list)) else out
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
            if nb % step_log == 0:
                print(f"ep{ep} step {nb} loss={float(loss.detach()):.4f} "
                      f"t={time.time()-t0:.0f}s", flush=True)
        ce = tot / max(nb, 1)
        if first is None:
            first = ce
        last = ce
        print(f"ep{ep} loss={ce:.4f} t={time.time()-t0:.0f}s", flush=True)
        res["ce_first"], res["ce_last"] = round(first, 4), round(ce, 4)
        res["epoch"] = ep
        live = [decode_codes(vq, bpe, r)
                for r in gen_samples(gen, 4, ep)]
        json.dump({"epoch": ep, "ce": ce, "samples": live},
                  open(OUTD + "/live_samples.json", "w"), indent=1)
        gen.train()
        json.dump(res, open(OUTD + "/train_metrics_partial.json", "w"))
        torch.save(gen.state_dict(), gen_out + ".tmp")
        os.replace(gen_out + ".tmp", gen_out)
        if max_t and time.time() - t0 > max_t:
            break
    res["ce_first"], res["ce_last"] = round(first, 4), round(last, 4)

    post = [decode_codes(vq, bpe, r) for r in gen_samples(gen, N_SAMPLES, 7)]
    res["parse_post"], res["comp_post"] = score(post)
    res["n_samples"] = N_SAMPLES
    res["train_minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(post, open(OUTD + "/code_samples_post.json", "w"), indent=1)
    torch.save(gen.state_dict(),
               OUTD + "/" + os.environ.get("GEN_OUT", "gen_code.pt"))
    json.dump(res, open(OUTD + "/train_metrics.json", "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
