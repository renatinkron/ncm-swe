#!/usr/bin/env python
"""Benchmark NCM vs AR generation on the same machine.

Measures per-model: wall time per sample, tokens/sec equivalent, level/step
counts, peak RSS (ru_maxrss delta via /proc), and dumps decoded samples.

  python scripts/bench_generate.py --ncm artifacts/gen_ncm/final.pt \
      --ar artifacts/gen_ar/final.pt --vqvae artifacts/vqvae/final \
      --bpe artifacts/corpus_bpe/bpe.json --n 8 --out artifacts/bench.json
"""
import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncm_swe.model.baseline_ar import ARBaseline
from ncm_swe.model.generator import LevelGenerator
from ncm_swe.tokenizer.bpe import CorpusBPE
from ncm_swe.vq.model import load_vqvae


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_gen(path, vqvae):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = LevelGenerator(
        vqvae, n_embd=cfg["n_embd"], n_layer=cfg["n_layer"],
        n_head_q=cfg["n_head_q"], n_head_kv=cfg["n_head_kv"],
        mlp_hidden=cfg["mlp_hidden"], glu=cfg.get("glu", True),
        use_qk_norm=cfg.get("use_qk_norm", False),
        rope_base=cfg.get("rope_base", 1e6), bias=cfg.get("bias", True))
    m.load_state_dict(ck["model"])
    return m.eval(), cfg


def load_ar(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = ARBaseline(vocab_size=ck["model"]["head.weight"].shape[0],
                   n_embd=cfg["n_embd"], n_layer=cfg["n_layer"],
                   n_head_q=cfg["n_head_q"], n_head_kv=cfg["n_head_kv"],
                   mlp_hidden=cfg["mlp_hidden"], glu=cfg.get("glu", True),
                   use_qk_norm=cfg.get("use_qk_norm", False),
                   rope_base=cfg.get("rope_base", 1e6),
                   bias=cfg.get("bias", True),
                   context_length=cfg.get("context_length", 128))
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    assert set(missing) <= {"head.weight"}, missing  # tied to wte at init
    return m.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncm", required=True)
    ap.add_argument("--ar", required=True)
    ap.add_argument("--vqvae", default="artifacts/tokenizer.pt")
    ap.add_argument("--bpe", default="artifacts/bpe.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--out", default="artifacts/bench.json")
    args = ap.parse_args()

    vqvae, _, _ = load_vqvae(args.vqvae)
    vqvae.eval()
    bpe = CorpusBPE.load(args.bpe)
    gen, gcfg = load_gen(args.ncm, vqvae)
    ar, acfg = load_ar(args.ar)
    ctx = acfg.get("context_length", 128)
    res = {"config": {"n": args.n, "temperature": args.temperature,
                      "top_k": args.top_k, "ctx": ctx},
           "hardware": {"cpu_count": os.cpu_count()}}

    # --- NCM: K passes, all codes per level parallel ---
    t0 = time.time()
    rss0 = rss_mb()
    idx, codes = gen.generate(batch_size=args.n, temperature=args.temperature,
                              top_k=args.top_k, return_per_level=True)
    dt_ncm = time.time() - t0
    texts_ncm = gen.decode_text(idx, bpe)
    n_tok_ncm = idx.shape[1]
    res["ncm"] = {
        "wall_s": round(dt_ncm, 3), "per_sample_s": round(dt_ncm / args.n, 3),
        "codes_per_sample": n_tok_ncm,
        "level_passes": len(codes),
        "tokens_equiv": [len(bpe.encode(t)) for t in texts_ncm],
        "peak_rss_mb": round(rss_mb(), 1),
        "params": sum(p.numel() for p in gen.parameters()),
        "samples": texts_ncm}

    # --- AR: L sequential steps w/ KV cache ---
    torch.manual_seed(0)
    prompt = torch.full((args.n, 1), bpe.bos_token_id, dtype=torch.long)
    t0 = time.time()
    out_ids = ar.generate(prompt, max_new_tokens=ctx - 1,
                          temperature=args.temperature, top_k=args.top_k)
    dt_ar = time.time() - t0
    texts_ar = [bpe.decode(r.tolist(), skip_special_tokens=True)
                for r in out_ids[:, 1:]]
    res["ar"] = {
        "wall_s": round(dt_ar, 3), "per_sample_s": round(dt_ar / args.n, 3),
        "tokens_per_sample": out_ids.shape[1] - 1,
        "steps": out_ids.shape[1] - 1,
        "peak_rss_mb": round(rss_mb(), 1),
        "params": sum(p.numel() for p in ar.parameters()),
        "samples": texts_ar}

    res["speedup_wall"] = round(dt_ar / max(dt_ncm, 1e-9), 2)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "hardware"},
                      indent=1)[:600])
    print(f"\n[bench] NCM {dt_ncm:.2f}s vs AR {dt_ar:.2f}s "
          f"(x{res['speedup_wall']}) -> {args.out}")


if __name__ == "__main__":
    main()
