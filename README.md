# ncm_swe — Next-Concept Modeling on a conventional laptop

An experiment testing whether **NCM** (Next-Concept Modeling,
HazyResearch/Together) — coarse-to-fine parallel generation over multiscale
discrete codes — beats an equally-sized autoregressive baseline on ordinary
CPU hardware (i5-1235U, 23 GB RAM, no GPU).

## Architecture

```
text ──BPE(8k)──> 128-token segments ──VQ-VAE──> 8-level code pyramid
                                              [1,8,16,32,48,64,96,128]
NCM generator: transformer w/ block-diagonal attention; each level's codes
predicted in parallel, conditioned on z_hat of coarser levels (K=8 passes).
AR baseline:   same transformer core, causal mask, KV-cache decode, same
               param count, same data (L=128 sequential steps).
```

Key properties:

- Stage-1 VQ-VAE vendored from upstream (`vendor/ncm-upstream`, MIT).
- Stage-2 generator built here (`src/ncm_swe/model/`), dims matched to
  Qwen2.5-Coder-0.5B (embd 896, 14q/2kv GQA, SwiGLU 4864) so real pretrained
  weights transplant 1:1 for the warm-start arm.
- Teacher corpus: Python stdlib code + PT-BR seed/template records +
  Qwen3-0.6B (llama.cpp GGUF) + Qwen2.5-Coder-1.5B (HF) + Qwen3-Coder-30B
  (GGUF, small slice).

## Pipeline

```bash
# 1. corpus (resumable; teachers time-boxed)
python scripts/build_corpus.py --corpus-dir artifacts/corpus

# 2. train stage-1 VQ-VAE (trains BPE first, cached at artifacts/bpe.json)
python scripts/train_vqvae.py --corpus-dir artifacts/corpus --steps 3000

# 3. encode corpus -> multi-level codes (artifacts/codes/{train,held}.npy)
python scripts/encode_corpus.py

# 4. train stage-2 (two arms; --warmstart transplants Qwen weights)
python scripts/train_generator.py --model ncm --warmstart --out artifacts/gen_ncm
python scripts/train_generator.py --model ar  --out artifacts/gen_ar

# 5. benchmark
python scripts/bench_generate.py --ncm artifacts/gen_ncm/final.pt \
    --ar artifacts/gen_ar/final.pt --out artifacts/bench.json
```

## Tests

```bash
python -m pytest tests/   # vq, corpus, infra, generator, train-resume
```

## Hardware constraints observed

- PyTorch CPU-only (`cuda: False`). Full fine-tuning of any real Qwen is not
  feasible; warm-start (weight transplant into N arch) is the reuse path.
- llama.cpp `llama-cli` needs `-cnv -st` for single-shot generation.
- transformers 5.x: `apply_chat_template` returns `BatchEncoding` — call
  `model.generate(**enc)`.

See `artifacts/report.md` for measured results.
