# ncm-swe — Non-Autoregressive Code Generation on Consumer CPU

**Can code generation escape the autoregressive memory-bandwidth wall on ordinary
hardware?** We measured, on a Ryzen 7 PRO 4750U (8-core Zen2, 32 GB DDR4, no GPU),
the two leading non-AR paradigms — and characterized exactly where each one breaks.

| Paradigm | Mechanism | Speed measured | Verdict |
|---|---|---|---|
| **Autoregressive** (llama.cpp, 7–8B Q4) | 1 token / forward | ~8–10 tok/s | **sovereign** (bandwidth-bound) |
| **Masked diffusion** (Dream-7B, `diffuse-cpp` Q4_K_M) | parallel unmasking, S passes | 0.3–1.7 tok/s | coherent but compute-wall: 5–8× slower |
| **Hierarchical VQ / NCM** (62M, native code VQ) | 6 level-passes / segment | **0.11 s/sample** | 40× faster than AR — but structurally incoherent |

## Headline results

- **First published quality×passes curve for masked diffusion on consumer CPU**
  (5 code tasks × S∈{8,16,24,32}, argmax, `ast.parse` + executed `pass@1`):

  | S | median wall | pass@1 |
  |---|---|---|
  | 8  | 58 s | 0/5 |
  | 16 | 58 s | **3/5** ← sweet spot |
  | 24 | 90 s | 2/5 |
  | 32 | 97 s | 1/5 |

  **Non-monotonic**: more passes = slower *and* worse (over-rendering drift).
  `sort_list` saturated (3/4); `binary_search` collapsed everywhere (0/4) —
  loop invariants are the hard case for parallel unmasking.

- **NCM reality check**: a native code VQ-VAE reaches held-out reconstruction
  CE **0.42** (channel open), and the level-renderer generates a full segment in
  **6 passes / 0.11 s** — yet produces 0/32 parseable samples **even trained to
  memorization** (loss 0.83). Discrete manifold mismatch between levels is the
  structural blocker, not data or capacity.

- **The Two Walls**: AR decode on this machine is capped by ~52 GB/s DDR4
  (measured); diffusion escapes it but hits a compute wall — each 7B dense
  forward pass costs ~4.9 s on Zen2.

## Articles

- [`articles/cpu_inference_walls.md`](articles/cpu_inference_walls.md) —
  characterization of the bandwidth and compute walls, quantization sweet spots,
  OpenMP measurements.
- [`articles/non_autoregressive_reality_check.md`](articles/non_autoregressive_reality_check.md) —
  NCM vs masked-diffusion on this hardware, with the measured curve above.

## Layout

- `src/ncm_swe/` — BPE, VQ-VAE, level/AR generators, encode/decode
- `scripts/` — training (`train_vqvae.py`, `train_generator.py`), eval
  (`_gen_probe_eval.py`, `_dream_curve.py`), Dream-7B wrapper
  (`dream_render.py`), live panel (`ncm_panel.py`, `ncm_repl.sh`)
- `data/` — code corpora (intent→unit), LoRA knowledge dump
- `docs/` — render-panel design, LLaDA/diffuse-cpp CPU dossier
- `RESEARCH_LOG.md` — full academic chronicle: every probe, number, dead end,
  ~30 annotated references. **The negative results are the point.**
- `SESSION_HANDOFF.md` — operational state for continuing agents

## Reproduce the diffusion render

```bash
git clone --recursive https://github.com/iafiscal1212/diffuse-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release diffuse-cpp && cmake --build build -j
huggingface-cli download diffuse-cpp/Dream-v0-Instruct-7B-GGUF dream-7b-q4km.gguf
python scripts/dream_render.py \
  "Write def binary_search(arr, target), index or -1. Code only." -n 160 -s 16 -t 8
```

## Status

Active research. Repo went public 2025-11; negative results are documented
rather than deleted. See `RESEARCH_LOG.md` §9–10 for what doesn't work and why.
