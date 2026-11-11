# RUN-001 Blind Review (cold pass over changed files)

Changed surface reviewed: `src/ncm_swe/**`, `scripts/{build_corpus,train_vqvae,
encode_corpus,train_generator,bench_generate,eval_samples}.py`,
`configs/*.yaml`, `tests/*`, `README.md`, `pyproject.toml`, `.gitignore`,
`vendor/` (reference copy, untouched).

## Findings

- **Minor — unsafe-pickle surface**: `torch.load(..., weights_only=False)`
  in `bench_generate.py` and `warmstart.py`. Acceptable here (loads only
  locally-produced checkpoints/models), but the pattern would be a risk if
  artifact sources ever became remote. No change required for this run.
- **Minor — manually promoted checkpoint**: `artifacts/gen_ncm/final.pt` was
  produced by re-wrapping `ckpt.pt` at step 250 (adds `steps_actual`). The
  bench loader tolerates it; provenance is documented in report.md. Not a
  code defect — a reproducibility note for `--resume` semantics.
- **Cosmetic — training-log warning**: `train_vqvae.py` logs `loss` via
  implicit float() on a grad tensor (warning in logs). Generator trainer
  already uses `loss.detach()`. Harmless; metrics unaffected.
- **OK — subprocess hygiene**: `build_corpus.py` invokes `llama-cli` with
  argv-list form, no `shell=True`, timeout-bounded.
- **OK — secret scan**: no credentials/keys in src/scripts/configs.
- **OK — mask & cache semantics**: block-diagonal level mask, causal AR
  mask, pre-GQA KV cache, tied embeddings — all covered by tests
  (15 green).

## Verdict

No blocking findings. Two minor provenance/safety notes recorded above;
everything material is test-covered and documented.

verdict: pass (2 minor, 1 cosmetic)
