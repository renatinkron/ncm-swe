# Vendor review — `vendor/ncm-upstream`

- Source: https://github.com/HazyResearch/ncm @ main (cloned 2026-10-30, depth 1), MIT license — `vendor/ncm-upstream/LICENSE` retained.
- Role: **architecture reference + reusable model definitions**. The upstream
  repo ships evaluation code only (no training scripts); model classes
  (`models/vqvae.py`, `models/quant.py`, `models/vae.py`, `models/common.py`)
  are training-ready (EMA codebooks, `vq_loss`, kmeans init, fine-drop aux).
- Review findings:
  - Imports assume repo root on `sys.path` (`from models...`, `from utils...`,
    `import hydra`, `omegaconf`). We keep files pristine and shim the path in
    `src/ncm_swe/_vendor.py` rather than editing vendored files.
  - `hydra.utils.get_class` used only for registry lookup in `vqvae.py` /
    `vqvae_tokenizer.py` — satisfied by installed `hydra-core`; no config
    framework adopted on our side (plain YAML → dict constructors).
  - No CUDA-hardcoded paths in `models/*` (`.to(device)` agnostic). Flex
    attention is optional (`use_flex_attention: false` in released configs) —
    we keep it off (not installed).
  - `evaluate.py`/`scripts/` are eval-only and import our repo-absent deps
    (rouge_score, sacrebleu) — not used; our own scripts replace them.
  - No network calls, no secrets, no executable payloads in vendored model
    files (grep-verified 2026-10-30).
- Trust posture: vendored code is treated as untrusted third-party input —
  it is never modified in place; local adaptations live in `src/ncm_swe/`.
