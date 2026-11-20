# NCM render-panel + LoRA module dump — research brief

## Context
User wants the NCM kept as the central strategy, exposed as a "renderer" (adjustable
refinement: steps/samplers/samples/filters) plus pluggable modules (LoRA per domain:
Python, Java, math, PT-BR/EN-US text). Deliverables this pass: design + tests +
a pt-BR/en-US knowledge dump for LoRA activation, all runnable on this CPU-only box.

## Constraints
- CPU-only (Ryzen 7 4750U, ~37 GB/s measured). Official NCM TinyStories ckpt
  (12L/768d, ~85M) runs on CPU via vendored `NextConceptTransformer` — proven by
  `scripts/ncm_official_probe.py` (container `lcm-infer`, `--network none`).
- `peft` and `datasets` NOT installed in `~/.venvs/ncm`; `lcm-infer` image has
  torch-cpu 2.5.1 + omegaconf only. LoRA needs either a new image build (one-time,
  with network) or a hand-rolled adapter (~60 lines for linear layers).
- Balance is nearly exhausted: one-pass scope. No GPU, no fine-tuning of big models.
- Official ckpt quality is TinyStories-level; panel measures *controllability*,
  not SOTA quality. Be honest in docs.

## Options
- **Panel on official ckpt** (`NextConceptTransformer.generate`): already supports
  temperature/top_k/top_p/cfg_scale + l0 source (unigram/model/uniform/smooth) via
  `sd["inference"]["level0_unigram"]`. `decode_multiscales` gives per-level preview
  = the "passes" knob for free. **Chosen.**
- **freq/mu/adj anti-repetition**: exists in `scripts/eval_shard.py::pick_tokens`
  but is eval-side, not in `generate()`. Panel implements it by subclassing
  `_sample_level` — ~40 lines, no vendor edits.
- **LoRA**: `peft` in a purpose-built image (cleanest, needs one networked build)
  vs hand-rolled LoRA on `c_attn/c_proj` + `lm_head` (zero deps, more code).
  Recommend hand-rolled for this pass; peft later if modules multiply.
- **Knowledge dump**: single JSONL `{lang, domain, topic, content}` mixing
  programming (EN), logic (EN+PT), text/reasoning (EN+PT). Authored directly —
  no external scrape (license + quality control). ~100 entries this pass.

## Recommendation
`scripts/ncm_panel.py`: CLI over the official ckpt exposing renderer knobs
(l0 mode, temp, top_k, top_p, cfg, freq/mu/adj, preview levels, n samples) +
quality metrics (unicode-fraction, dup-4gram, distinct ratio, per-level CE on
held text) → JSON report. Tests cover helpers/metrics/schema without the ckpt.
`data/lora_dump/knowledge_dump.jsonl` + `scripts/dump_knowledge.py` validator.
LoRA module design documented; training is next-run work.

## Open Questions
- Does `_sample_level` receive per-position logits (B,T,V) so penalties apply
  per slot? Verify against `next_concept.py` before wiring freq/mu/adj.
- CFG needs a real null prefix (EOS) — cfg_null_eos=50256 is GPT2 vocab; must map
  to this tokenizer's pad/eos id.
- Level-0 is fixed-length (8 codes); "number of passes" = K levels only. True
  variable-depth refinement needs `predict_level0` ckpts or early-exit — document
  as future knob, not fake it.
