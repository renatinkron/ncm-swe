# RUN-023 Blind Review

Reviewer: self-review (cold pass over `git show HEAD`); independent subagent
unavailable (model rate limit at dispatch time). Scope: generator.py,
sample_official.py, train_gen_official.py, bench_generate.py, configs.

## Findings

**minor-1** `generator.py::generate` — `self.last_passes = passes + 1` counts a
level-0 pass even when `l0` is injected externally (no model pass occurs).
Reported `passes: 12` in unigram/uniform JSONs actually means 11 forward
passes. Fix: `+ (1 if l0 is None else 0)`.

**minor-2** `sample_official.py:~171` — `model.corrupt = 0.0` is set for the
exposure probe and never restored. Harmless in this one-shot script (process
exits after JSON dump) but would mislead if the function were ever imported.
Fix: save/restore `prev = model.corrupt`.

**minor-3** `generator.py::_corrupt_codes` — `rates = torch.rand(B,1)*corrupt`
is drawn unconditionally; unused in `linear` mode but still consumes RNG
state, so flat/linear streams diverge at equal seeds and one extra tensor is
allocated per step. Cosmetic. Fix: move inside the `else` branch.

**minor-4** `sample_official.py` exposure probe — `ce_self_gen` includes the
l0 CE term even under `--l0 unigram|uniform`, where l0 was externally seeded,
not self-generated. The label is slightly misleading (fine levels are the
self-generated part); totals remain comparable because the GT side measures
real l0 the same way. Fix: document or exclude l0 from the self-gen total.

**minor-5** `generator.py::generate` — l0 range/shape checked via `assert`,
stripped under `python -O`. Acceptable for scripts; consider `raise` if the
API ever becomes library-facing.

**minor-6** `_corrupt_codes` flat path — `torch.rand(idx.shape)` allocates a
full-width mask then slices `[:, ofs:ofs+L]` per level; wasteful but correct.

## Observations (non-findings)
- Exposure probe compares 8 self-gen stacks vs 8 random held stacks —
  intended design (own-codes vs corpus-codes CE), seeds fixed via
  `torch.manual_seed` + `np.random.seed`.
- Linear schedule math verified: j=1 → hi, j=nlv → lo, level 0 skipped,
  targets clean. Matches official (0.75→0.25) semantics.
- `corrupt` gate: `forward` applies corruption only when `self.corrupt > 0`;
  trainer default `--corrupt 1.0` keeps linear active. Consistent.
- intra/`boundaries` code (earlier work in same commit): `clamp_min(1)`
  guards the all-decided case; `cur/decided` slicing in the intra loop is
  correct for partial chunks.
- l0 assert shape/range validation present; `vocab_sizes[0]` bound correct.
- Backward-compat: `l0=None` path is the original code verbatim — verified
  bit-exact by `bench_l0model_sanity.json` vs `bench_v2.json` at same seed.

## Verdict

PASS — 0 blockers, 0 majors, 6 minors. All minors are metric-labeling or
cosmetic; none alter the published numbers except `passes: 12→11` in the
l0-external JSONs, which the paper does not cite.
