# RUN-007 Blind Review

Cold review: `scripts/train_level_heads.py` (offsets extension),
`scripts/spec_pyramid.py` (new), `artifacts/report.md` (RUN-007 section),
`artifacts/spec_pyramid_bench.json`.

## Findings

- **info** — `spec_pyramid.py:71`: `c = ids.shape[1]` dead variable
  (kept from indexing rework; harmless).
- **info** — `spec_pyramid.py`: `draft_token` runs head layers under
  grad-enabled context (backbone forward is `@torch.no_grad`, heads are
  not) — autograd graph per draft call; cosmetic RSS/compute waste,
  consistent with decode_pyramid.py:59 same-class finding.
- **info** — `train_level_heads.py`: `loss, per[j] = ..., float(l_j)`
  on requires_grad tensor — scalar-conversion warning class (same as
  prior runs; works, noisy stderr).
- **minor** — `train_level_heads.py --offsets K`: no guard for
  `seq <= K` — `h[:, -1-j]` wraps silently if user sets seq < K+1.
  Default seq=64 makes it latent, but a foot-gun for reuse.
- **verified ok** — `spec_pyramid.py` verify indexing: `base=w-1-K`
  positions hidden at base+j → predicts ver index c+j; `ar[K]` reads
  w-1 (bonus); `feat = hv[0, base+m+1]` exists whenever m<K
  (base+m+1 ≤ w-1) — the two earlier off-by-ones are fixed and the
  full-accept feat=None path is required, not optional.
- **verified ok** — correctness property measured: `text_equal_ar=true`
  in smoke and all bench runs (emitted tokens are always AR argmax).
- **verified ok** — report.md numbers match
  `spec_pyramid_bench.json` (tok/fwd 0.97/0.94/0.94, accept zeros,
  AR 3.67-4.80 vs spec 0.92-0.96 tok/s).
- **verified ok** — checkpoint format change (ModuleList vs single
  LevelHeads) is confined to `--offsets >1`; old out-dirs untouched;
  K auto-detected from state dict in spec_pyramid.

## Patterns

n/a — the requires_grad→scalar warning appears in both decode scripts
but is cosmetic; no functional defect cluster.

## Verdict

No blockers, one minor (latent seq/offsets foot-gun, docs-visible in
docstring). Negative benchmark result is real and correctly attributed
(code→token reconstruction gap, not harness bug): `equal_ar=True`
proves the verify path is sound, so accept=0 reflects draft quality.
