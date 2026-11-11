# RUN-006 Blind Review

Cold review of changed files: `scripts/decode_pyramid.py`,
`scripts/chain_watcher.sh`, `artifacts/report.md` (appended),
`artifacts/bench_specdecode_thinkpad.json`, `artifacts/pyramid_bench.json`.

## Findings

- **info** — `decode_pyramid.py:59`: `float(conf)` triggers
  requires_grad→scalar warning; decode path lacks `torch.no_grad()`
  around head evals (backbone forward is decorated, heads are not) —
  builds autograd graph per position; works, wastes RSS slightly.
- **info** — `decode_pyramid.py`: `torch.save({"stats": out},
  "/tmp/_last_dec.pt")` leftover debug write to /tmp; harmless junk.
- **info** — `decode_pyramid.py`: adaptive exit uses `min(confs)` over
  all levels so far — one weak level veto blocks exit; conservative
  (documented in report as known limitation).
- **info** — `chain_watcher.sh`: no signal trap; orphaned watcher would
  poll /proc forever if its own parent dies without the watched PIDs —
  bounded by both-done exit condition; acceptable.
- **verified ok** — `chain_watcher.sh`: artifact-first check ordering
  handles already-dead PID + existing output (the RUN-005 self-match
  class of bug); `/proc/$PID` check cannot self-match.
- **verified ok** — `decode_pyramid.py`: `agreement()` runs after `dt`
  is captured — its extra O(n) backbone forwards do not contaminate
  the reported tok/s.
- **verified ok** — report.md numbers trace to artifacts:
  bench_471m ×2,27; pyramid 2,42 vs 3,65 tok/s; spec-decode 19,2 vs 18,0.

## Patterns

n/a — fewer than 3 findings share a root cause. Prototype scripts carry
cosmetic shortcuts consistent with research tooling, not regressions.

## Verdict

No blockers, no majors. Scripts are additive research tooling; existing
pipeline untouched. Reported numbers match persisted artifacts.
