# RUN-008 Blind Review

Cold review: `containers/lcm/Containerfile`, `scripts/lcm_probe.py`,
`artifacts/report.md` (RUN-008 section), `artifacts/lcm_probe.json`.

## Findings

- **info** — `lcm_probe.py:71`: `torch.load(weights_only=False)` on a
  community-mirrored checkpoint — pickle-execution risk is real but
  bounded: runs inside `--network none` rootless container with ro
  mounts; warning is inherent to the format Meta ships.
- **info** — `lcm_probe.py`: prompt embedded as ONE SONAR concept
  (`predict([prompt])`) — LCM expects sentence-level inputs; multi-
  sentence prompts would collapse into one concept. Probe uses
  single-sentence prompts, consistent.
- **info** — `eos_vec` not provided → generator warns and caps at
  `max_gen_len` (no early EOS stop). Fine for a probe; a production
  harness would need the SONAR EOS vector (repo computes it in eval).
- **info** — `wtpsplit` installed in image but unused (kept for future
  sentence segmentation); harmless image weight.
- **verified ok** — `max_rss_gb` read via `getrusage` inside the
  container = container-resident measure; correct for the report.
- **verified ok** — `concepts_per_s` = 6 generated / gen_s; the
  decoded `seq[0]` is the re-decoded prompt (visible in log output) —
  report's "0,07 conc/s" arithmetic checks out (6/84).
- **verified ok** — `--network none` run produced real output:
  SONAR cache mount covered all assets; offline execution is genuine.
- **verified ok** — report.md mechanism table accurate: SONAR decoder
  is a full seq2seq (~GBs in the warm cache), not a projection — the
  "moved, not eliminated" claim is evidenced by the 6.5GB asset cache.

## Patterns

n/a — findings are all info-level operational notes for a probe script.

## Verdict

No blockers, no majors. Third-party code boundary respected end-to-end
(container + ro mounts + offline). Measurement is real and the
architectural conclusion (LCM moves token cost into a per-concept
decoder) is supported by the artifacts.
