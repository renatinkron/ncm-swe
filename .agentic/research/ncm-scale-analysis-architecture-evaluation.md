# Architecture Evaluation — scaling the NCM experiment on CPU

## Decision under evaluation

Scale generator from ~60M → ~300M (lm1b-medium-class) on the same CPU
pipeline; optionally bigger VQ-VAE. Billions ruled out (RAM).

## State

- Model 300M×4B ≈ 1.2GB weights + Adam (2×fp32 copies ≈ 2.4GB) +
  activations(seq512×batch8, 24 layers) ≈ 2-4GB → **~6-8GB peak** — fits
  23GB with headroom for OS + data. 725M pushes ~16-18GB — marginal.
- Continuous trainer/ckpt machinery unchanged (state dict format identical).

## Flow

Config-only change: `configs/generator.yaml` dims (24×1024, 16q/4kv GQA,
intermediate ~5504 SwiGLU) → fresh `gen_ncm_300m`/`gen_ar_300m` runs.
Warm-start: Qwen-0.5B dims no longer match — either (a) skip warm-start,
or (b) partial transplant (first N layers); (a) is honest and simpler.
Data: corpus 6.2MB — **must grow in parallel** (teachers continuous) or
300M memorizes faster than 60M, adding no real quality.

## Computation

- ~50s/step at 300M (measured scaling ~linear from 13s/step@60M).
- 2000 steps ≈ 28h — inside the days-regime envelope.
- Throughput-at-inference also drops ~4× — NCM's structural advantage
  (6 passes vs ~127) is what keeps generation viable at bigger scale:
  this is *exactly* where NCM's premise pays off relative to AR.

## Failure analysis

- RAM exhaustion → OOM-kill mid-run: mitigate with batch 4 + seq 512 +
  checkpoint every 100 steps (resume covers).
- Overfit-on-6MB: mitigate by continuing corpus growth; evaluate by
  heldout loss, not just train loss (already logged per run).
- Silent divergent run for days → metric logging + periodic bench marcos
  catch it.

## Trust boundaries

- No new external deps; same self-produced artifacts; torch.load local.
- Generated samples never executed — same as before.

## Verdict

300M is the defensible CPU ceiling for a *trainable* next experiment;
725M only if paired with corpus ≥50MB (days of teacher gen). The pair
"300M + growing corpus + stronger tokenizer" maximizes quality/hour on
this hardware; billions remains out of scope physically.
