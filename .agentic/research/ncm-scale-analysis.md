# Research: NCM scale analysis — can we reach "billions"?

## What the paper actually did (verified from vendored release)

Released NCM generators (config/eval/*.yaml in vendor):
- **LM1B small**: 12 layers × 768 embd ≈ **~100M**
- **LM1B medium**: 24 × 1024 ≈ **~300M**
- **LM1B large**: 36 × 1280 ≈ **~725M**
- Tokenizer: BERT pre-tokenizer (30,528 vocab) + VQ-VAE (6×384, 12 levels)
- Tasks: LM1B, XSum, WMT14, Sudoku, TinyStories, OWT — benchmark suites
  with large corpora, not 6MB.

**The paper never went to billions.** Its claim: NCM matches/beats diffusion
baselines at **9–31× faster generation** at ~100–725M scale. "Billions" is
not part of the NCM result — it's LLM-scale intuition misapplied.

## What our run proved at 60M

- Mechanism works (6 level-passes beat ~127 token steps: ×1.05–×1.34/sample).
- Quality ceiling hit at 60M params × 6MB corpus: "plausible fragments",
  parse 0.0. AR loss 3.30→1.5 shows memorization, not generalization.

## What "billions" would require on this CPU (i5-1235U, 23GB)

- Training step cost ≈ linear in params. Measured ~13s/step at 60M →
  ~160s/step at ~725M (12×). 2000 steps ≈ **~4 days** — technically
  feasible with the days-regime, ~12-15GB RAM (marginal fit in 23GB).
- **~1B**: ~3min/step → 2000 steps ≈ 4 days + ~18-20GB RAM (won't fit with
  Adam states + activations at seq 512 — borderline/impossible).
- **Billions (3B+)**: ~30GB+ RAM — physically impossible here.

## The real bottleneck isn't params — it's DATA

A 300-700M model on a 6MB corpus memorizes in ~1 epoch and plateaus
(overfitting visible: AR train loss 1.5 while samples stay fragmentary).
To benefit from more params we'd need **~50-500MB corpus** — teachers
produce ~80 chars/s on CPU → 50MB ≈ 7+ days of pure teacher generation.
Python-lib harvesting is the only "free" lever (limited to installed
source, already mostly consumed).

## Honest options ranked

1. **~300M (lm1b-medium-equivalent)**: ~50s/step → 2000 steps ≈ 28h.
   Feasible; fits RAM (~7GB). Needs more data to not just memorize — pair
   with continued corpus growth. **The realistic "bigger" target.**
2. **~725M**: ~4 days/2000 steps, RAM-marginal. Only sensible if corpus
   grows proportionally — otherwise an overfit demo.
3. **Billions**: impossible on this hardware — needs GPU/cloud or a
   cluster. Not honest to attempt.
4. **Better-than-params lever**: improve tokenizer quality (train VQ-VAE
   longer / bigger codebooks) — NCM quality is bounded by the code space;
   a better tokenizer lifts NCM quality at fixed params, cheap.

## Reuse check

All components exist — scaling is config-only (generator dims in
configs/generator.yaml; Qwen-1.5B warm-start available for a ~1.5B-class
transplant, but RAM prevents running it at train speed).
