# Architecture Evaluation — ncm-vs-llm-cpu-experiment

## Context and assumptions
- Demand (fact): train an NCM-style model on conventional hardware, reusing
  Qwen(-Coder) training where possible, "well above ordinary" result.
- NCM (fact, arXiv 2610.14162 + released configs): two stages —
  (1) multiscale residual VQ-VAE text tokenizer producing K code levels over a
  position pyramid (released LM1B: levels 1,8,16,24,32,40,52,64,76,92,108,128;
  12 books × 8192 codes; enc 6L/384d, dec 4L, BERT pre-tokenizer);
  (2) transformer generator predicting all codes of level k in parallel
  conditioned on levels <k (LM1B-small: 12L/768d, ctx 129, RoPE, QK-norm,
  block-diagonal attention mask, shared head + per-level bias).
- Hardware (fact): i5-1235U, 12 threads, 23 GiB RAM (~10 free now), Iris Xe
  iGPU (no CUDA), 106 GB free disk. torch 2.15.0+cpu venv `/tmp/llmenv` exists;
  CPU SFT of a 0.5B model already proven on this machine (colibri).
- Teachers (fact): `/tmp/llm/qwen05` (0.5B), `/tmp/llm/qwen15` (1.5B) HF dirs;
  `/home/kron/models/Qwen3-Coder-30B-...-Q3_K_XL.gguf` (13 GB, llama.cpp);
  `Qwen3-0.6B-Q8_0.gguf` (610 MB). colibri `.models/` empty.
- Assumption: "full" teacher = 30B GGUF via llama.cpp for a capped corpus
  slice, not full pretraining. Assumption: repo will be git-initialized.

## System map
```
[teachers: Qwen GGUF/HF] --inference--> corpus.jsonl (raw text/code)
      |                                       |
      v                                       v
[pre-tokenizer] --> [VQ-VAE trainer] --> tokenizer.pt --encode--> codes.npy
                                              |                    |
                                              v                    v
                                     [generator trainer] <---------+
                                              |
                                              v
                              generator.pt --> [eval: AR-matched baseline,
                              tok/s, task acc, code-pplx, samples]
```
Single host, sequential pipeline, long-running CPU jobs, all state on disk.

## [■] State
- Artifacts: corpus shards (jsonl), tokenizer.pt, encoded codes (.npy/.bin),
  generator checkpoints per epoch, metrics.jsonl, experiment config. Owner:
  this repo (`artifacts/` git-ignored; code/config tracked).
- Durability: hours-long training on a laptop → **mandatory resumable
  checkpoints** (save every N steps + optimizer state; survive SIGINT/power).
  Idempotent corpus generation (skip existing shard) — teacher runs are the
  most expensive stage to redo.
- No external state, no DB, no WAL needed; atomic-write checkpoint (tmp+rename)
  prevents half-written .pt after crash.

## -> Flow
- Pipeline is strictly sequential per stage but stages overlap safely:
  corpus gen → VQ-VAE train → corpus encode → generator train → eval.
- Backpressure = disk: encoding corpus expands size ×levels; 5M tokens ≈
  ~1–2 GB codes — fine within 106 GB.
- Teacher generation is the bottleneck hop (30B Q3_K_XL on CPU ≈ 1–3 tok/s via
  llama.cpp estimate → cap its share to a few K generations or hours budget;
  1.5B ≈ 5–15 tok/s estimate → bulk corpus).
- No concurrency required inside training beyond intra-op threads (torch
  `torch.set_num_threads(12)`, MKL). Optional: parallel corpus-gen workers.

## ⚡ Computation
- Budget math (estimates, to be measured in-task): CPU torch on this class of
  machine ≈ 0.5–2 GFLOPs effective/sustained per thread ×12 → order 5–20
  GFLOP/s. A 20M-param generator × ~6 FLOP/param/token ≈ 0.12 GFLOP/token →
  ~40–160 K tok/s forward-equivalent… realistically 5–20 K tok/s with
  backward+overhead → a 2–5M-token corpus, few epochs = **hours, feasible**.
  A 100M-param generator on CPU ≈ 3–5× slower → days; borderline. **Cap v1
  generator at ~10–40M params**; scale only if measured throughput allows.
- VQ-VAE ~5–15M params trains in well under an hour at this corpus scale.
- Memory: model ≤40M fp32 params ≈ 160 MB + Adam states ×3 ≈ 640 MB + batch
  activations ≈ <2 GB — comfortably inside 10 GB; keep batch/ctx modest,
  fp32 (no bf16 on this CPU path unless measured faster via AVX512 — i5-1235U
  has AVX2 only).
- Key measurement: steps/s + tok/s in first 15 min of each stage; abort-and-
  rescale rule if projected wall-time > agreed budget.
- NCM's payoff here is real: generation touches K≈8–12 sequential passes vs
  L≈100+ AR steps — aligned with colibri's bandwidth-bound finding.

## ✕ Failure
- **Codebook collapse** (VQ-VAE): code perplexity ≪ vocab → dead levels.
  Mitigate: monitor per-level codebook perplexity + recon BLEU/loss from
  step ~100; aux losses (fine_drop, code_corruption) as in released config.
- **Teacher corpus too slow**: 30B at ~1 tok/s → cap by wall-clock budget, not
  token count; fall back to 1.5B-only corpus (quality hit documented).
- **Fixed output length** (NCM inherent): answers longer than level-K length
  truncate. Mitigate: corpus normalized to target length; length-conditioning
  or segment continuation = v2, flagged as accepted limitation.
- **Divergence/NaN on CPU fp32**: grad clip + loss-spike detector → auto-restore
  last checkpoint with lower lr.
- **Eval unfairness**: AR baseline must be same param scale, same corpus, same
  train steps — else speedup claim is invalid. Lock this in the contract.
- Power/thermal on laptop: nice-level + checkpointing covers interruption.

## 🔒 Trust boundaries
- Generated code is never executed in this experiment's v1 (unlike colibri's
  loop) → no subprocess-escape boundary needed; if exec-eval is added later,
  reuse colibri's isolated-subprocess pattern (separate spec).
- Vendored MIT code (HazyResearch model/tokenization): treat as untrusted
  third-party input — review before import, pin files, keep LICENSE.
- HF/local checkpoints: integrity = local files only, no network at train
  time; no credentials involved anywhere in pipeline.
- Prompt-injection surface: none — corpus is generated data, not instructions
  to the agent.

## Risks and blind spots
- NCM quality on a *code/PT-BR instruction* domain is unproven (paper: LM,
  translation, summarization, sudoku). The experiment may show speed win but
  quality loss — that IS a valid experiment result; report honestly.
- Warm-start from Qwen AR weights: hypothesis, not fact — AR next-token
  weights may or may not transfer to level-parallel prediction; keep
  from-scratch arm as control so the warm-start claim is falsifiable.
- Codebook/vocab hyperparams are guesses; released config is the prior.
- 30B GGUF teacher throughput unmeasured (llama.cpp presence unverified —
  binary check in task).

## Trade-offs and decision
- **D (staged hybrid)** from research brief stands: faithful small NCM +
  Qwen-distilled corpus + optional warm-start arm.
- Rejected: (a) 100M+ scale — days of CPU, low marginal info; (b) train on
  GGUF extraction for warm-start — extra conversion layer, use HF dirs
  instead; (c) GPU-less official eval code — CUDA-oriented, write own.
- Smallest falsifying test: end-to-end 8-level NCM on a 100K-token slice
  must produce coherent reconstruction + measurable tok/s before any
  full-scale run.

## Validation plan
1. VQ-VAE: recon loss ↓, per-level codebook perplexity in healthy band,
   decode-level samples readable.
2. Generator: level-wise CE loss ↓ across all levels (not only level 0).
3. End-to-end: NCM tok/s vs param-matched AR baseline on same corpus/hardware;
   task metric on held-out set; qualitative level-by-level decode dump.
4. Warm-start arm: ≥ from-scratch arm's quality at equal steps, else report
   negative result.

## Inputs for spec_to_tasks
- Constraints: generator ≤40M params v1; corpus 1–5M tokens; resumable
  checkpoints mandatory; fp32; 12 threads max; artifacts under `artifacts/`
  git-ignored; no code execution of model outputs.
- Acceptance: (a) tokenizer trains & encodes corpus with healthy codebook
  usage; (b) NCM generator generates parseable outputs at K passes; (c)
  measured tok/s > matched AR baseline at comparable task quality OR an
  explicit documented negative result; (d) reproducible: config + seeds +
  metrics committed.
- Tests: unit (VQ quantizer shapes, mask correctness, level-schedule),
  integration (encode→train→decode pipeline on 100K-token smoke slice),
  benchmark (tok/s A/B).
- Env: reuse `/tmp/llmenv` or create project `.venv`; pin torch==2.15.0+cpu
  equivalent; document `hf`/llama.cpp availability or add to deps.
