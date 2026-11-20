# Blind Review — RUN-043

Scope: `scripts/_bpe_probe.py` (new). Adjacent modules read for convention/boundary
checks only: `scripts/ncm_codetrain.py`, `scripts/_constr_decode.py`. No spec,
task_plan, contract, or execution/compliance reports were opened.

## Findings

| # | file:line | severity | issue |
|---|-----------|----------|-------|
| 1 | `_bpe_probe.py:27-28,163-164` | **major** | **Silent train/held-out contamination via env knob.** `N_TRAIN` is env-tunable (default 19000) but `HOLDOUT` is hardcoded `(19000, 20000)`. `train_rows = rows[:N_TRAIN]` overlaps `held = rows[19000:20000]` for any `N_TRAIN > 19000`, and `load_rows(HOLDOUT[1])` always materializes 20000 rows so the overlap is silent, not a crash. The docstring advertises `N_TRAIN` as a tunable and claims "Comparação honesta … held-out rows 19000:20000" — one env tweak (`N_TRAIN=20000` to "use all data") turns the held-out CE/top-1/parse metrics into train-set metrics. No `assert N_TRAIN <= HOLDOUT[0]`. With defaults it is consistent, so this is a loaded footgun rather than an active bug — but it sits exactly on the artifact's stated purpose (a decision metric). |
| 2 | `_bpe_probe.py:26,57,95` | minor | `VOCAB = 8192` hardcoded and never validated against `fast.get_vocab_size()`. If `bpe_code_tokenizer.json` ever differs: ids ≥ VOCAB crash loudly in `emb` (acceptable), but a *smaller* real vocab silently wastes embedding rows and skews CE/top-1 vs. the true token distribution. One-line assert missing. |
| 3 | `_bpe_probe.py:164,205-209` | minor | **Empty-heldout crash.** If the corpus/cache yields < 20000 rows, `held` is a 0-row tensor, `range(0, 0, 64)` produces no chunks, and `torch.cat(logits_chunks)` raises `RuntimeError` on the empty list. Same 20000-row assumption exists in the sibling, so it is convention-consistent, but the boundary is unguarded and the failure is uninformative. |
| 4 | `_bpe_probe.py:183,194-199` | minor | **Checkpoint lags evaluated model.** `torch.save` runs only inside `step % STEP_LOG == 0`. When the wall-clock budget expires mid-interval, the on-disk `gen_bpe_probe.pt` is up to `STEP_LOG-1` steps older than the in-memory model the held-out metrics describe — an `EVAL_ONLY=1` rerun will not exactly reproduce `bpe_probe_metrics.json`. `STEP_LOG` doubling as the save cadence makes a user-requested quieter log degrade checkpoint fidelity. A final save after the loop closes the gap. |
| 5 | `_bpe_probe.py:189-191` | minor | **No gradient clipping.** Sibling `ncm_codetrain.py:211` clips to 1.0. From-scratch training at `LR=3e-4` (this file's own default, chosen because "LM direto aguenta mais") with unclipped AdamW has no spike protection — a convention drift that matters more here than in the fine-tuning sibling. |
| 6 | `_bpe_probe.py:74,168,184` | minor | **Unseeded init and data sampling.** Weight init and `torch.randint` batch draws use the ambient RNG; only `generate` calls `torch.manual_seed` (and reseeds the *global* RNG mid-run). Sibling uses a dedicated `torch.Generator().manual_seed(0)` for batch order (`ncm_codetrain.py:197`). Run-to-run variance of the probe cannot be reproduced or bounded from the artifacts. |
| 7 | `_bpe_probe.py:145-156,134` | minor | **`filler_fraction` fails closed toward 0.0.** Any `tokenize` failure (unterminated string, `TokenError`, stray indentation) returns 0.0 — so the most filler-heavy possible output (one giant unterminated string, or pure garbage) is scored as *zero filler*, biasing `filler_frac` downward exactly where the metric should peak. Garbage is indistinguishable from filler-free text in the aggregate. |
| 8 | `_bpe_probe.py:205-212` | minor | **~4.3 GB logits materialization.** Held-out eval concatenates full logits before slicing: 1000 × 128 × 8192 × fp32 ≈ 4.3 GB transient on CPU, plus the argmax pass. Streaming `ce`/argmax per 64-row chunk would be near-free; as written, eval can OOM on a modest container after an otherwise-successful 3 h train. |
| 9 | `_bpe_probe.py:214-216` | minor | **Metrics JSON lacks hyperparameter provenance.** `res` records `steps`, `train_secs`, and eval numbers but omits `LR`, `BATCH`, `N_TRAIN`, model dims, and the env flags (`SMOKE`, `EVAL_ONLY`). Sibling records `epochs`/`lr`/`n_rows` in its metrics (`ncm_codetrain.py:179`). `bpe_probe_metrics.json` cannot be interpreted standalone later — which hyperparameters produced this number is unrecoverable from the artifact. |
| 10 | `_bpe_probe.py:76,88-90,114-116` | minor | **Token 0 aliases BOS and terminator.** Generation seeds every row with literal token 0; `decode` truncates at first 0 and `generate` early-breaks on all-0. Sibling convention (`ncm_codetrain.py:87-88`, `_constr_decode.py:57`) treats 0 as EOS/end and 1 as pad. Whether token 0 occurs inside raw corpus rows depends on the tokenizer's special-token behavior — if rare, the model's first-step distribution is conditioned on a nearly unseen input (the 127 remaining steps recondition, so impact is bounded but unmeasured). |
| 11 | `_bpe_probe.py:16` | info | Dead imports: `pickle` and `sys` imported, never used (sibling uses both; this file needed neither — suggests copy-paste of the import line). |
| 12 | `_bpe_probe.py:14,181-183,200-201` | info | Docstring drift: "SMOKE=1 … ~30 steps + eval mini, sem budget" — code applies a 120 s budget *and* runs the full-size eval (1000-row held-out CE + 2×32 generations). Neither "mini" nor "sem budget" is accurate. Harmless, but the docstring is the only usage doc. |
| 13 | `_bpe_probe.py:103` | info | `open(f).read()` leaves the handle unclosed and omits `encoding=`; identical pattern in sibling `ncm_codetrain.py:98` — convention-consistent, noted for completeness. |
| 14 | `_bpe_probe.py:171-175` | info | `EVAL_ONLY=1` still constructs `AdamW` over all parameters — dead allocation, no effect. `torch.load(weights_only=False)` on the script's own artifact matches house convention (siblings do the same); acceptable risk for a self-produced checkpoint. |
| 15 | `_bpe_probe.py:119-126` vs `_constr_decode.py:46-51` | info (positive) | `n_real` is logic-identical to the sibling's hardened oracle (top-level `tree.body` stmts excluding `Expr(Constant)`), and generation params `temp=0.6, top_k=24` match `_constr_decode.py:117-118` defaults — the "mesmo oráculo endurecido" claim is borne out in code. |
| 16 | `_bpe_probe.py:198-199,135-141` | info (positive) | Atomic checkpoint write via `tmp` + `os.replace` matches sibling convention (`ncm_codetrain.py:231-232`). Eval is safe by construction: generated text is only `ast.parse`d/`tokenize`d — never `compile`d/`exec`d (sibling `ncm_codetrain.py:139` does `compile`; this file does not, so nothing generated is ever executed). |
| 17 | `_bpe_probe.py` vs siblings | info | Correctly drops the `sys.path`/`os.chdir(ROOT)`/hydra-shim dance — pure `tokenizers`+torch deps make vendor imports unnecessary. Boundary is clean. |

### Trajectory (`.agentic/state/runs/RUN-043.events.jsonl`)

- **t1 — positive signal.** `execute` spanned 20:51:12 → 00:16:34 (~3 h 25 m), consistent with `TRAIN_SECS=10800` (3 h) plus data prep/encoding — wall-clock evidence the training actually ran rather than a fabricated pass.
- **t2 — observation.** First `test` event (00:17:42) recorded `verdict: non_compliant, nc: 4, blocking: NC-1_missing_handoff` ~14 min before the step closed in `loop.json` (00:31:23) — the verdict event precedes artifact completion. Ordering is mildly unusual but not a violation; the step did route to `remediate` as required.
- **t3 — observation.** Remediation loop ran `remediate` (9 s) → `execute` (4 s) → `test` compliant with `nc_fixed: NC-1..4` and `trainer_alive: true` probe evidence. The 4 s execute is plausible for a docs-only fix (blocking NC was a missing handoff, no code change needed). No claimed-but-absent dispatches; every `step_log` entry has a matching event.
- No trajectory findings rise above `info`.

## Patterns

**P1 — Unguarded constants/env drift (findings 1, 2, 3; partially 10).** Three-plus
findings share one root cause: hardcoded constants (`HOLDOUT`, `VOCAB`, the
20000-row assumption, token-0 semantics) and env knobs (`N_TRAIN`) are decoupled
with zero cross-validation. Every failure mode is silent-or-cryptic rather than
fail-fast. A five-line preamble (`assert N_TRAIN <= HOLDOUT[0]`, `assert
fast.get_vocab_size() == VOCAB`, `assert len(rows) >= HOLDOUT[1]`) would
eliminate the whole class.

## Verdict

Self-contained, convention-aware probe: the hardened oracle is faithfully
replicated, sampling params match the sibling harness, checkpointing is atomic,
and nothing generated is ever executed — no security findings. The single
`major` is environmental, not default-path: `N_TRAIN > 19000` silently converts
the held-out comparison into a train-set measurement, and this script exists
precisely to make that comparison trustworthy. The `minor` cluster is dominated
by missing validation (P1) plus reproducibility gaps (unseeded run, no
hyperparameter provenance in the output JSON) that weaken the artifact's value
as a decision record but do not invalidate a default-config run. Recommend:
land the P1 assert block and a final post-loop checkpoint save; consider
recording `LR`/`BATCH`/`N_TRAIN` in `res` before this probe's numbers are cited
against the VQ baseline.

Verdict counts: 1 major, 9 minor, 7 info, 0 blocker.
