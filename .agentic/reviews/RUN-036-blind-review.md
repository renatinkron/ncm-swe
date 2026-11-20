# Blind Review — RUN-036

Scope: `scripts/convdistill.py`, `scripts/kpack.py`, `scripts/chat_student.py`,
`scripts/watch_convdistill.py` — reviewed cold, no spec/task context.

## Findings

### Major

- **[major] `convdistill.py:146-154, 363` vs `kpack.py:91`, `chat_student.py:41`** —
  LoRA-mode checkpoint is unloadable by the shipped consumers. In `MODE=lora`,
  `torch.save(model.state_dict(), student_final.pt)` serializes the wrapped
  module tree (`...q_proj.linear.weight`, `...q_proj.A`, `...q_proj.B`). Both
  `kpack.py` and `chat_student.py` call `model.load_state_dict(...)` (strict
  default) on a *plain* model → guaranteed `RuntimeError` on every parameter.
  No code path merges adapters or re-injects LoRA before load, so the
  advertised `MODE=lora` produces a `student_final.pt` the delivered pipeline
  cannot consume. Worse, a `strict=False` workaround would silently load base
  weights and report results for an untrained student.

- **[major] `convdistill.py:95, 233-234, 362-363`; `kpack.py:89-93`; `chat_student.py:40-42`** —
  No fault tolerance + checkpoint only at the very end. `teacher_health()` is
  checked once at startup; every `teacher_chat` `urlopen` is unguarded, so one
  transport error/timeout/HTTP-500 anywhere in 24 rounds kills the run.
  `student_final.pt` is written only after the final eval, so all training is
  lost. Downstream the failure is silent: both consumers only `print` when the
  ckpt exists and otherwise run the **base** student — and `kpack_eval.json`
  (kpack.py:140-146) records no field saying which weights produced the
  numbers, so the artifact would attribute scores to a trained model that
  never existed.

- **[major] `kpack.py:56-62, 117-135`** — The transplant eval cannot falsify
  its own claim. `sig_rubric` derives "expected tokens" that include the
  generic literals `"def "`, `"return"`, `"class "`, `"self."` — present in
  essentially any Python answer — so naked and armed rubrics both saturate.
  Additionally nothing enforces disjointness between card probes
  (teacher-chosen coding tasks) and the 8 fixed eval questions: a
  near-duplicate card injects the ideal answer verbatim. The docstring claims
  "catalog > naked proves the pack delivers knowledge," but the artifact can
  only ever emit ceiling numbers — a metric structurally incapable of
  distinguishing the two arms, which misleads any reader of `kpack_eval.json`.

### Minor

- **[minor] `convdistill.py:78-80, 231, 271-282`; `kpack.py:74-85`** — Artifacts
  are not run-scoped. `metrics.jsonl`/`transcript.jsonl` are opened `"a"`: a
  re-run appends a second baseline (`round: -1`) and a second `0..N` series
  into the same files. `eval_frozen.json` is silently reused if present even
  if `EVAL_QUESTIONS` or `STUDENT` changed (stale teacher answers scored
  against new questions). `kpack.py` then builds cards across *all* historical
  transcript rows, so `knowledge_pack.json` can mix runs with no marker.

- **[minor] `convdistill.py:194-199`** — Label masking assumes prefix-stable
  tokenization. `pid = tok(prompt)` and `fid = tok(full)` are tokenized
  separately, then `labels[:, :pid.shape[1]] = -100`. A BPE merge across the
  prompt/answer boundary (the prompt ends with `assistant` + newline) shifts
  alignment by a token: an answer token is silently masked out of the loss (or
  a prompt token left unmasked), and the trained boundary differs from what
  inference actually conditions on. No assert compares the two prefixes.

- **[minor] `convdistill.py:218-222, 263-268`** — `weight_drift` under-reports
  and overwrites. The snapshot keeps only the first 4 trainable params, yet
  the metric reads as model-level drift. Worse, `name.rsplit(".", 2)[0]` maps
  `q/k/v_proj` of the same layer to one key, so later params silently
  overwrite earlier norms; in lora mode A/B of several projections collapse
  under a single key too. metrics.jsonl loses entries with no indication.

- **[minor] `convdistill.py:301-313, 348`** — Teacher JSON parse failure skips
  the round in metrics only. The transcript gets an `{"error": ...}` row but
  `metrics.jsonl` gets no row at all → silent round-number gaps for anyone
  analyzing metrics alone.

- **[minor] `convdistill.py:108-119`** — `parse_teacher_json` validates keys,
  not types. Non-string `probe`/`ideal_answer` or non-string items in
  `expected_tokens` crash mid-run inside `tok()`/`rubric` (`in <string>`
  requires str). The system prompt's own contract — `expected_tokens` must
  appear verbatim in `ideal_answer` — is never checked, so a malformed card
  silently poisons the rubric and later the kpack catalog.

- **[minor] `convdistill.py:200-201, 217, 344`** — `NaN` leaks into
  metrics.jsonl. The early-return `float("nan")` for degenerate label cases,
  plus any CE NaN, flows through `round(loss, 4)` into `json.dumps`, which
  writes the bare token `NaN` — invalid per the JSON spec (Python-only
  extension). Strict parsers downstream reject the whole file.

- **[minor] `convdistill.py:322-337`** — Retention only ever re-probes
  `failed_items[-1]`. Older failed items (list capped at 20) are never
  retested and can never exit the list, so reported `retention`/`retries`
  always reflect the freshest failure while the list shape implies a queue of
  pending items being worked through.

- **[minor] `convdistill.py:364-366`** — No seed and no model identity in the
  record. Only the final row logs `mode`/`lr`; nothing records the STUDENT
  path, teacher model id, or an RNG seed (LoRA `kaiming_uniform_` init is
  unseeded) → the run is irreproducible and metrics can't be attributed to a
  specific checkpoint version.

- **[minor] `kpack.py:102-110`** — `knowledge_pack.json` metadata is hardcoded:
  `"dim": 896` and `"qwen2.5-0.5b embed_tokens mean-pool"` are emitted
  regardless of `STUDENT`, and the description calls the pack
  "model-agnostic" / "query by cosine over your own embedding space" — but the
  index is only valid in *this* student's embedding space; another model's
  embeddings are incomparable. The artifact's self-description overstates its
  portability.

- **[minor] `convdistill.py:177`; `kpack.py:51-52`; `chat_student.py:44, 57-58`** —
  `eos_token_id=tok.convert_tokens_to_ids(IME)` is unguarded. If the
  configured tokenizer lacks the im_end special token, this yields `None` →
  generation never early-stops (always burns MAX_NEW) and, because decode uses
  `skip_special_tokens=True`, post-EOS hallucinated turns land inside
  transcript/livechat entries as if they were the student's reply.

- **[minor] `kpack.py:74-75, 97, 123, 142-143`** — Fragile transcript intake:
  each line is `json.loads`-ed twice, the `open()` handle in the comprehension
  is never closed, and any malformed/torn trailing line (common when the
  producer died mid-write) crashes the script. `torch.stack([])` crashes when
  no cards exist; `eval_frozen.json` missing/empty → FileNotFoundError or
  `ZeroDivisionError` on the means. No graceful degradation anywhere.

- **[minor] `watch_convdistill.py:14-21`** — The watcher re-reads and re-parses
  the entire transcript every 3 s (O(n) growth over a long run) and crashes on
  any malformed or partially-flushed line — i.e., it is most likely to die
  precisely when the run does something anomalous. An unclosed `open(T)` per
  poll also leaks FDs until GC.

### Nit

- **[nit] `convdistill.py:255`** — `if p is not m.linear.weight` is dead code:
  `m.parameters(recurse=False)` never yields the child Linear's weight. The
  guard reads as if it protected the base weight — it protects nothing.
- **[nit] `kpack.py:98-100`** — `(ART/"kpack_index.npy").write_bytes(b"")` is a
  dead write immediately overwritten by `np.save`; `import numpy` mid-function.
- **[nit] `kpack.py:15,17`; `watch_convdistill.py:6,11`** — unused imports
  (`math`, `sys`, `os`) and unused path var `M`.
- **[nit] `convdistill.py:358-362`** — when `ROUNDS % EVAL_EVERY == 0`, the
  periodic eval@N and the `final` eval both run → duplicate measurement rows
  (round N-1 "eval" + round N "final") in metrics.jsonl.
- **[nit] `kpack.py:131`** — `question[:80]` truncates the recorded eval
  question (the `Stack`-class prompt is ~100 chars) in `kpack_eval.json`.
- **[nit] `convdistill.py:239-240`** — unguarded `/proc/meminfo` read and
  `re.search(...).group(1)` → `AttributeError`/`FileNotFoundError` off Linux.
- **[nit] `chat_student.py:61`** — no `ART.mkdir`: the livechat append crashes
  if `artifacts/convdistill/` doesn't exist (works only because convdistill
  created it first).
- **[nit] `convdistill.py:220`** — `dict(model.named_parameters())` rebuilt on
  every snapshot iteration.
- **[nit] `chat_student.py:55-57`** — `temperature=None, top_p=None` passed when
  `TEMP <= 0` → HF GenerationConfig "flag unused" warnings; harmless noise.
- **[nit] `convdistill.py:193-197, 183-190`** — no input truncation in
  `sft_step`/`ce_on_text`: a verbose teacher answer makes a very long
  single-sample batch (fp32 logits `[1, S, ~151k]`).
- **[nit] `convdistill.py:149-153`** — `inject_lora` mutates the module tree
  while iterating `model.modules()`; safe today (the wrapper child is named
  `linear`, not in `names`) but fragile against future name additions.

## Patterns

1. **Artifacts under-record provenance and semantics** (findings 2,3 + minors
   on run-scoping, drift overwrite, round gaps, retention scope, seed/model
   id, pack metadata): every produced file reads as self-describing but omits
   or distorts what produced it — silent base-weight fallback, stale eval set
   reuse, saturating rubric, missing rounds. A reader of `metrics.jsonl` or
   `kpack_eval.json` can be misled without anything looking wrong.
2. **Unguarded trust boundaries** (teacher JSON typing, tokenizer eos lookup,
   torn jsonl lines, prefix-tokenization assumption): external inputs are
   trusted structurally; failures surface as crashes or silent skew far from
   the cause — worst case, mid-run crashes in a script that checkpoints only
   at exit.
3. **Dead/vacuous code** (dead requires_grad guard, dead `write_bytes(b"")`,
   unused imports/vars): code that looks meaningful but does nothing, which
   itself can mislead a reader about what is protected.

## Trajectory

`.agentic/state/runs/RUN-036.events.jsonl` shows a coherent chain:
execute (≈50 min elapsed → consistent with a real 24-round run) → test
(nc=2) → remediate → re-execute → test (nc=0) → history retry → passed. No
step self-reports "passed" without a corresponding dispatch, and the NC-2
remediation (`retention_attempt` field) is visible in the code at
`convdistill.py:323,326,351`. No anomalies flagged.

## Verdict

Mechanically the pipeline runs end-to-end in the default (full fine-tune) path
and its prints/metrics are consistent, but the delivered code is fragile at
exactly the points a blind reader would trust: the `MODE=lora` path emits a
checkpoint none of the shipped tools can load; a single network hiccup destroys
the whole run and downstream tools then silently evaluate the *untrained* base
model with no provenance recorded; the headline kpack metric cannot
discriminate its two arms; and the jsonl artifacts are neither run-scoped nor
strict-JSON-safe. Recommended: fix checkpoint save/load symmetry (merge or
re-inject adapters), add retry + periodic checkpointing + weight-provenance in
`kpack_eval.json`, harden `sig_rubric` (drop generic keyword tokens, enforce
card/eval disjointness), and validate teacher JSON types before use.

major: 3, minor: 12, nit: 11
