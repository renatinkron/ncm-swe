# Blind Review — RUN-037

Cold review of `scripts/domcart.py` (adjacent module `scripts/convdistill.py`
inspected for interface/contracts only). Focus areas: correctness, subprocess
safety, pattern extraction, LoRA inject/unwrap, silent failure modes.

## Findings

| file:line | severity | issue |
|---|---|---|
| scripts/domcart.py:69-71 | major | **Pattern extraction corrupts non-`r"..."` notations.** Alt-4a `\\\\d` matches source-level double-backslash text (`"\\d{5}"`, i.e. valid Python non-raw-string notation) and captures it *verbatim*; passed through JSON to `TEST_WORKER`'s `re.compile`, `\\d` parses as literal-backslash + `d{n}` — a semantically correct student answer is scored 0 on all positive tests. Alt-3 `re\.compile\(\s*["']...` has the same verbatim-capture defect for `re.compile("\\d{5}")`. Separately, alt-4b's char class `[\\w^\[\]().*+?{}$|\-]` is parsed as literal backslash + literal `w` + metachars (the `\\w` was almost certainly meant as the `\w` class) — it contains **no digits and no letters other than `w`**, so `"\d{5}"`, `"[0-9]{5}-[0-9]{4}"` etc. can never match it. Net effect: only the `r"..."` form (what the teacher-trained format produces) extracts correctly; other correct notations silently fail or fall through to the prose fallback → **plugged-vs-baseline delta is biased upward** because the baseline/unplugged model emits untrained formats more often. Fix: unescape captured literals (e.g. `ast.literal_eval` on the quoted token), repair the char class to include `\w`/digits/letters, and cover `re.match/search/fullmatch(` too. |
| scripts/domcart.py:79-80, 121-135 + EVAL_TASKS:42-67 | major | **Structural ~0.50 pass_rate floor + always-compiling prose fallback.** 17 of 34 test cases have `want=False`; any pattern that compiles but matches nothing earns every negative case → ~0.50 floor for a *useless* pattern. The fallback (first line of answer, quotes stripped) returns prose like `"Here is the regex:"` whenever `len < 120` — plain text always compiles, so `compile:true` is recorded on garbage rows and `err:"no_pattern"` almost never fires. The reported `base_pass_rate=0.500` equals exactly the never-matching floor, meaning the "generalist baseline" may be measuring extraction failure, not model skill — and `unplug_pass_rate` has the same ambiguity. Artifact readers cannot distinguish "model regressed to floor" from "extractor failed" without manually auditing every row's `pattern`. Fix: record extraction-method (regex-hit vs fallback vs none) per row, record the worker `err`, and flag fallback-extracted "patterns" containing spaces/prose. |
| scripts/domcart.py:105-107, 125-129 | minor | **Failure causes collapsed to a boolean.** `run_pattern_tests` computes `err` (`"compile"`, timeout message, `"no_pattern"`) but `eval_domain` only persists `res.get("ok")` as the `"compile"` field — the `err` string is dropped entirely. A ReDoS timeout, a compile error, a worker crash, and no-pattern-found are indistinguishable in `metrics.jsonl`. Per-row `err` should be recorded. |
| scripts/domcart.py:105 | minor | **Worker stderr discarded → cryptic errors.** If the worker dies before printing (e.g. `json.loads(sys.argv[1])` fails), stdout is empty, `splitlines()[-1]` raises `IndexError`, and the surfaced err is `"list index out of range"`. Include `out.stderr` tail / `returncode` in the error payload. |
| scripts/domcart.py:150-152, 226-229 | minor | **Cartridge artifact under-specified and never round-trip validated.** `lora_state` saves `A`, `B`, and `lora_r` but not `scale`/alpha — the scaling factor (16.0/r) exists only hardcoded in `convdistill.LoRA`; a consumer loading `cartridge_regex.pt` with a different alpha convention silently applies wrong-magnitude deltas. The saved `.pt` is also never reloaded into a fresh model to prove the artifact alone reproduces the plugged eval — the headline artifact is asserted, not demonstrated. No base-model identifier is stored either. |
| scripts/domcart.py:201-218, 238-249 | minor | **Silent no-op paths produce normal-looking metrics.** If `inject_lora` returns 0 (unexpected module names) or `lessons == []` (teacher dies after health check / all parse attempts fail), the training loop does nothing, plug ≡ base, and `metrics.jsonl` still records a full "cartridge" record. `n_inj`/`len(lessons)` are logged but no guard marks the record degenerate. |
| scripts/domcart.py:194, 215 | minor | **No type validation on teacher fields.** `parse_teacher_json` (convdistill.py:108-119) checks key presence only; if `probe`/`ideal_answer`/`skill` come back as list/dict, `covered.append(raw["skill"])` succeeds but `sft_step` crashes mid-run on `prompt + answer` TypeError, losing all metrics accumulated to that point. |
| scripts/domcart.py:215-224 | minor | **NaN loss poisons the cartridge silently.** `sft_step` early-returns only when <2 supervised tokens; a NaN/inf forward loss still backprops → `opt.step()` writes NaN into A/B → subsequent plug-eval generations are garbage (scored at the ~0.5 floor, see major #2) while unplug-eval stays clean. Loss isn't checked before stepping. |
| scripts/domcart.py:64-66 | minor | **Ambiguous expected label.** `("World9", True)` for "word starting with an uppercase letter" encodes an unstated `\w+`/alnum-word assumption; a defensible `^[A-Z][a-z]+$` answer is scored wrong on that case. |
| scripts/domcart.py:15, 21 | nit | Dead imports: `time`, `merge_lora` (script unwraps rather than merges). |
| scripts/domcart.py:132 | nit | `res.get('total')` prints `0/None` on failure rows — cosmetic. |
| scripts/domcart.py:171, 223, 236, 243-249 | nit | `math.exp(ce)` can `OverflowError` (CE>709) or propagate NaN into `metrics.jsonl`, which `json.dumps` emits as bare `NaN`/`Infinity` tokens — invalid strict JSONL. |
| scripts/domcart.py:114-115 | nit | `convert_tokens_to_ids` of the im-end special token may return `None` for tokenizers lacking it → `eos_token_id=None`, generation runs to `max_new_tokens` (silent slowdown, not a crash). |
| scripts/domcart.py:212-213 | nit | `sft_step` imported locally despite top-level convdistill import block; epoch count `range(2)` is a magic constant while LESSONS/LORA_R/LR are env-configurable. |

## Positive checks (verified correct)

- `unwrap_lora` (138-142): restores the original `nn.Linear` objects; base weights were never mutated (frozen + optimizer holds only A/B) → unplug is bit-exact. Module-tree mutation during `model.modules()` iteration is safe here (each mutated module's frame is suspended at `yield`, before its `_modules.items()` iterator is created; `named_children()` is materialized via `list()`).
- `lora_state` runs before unwrap (227 vs 232) and clones tensors — save ordering is correct.
- Subprocess containment: `sys.executable -c` worker, JSON via argv, `timeout=5`, broad `except` converts all failure modes to `{ok:False}` — bounded, no hangs; TimeoutExpired/JSONDecodeError/IndexError all funnel to err.
- Train/eval ordering: baseline eval before LoRA injection; `manual_seed(0)` precedes model load and `kaiming` init.

## Patterns

The two majors share one root cause: **the extractor operates on surface text
with no unescape/semantic step**, so "what the model wrote" and "what
`re.compile` receives" diverge — and the always-compiling prose fallback plus
the 17/34 negative-test floor convert that divergence into a plausible-looking
~0.50 pass_rate rather than an observable failure. Three minors (err dropped,
stderr discarded, no degenerate-record flag) share a second theme: **failure
diagnostics are computed but not persisted**, so `metrics.jsonl` cannot
self-describe *why* a row scored as it did.

## Trajectory

`.agentic/state/runs/RUN-037.events.jsonl` reviewed: begin → task_plan →
execute → test(compliant, nc=1) → remediate → execute → test(NC-1 reverif,
`diff_scope:"single-constant"`, `timeout:"5"`, py_compile ok) → history. Every
test event cites concrete evidence; the remediated NC matches the present
`timeout=5` in `run_pattern_tests`. No self-reported passes without tool-call
evidence; no claimed-but-absent dispatches. Clean.

## Verdict

The LoRA machinery (inject → freeze → train A/B → save state → unwrap) is
correct and unplug is provably bit-exact, and the subprocess eval is properly
sandboxed with a timeout. However the headline metric is not trustworthy as
written: pattern extraction silently corrupts correct non-`r"..."` notations,
the char-class alternative is effectively dead, and the negative-test floor
plus prose fallback manufactures a ~0.50 pass_rate for patterns that match
nothing — while the recorded rows omit the err/extraction provenance needed to
detect any of this. The reported plug-vs-base delta is likely inflated; the
absolute pass rates should be re-derived after fixing extraction and recording
per-row diagnostics.

major: 2, minor: 7, nit: 5
