# Blind Review — RUN-047

Scope: `scripts/_teacher_gen.py`, `scripts/_eval_deep256.py`, `scripts/_bpe_probe.py`.
Adjacent modules consulted for boundary/interface checks only: `scripts/code_bpe.py`,
`scripts/_func_units.py`, `scripts/_eval_bpe_prefix.py`, `scripts/_overfit_test.py`,
`scripts/ncm_codetrain.py`. No spec/task_plan/contract opened.

## Findings

### Blocker

None.

### Major

1. **`scripts/_bpe_probe.py:132-141` — `decode()` is not the inverse of the byte-level BPE encoding; all non-ASCII output is corrupted and two byte ranges are mapped to wrong codepoints.**
   `code_bpe.py:32-43` (the repo's own `bytes_to_unicode`) assigns excluded bytes sequential
   codepoints `256+n`: bytes 0–32 → 256–288, bytes 127–160 → **289–322**, byte 173 → **323**.
   `_BMAP` instead uses `256+b`, which is coincidentally right only for bytes 0–32; its
   entries for keys 383–416 (`256+127..256+160`) address codepoints that never occur in the
   byte-level alphabet — dead entries — while the real codepoints 289–322 and 323 are absent
   (byte 127 decodes as `'ġ'`, byte 173 as `'Ń'`, etc.).
   Separately, the join+`translate` approach never performs the latin-1→UTF-8 reassembly
   (`code_bpe.py:50` does `bytes(u2b[c] for c in s)`; `decode()` keeps per-byte chars), so any
   multi-byte UTF-8 sequence decodes to mojibake (`é` → `Ã©`). The corpus has ~165k non-ASCII
   chars (`data/corpus_code`), so this materially corrupts every saved sample JSON
   (`*_deep_*.json`) and distorts `ast.parse`/tokenize metrics — mojibake inside strings and
   comments still parses, in identifier positions (e.g. `©`) it forces SyntaxError.
   Same function: `if 0 in ids: ids = ids[:ids.index(0)]` truncates at token 0 — consistent
   with BOS=EOS=`<|endoftext|>` (confirmed by `code_bpe.py:71`) — but **PAD id 1 is not
   filtered**: `id_to_token(1)` emits the literal text `"<|pad|>"` into decoded output
   whenever the model emits id 1 before any 0.

2. **`scripts/_bpe_probe.py:220-221, 240-249` — heldout CE and top-1 accuracy are computed over padding targets, inflating both metrics.**
   Unit rows are `[0] + ids[:126] + [0] + [1]*(CTX-len)` (lines 108-111); for typical short
   units, a large fraction of the 127 supervised positions are pad id 1, which the model
   learns trivially ("after 0 → always 1"). `ce`/`acc` count every position with no
   `ignore_index`/pad mask, so `heldout_top1` drifts toward 1.0 and `heldout_ce` is deflated
   by free positions — exactly the "padding inflating accuracy" failure mode. Sibling code in
   the same repo masks pad for the analogous metric (`_overfit_test.py:41-42` masks `~pad`;
   `:55` uses `ignore_index=PAD_ID`), so this convention is internally inconsistent, not a
   project-wide norm. The same unmasked loss is used in training (line 220), which is a
   design choice, but reporting pad-inclusive numbers as the run's headline heldout metrics
   is misleading.

3. **`scripts/_bpe_probe.py:100-104, 115-119, 128` — `.npy` row caches keyed only by file basename + CTX silently serve stale data.**
   `bpe_{stem}_ctx{CTX}_rows.npy` / `bpe_rows_ctx{CTX}.npy` are reused whenever the cached
   array is long enough, with no mtime/hash/size check of the source file. Regenerating
   `UNITS_FILE` with a different `SEED`/`N_OUT` under the same name (the workflow
   `_teacher_gen.py` is built around) makes every subsequent train/eval silently consume the
   old rows. Different paths with the same basename also collide. `np.save` is non-atomic, so
   a crash or concurrent run can leave a torn cache on the shared artifact dir.

4. **`scripts/_teacher_gen.py:384-388` — stats filename derivation can overwrite the units file it just wrote.**
   `OUT_NAME.replace(".txt", "_stats.json")` is a no-op when `OUT_NAME` lacks a `.txt`
   substring; the stats JSON is then written to `OUT_NAME` itself, silently clobbering the
   freshly generated corpus file with a small JSON. Conditional on env input, but total loss
   of the run's output when triggered. (Adjacent module `_func_units.py:98` carries the
   identical latent bug.)

### Minor

5. **`scripts/_eval_deep256.py:29` — top-p filter is a wider nucleus than standard top-p.**
   `drop = cum - F.softmax(s,-1)[:, :1] > top_p` drops token i only when
   `cum[i] > top_p + p_top`; standard nucleus drops when `cum[i] > top_p` (keeping top-1).
   With a peaked distribution (p_top large) the effective threshold can exceed 1.0 and keep
   the whole tail, so the `"topp09"` label understates the kept probability mass and the
   topk-vs-topp comparison is skewed.

6. **`scripts/_eval_deep256.py:45` — `breakdown` misclassifies string literals as docstrings.**
   `tok.start[1] in (0, 4)` counts any STRING token whose quote begins at column 0 or 4 as a
   docstring — e.g. a top-level `x = "..."` (quote at column 4) is counted as docstring,
   inflating `doc_mean`.

7. **`scripts/_bpe_probe.py:108-109` + `_func_units.py:17` — off-by-one between the unit-length filter and the row layout silently drops a token.**
   `_func_units` admits units up to `MAX_TOK=127`; `load_rows` keeps `ids[:CTX-2]` = 126 and
   appends EOS, so a certified 127-token unit loses its last token with no truncation stat or
   warning (no counter exists for truncated units at all).

8. **`scripts/_bpe_probe.py:187-191, 248` — no validation that enough rows were loaded.**
   If `len(rows) < HOLDOUT[1]` (e.g. a units file smaller than 20000 units — `N_OUT` default
   is 4000), `held` is empty/short; empty gives a cryptic `ZeroDivisionError` at
   `hce = ce_sum / tokens`, short gives an honest-but-easy-to-miss reduced heldout.

9. **`scripts/_eval_deep256.py:17, 72-74` — argmax rows are 8 identical samples; seed ≥ CTX silently degenerates.**
   Argmax is deterministic, so `n=8` produces 8 copies of one sample and `parse p_/8` reports
   8× a single draw (wasted compute, misleading denominator). And if `len(seed_ids) >= CTX`,
   `range(P.CTX - len(seed_ids))` is empty and the function returns the unmodified prompt as
   all rows — silent degenerate metrics.

10. **Non-atomic writes and silent overwrites on the shared artifact dir across all three files.**
    `_teacher_gen.py:384-388`, `_bpe_probe.py:267` (default `bpe_probe_metrics.json` clobbered
    every run), `_bpe_probe.py:113,128` (`np.save`), `_eval_deep256.py:80,99,102` —
    all write in place with no tmp+rename. `_bpe_probe.py:231-232` demonstrates the correct
    `os.replace` pattern for the checkpoint but it isn't applied to metrics/samples/caches.
    Concurrent runs or readers can see partial files; repeated runs silently replace prior
    metrics.

11. **`scripts/_bpe_probe.py:62-63, 201-207` / `scripts/_eval_deep256.py:62` — `weights_only=False` torch.load on a shared artifact dir.**
    Full-pickle deserialization of files anyone on the shared FS could replace. Repo
    convention matches (`ncm_codetrain.py:149`), but `weights_only=True` suffices for
    `{"model": state_dict}` payloads.

12. **`scripts/_teacher_gen.py:73, 78-81, 88-91` — generated units reference free variables unrelated to declared parameters.**
    `fam_validator` renames the `(x)` signature param to a random noun while the body
    references independently sampled nouns (`def is_empty(data): return len(items) > 3`);
    `fam_lookup` iterates a different noun than its parameter; `fam_guard` checks NAMEPOOL
    names that are usually not among `params`. Parse-valid per the stated contract, but the
    teacher corpus systematically teaches bodies that don't bind their parameters.

13. **`scripts/_teacher_gen.py:363,383` — silent underdelivery on dedup exhaustion.**
    The `tries < N_OUT*20` cap can end the loop with `kept < target`; only the stats JSON
    records the shortfall, no warning or nonzero exit.

14. **`scripts/_bpe_probe.py:198-207, 231` — eval-mode/config edge cases.**
    `INIT` load is silently overwritten by the `EVAL_ONLY` CKPT load when both are set
    (order-dependent precedence, no warning). The final ≤`STEP_LOG`−1 steps of a budget-ended
    training run are never checkpointed (save happens only inside the log-step branch).
    `AdamW` is constructed even in `eval_only` mode (harmless, wasteful).

### Nit

15. `scripts/_teacher_gen.py:44-46` — dead code: `tail` ternary (with the odd
    `isinstance(acc, list) is False` expression) is computed then unconditionally overwritten
    by `tail = "return acc"`; `body = f"acc {op}= x" if op != "*" else "acc *= x"` is a
    redundant conditional (both branches produce `"acc *= x"`).
16. `scripts/_teacher_gen.py:209` — `path` built but never used.
17. `scripts/_teacher_gen.py:367` — `"\n\n"`-rejections counted as `parse_fail` (mislabeled
    stat; the check itself is correct — it protects the `\n\n` unit separator relied on by
    `_bpe_probe.load_rows` and `_func_units` "LESSONS format"). Docstring-only rejections
    (374-377) are counted in no stat.
18. `scripts/_bpe_probe.py:16` — unused `pickle`, `sys` imports.
19. `scripts/_bpe_probe.py:154-167` — `eval_gen` parses each text twice (`ast.parse` then
    `n_real` re-parses); `filler_fraction` returns 0.0 on tokenize failure, biasing filler
    downward exactly on unparseable outputs.
20. `scripts/_eval_deep256.py:12` — `gen(model, fast, ...)` never uses `fast`;
    `F.softmax(s, -1)` computed twice at line 29; hardcoded `/8`, `/32` in prints (76, 97)
    will lie if n/N_EVAL change.
21. All three files leak file handles via `json.dump(open(...))` / `open(f).read()` without
    context managers (fine on CPython refcounting, still bad practice); no explicit text
    encoding anywhere — adjacent `_func_units.py:63` uses `encoding="utf-8", errors="replace"`.
    Python 3.11 under a C/POSIX container locale usually auto-enables UTF-8 mode (PEP 540), so
    impact is low, but it is inconsistent.
22. `scripts/_bpe_probe.py:140` — `fast.id_to_token(i) or ""` silently drops out-of-vocab ids
    instead of flagging them.
23. Hardcoded `/repo/...` and `/work/...` paths plus `sys.path.insert(0, "/repo/scripts")`
    (`_eval_deep256.py:6`) — container-bound by design; noted for portability completeness.
    No other Python-3.11 portability issues found (ast/tokenize/torch APIs used are stable).

### Trajectory (events.jsonl scan)

- `RUN-047.events.jsonl`: begin → issue_create → task_plan → execute → test(nc:0) → history.
  The ~14.7 h `task_plan`→`execute` window is consistent with the run's +6 h warm-start
  training leg plus deep eval — no anomaly.
- `t:"test"` reports `nc:0` with no attached tool-call evidence and passed in 11 s — thin
  pass signal relative to a compute-heavy deliverable; noted, not provable as a failure.
- No `blind_review`/`security_gate` events yet — expected; this review is the emitted
  `blind_review` step.

## Patterns

- **Shared-artifact-dir writes are unprotected almost everywhere** (findings 3, 4, 10):
  non-atomic `np.save`/`json.dump(open(...,"w"))` and content-blind caches on
  `/work/artifacts/ncm_code`, while the atomic `os.replace` pattern is already used once for
  the checkpoint. Root cause: no write-through-tmp convention for artifacts.
- **Metrics are computed over raw layouts without accounting for special-token structure**
  (findings 1, 2): pad/EOS tokens are treated as ordinary supervised positions and ordinary
  decoded text, inflating or polluting reported numbers — the repo's own `_overfit_test.py`
  already shows the masked convention.

## Verdict

No blockers; four majors. The functional pipeline is coherent (BOS/EOS layout verified
consistent with the trained tokenizer; causal mask, loss shift, dedup, and `\n\n`-separator
guards all correct), but the eval story has real defects: `decode()` cannot reconstruct
non-ASCII text and mis-maps two byte ranges, pad positions inflate the headline heldout
metrics, basename-keyed caches can silently serve stale training data, and the stats-path
derivation can overwrite its own output under a non-`.txt` `OUT_NAME`. Findings 1–4 should
be fixed before the produced metrics/samples are trusted; the rest are hardening and hygiene.
