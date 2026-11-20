# RUN-039 — Blind Review

Cold pass over changed/new files: `scripts/dec_boost.py`,
`scripts/_diag_channel.py`, `scripts/_eval_lowtemp.py`,
`scripts/ncm_kdistill.py`, `scripts/code_recon.py`,
`scripts/ncm_codetrain.py`, `scripts/teach_corpus.py`, `AGENTS.md`.

## Findings

### F1 (major, methodological) — `ncm_kdistill.py:127` scores chunks, not rows
`scores[i:i+32] = float(loss)` assigns the batch-MEAN loss of a 32-row
chunk to every row in it. The "top-loss selection" therefore selected
rows from high-mean chunks, not high-loss rows — a much coarser proxy
than the hypothesis under test. The negative result stands but must be
worded honestly: **top-chunk-mean selection refuted**; per-row-loss
selection remains untested (per-row scoring needs per-sample loss —
`F.cross_entropy(reduction='none')` or equivalent — ~32× more forward
work for the same coverage).

### F2 (minor) — eval split is the first N corpus rows in file order
`code_recon.py:98` and `dec_boost.py:95` both take `rows[:512]` as the
held-out set. Shards are glob-sorted, so eval rows likely come from one
source file — recovery may be biased by that file's style and not
represent corpus diversity. A shuffled or strided split would be more
representative. (Mitigates the 96.37-vs-93.33 cross-run comparison
concern only partially — same bias applies to both.)

### F3 (minor) — hardcoded `PAD_ID = 1` / EOS id `0` never asserted
`dec_boost.py:49`, `code_recon.py`, `_eval_lowtemp.py`, and
`_diag_channel.py` all assume token ids 0=EOS / 1=PAD without checking
`bpe.pad_token_id`/`eos_token_id`. If the code-BPE vocab assigns
different ids, `ignore_index` and EOS-truncation silently corrupt
metrics. One assert per script would close it.

### F4 (info) — cached `codes_code.npy` predates current emb state
Codes were encoded under the 96.37%-era embedding; `dec_boost`
continued tuning `emb.weight`, so the effective token→code mapping
drifted slightly. Generation is unaffected (no encode at inference), but
any future retrain on cached codes trains on a mildly stale encoding.
Re-encoding the corpus under the final emb is the clean fix if the
generator is retrained.

### F5 (info) — `0.25 * vq_loss` in dec_boost loss still reaches `emb.weight`
Commitment loss gradients flow through the frozen encoder's output into
the trainable embedding — a small, probably stabilizing pull toward
codebook-consistent latents. Not a bug, but it means the objective is
not pure reconstruction CE.

## Scope check
No changes outside contract scope. AGENTS.md rule addition is
documentation-only and consistent with the incident it records.

## Verdict
Core results unaffected: channel repair (65→93.3%), generator CE/parse
measurements, and the architecture benchmark stand on code paths not
touched by these findings. F1 requires only a wording correction in
run artifacts; F2/F3 recommend follow-up hardening, not rework.

non_conformities: 0 (findings documented; F1 wording fix applied to
execution_report/handoff wording: "top-chunk-mean", not "top-loss")
