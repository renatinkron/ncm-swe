# Blind Review — RUN-024

Scope: `scripts/eval_ckpt_v3.sh`, `scripts/watch_v3.sh` (commit 299fbf9). Cold review; adjacent modules checked: `train_gen_official.py`, `sample_official.py`.

## Findings

### Major

| file:line | issue |
|---|---|
| eval_ckpt_v3.sh:24 + watch_v3.sh:24-26 | **Torn-snapshot race.** Trainer `torch.save`s `ckpt.pt` in place every ~500 steps (train_gen_official.py:102-110 — no tmp+rename). A `cp` racing a save copies a half-written zip → `torch.load` crash (sample_official.py:137) or silently bad weights. The watcher's `sleep 30` comment claims it "confirma que metrics avançou além do save" — the code never re-reads metrics; the guard exists only in the comment. And the ckpt dict has no `step` key (lines 102-109), so label↔content can never be verified post-hoc. Fix: poll `stat -c %Y,%s` until stable across two checks, or make the trainer save via tmp+rename; record actual step in the ckpt dict. |
| eval_ckpt_v3.sh:18, watch_v3.sh:22 | **Boundary math off by one save period at exact save steps.** `(LAST-3000)/500*500-1+3000` = `floor(local/500)*500-1`; when `local%500==499` (LAST=4499,4999,…) it returns the *previous* save (4499→3999). Standalone evals run at those moments are mislabeled 500 steps behind → eval JSON filed under the wrong step. Trigger depends on `log_every` (watcher flow is immune since LABEL overrides, but the formula is wrong). Fix: `CKPT_STEP=$(( (LAST-3000+1)/500*500 - 1 + 3000 ))`. |
| watch_v3.sh:15 → eval_ckpt_v3.sh:11 | **"final" eval never reads `final.pt`.** `eval_ckpt_v3.sh "final"` still snapshots `ckpt.pt`; `final.pt` (train_gen_official.py:118-125, saved once post-training) is never loaded. Correct only when `steps%500==0`; otherwise the headline "final" metric describes weights up to 499 steps stale, mislabeled as final. Fix: parametrize ckpt path — `eval_ckpt_v3.sh final artifacts/gen_ncm_v3/final.pt`. |

### Minor

| file:line | issue |
|---|---|
| eval_ckpt_v3.sh:23 | Dedup trusts file existence; `json.dump` is non-atomic (sample_official.py:190-191), so a killed run leaves truncated JSON reported "já avaliado" — and watch_v3.sh:27 then marks the boundary done. Corrupt eval permanently cemented. Fix: skip only if `json.load` succeeds, else re-eval. |
| watch_v3.sh:5-6,26 | No `set -euo pipefail` (eval has it); `cd` unchecked → a moved script silently polls the wrong dir for 60h; `scripts/eval_ckpt_v3.sh` depends on exec bit — if missing, every cycle fails and retries till MAXSEC. Fix: `set -euo pipefail`, `cd … \|\| exit 1`, call `bash scripts/…`. |
| eval_ckpt_v3.sh:19-21,38 | `LABEL` unsanitized → lands in `$SNAP`/`$OUT` paths (traversal: `label=../../x`) and inside `python3 -c '…open('$OUT')…'` quoting. Fix: `[[ $LABEL =~ ^[A-Za-z0-9_-]+$ ]]`. |
| eval_ckpt_v3.sh:27-28 | Redundant `-v $PWD/artifacts:/work/artifacts` (repo already at `/repo`); no `:Z` → SELinux denial on Fedora/podman hosts; stray `-i` with no stdin use. Fix: drop the second mount or add `:Z`; drop `-i`. |
| watch_v3.sh:30 | `MAXSEC` checked only *after* a full eval → overrun; persistent eval failure retries every ~150s for up to 60h (BOUNDARY stays > LAST_DONE) → log spam, no backoff/cap. Fix: check timer before eval; cap consecutive failures. |
| watch_v3.sh:19 | Non-numeric `LAST` (`"step":null` → `None`) makes `$((…))` error every cycle; without `set -e` it survives but spams. Fix: `[[ $LAST =~ ^[0-9]+$ ]] \|\| { sleep 120; continue; }`. |
| watch_v3.sh:14 | `LAST=` in the `final.pt` branch is dead — computed, never used. Fix: delete. |
| eval_ckpt_v3.sh:17-18, watch_v3.sh:22-23 | Magic constants duplicated: offset `3000` hardcoded in both files while trainer takes `args.offset`; `-ge 4499` bakes in this run's first-eval boundary. Offset change → silent mislabels everywhere. Fix: derive offset from metrics or one env var. |
| eval_ckpt_v3.sh:20 | Snapshots (`ckpt_step*.pt`, ~hundreds MB each) accumulate forever in `v3_eval`. Fix: retention policy or document. |
| eval_ckpt_v3.sh:17 | `tail -1` + python with no fallback — missing/empty metrics file dies cryptically under `pipefail` (watcher has `|| echo 0`, eval doesn't). Fix: same fallback or explicit error. |

## Patterns

Label-vs-content is never verified (findings 1–3): the boundary label comes from `metrics.jsonl`, the bytes are whatever `ckpt.pt` holds at copy time, and "final" reads a different file than its name claims — three findings share the root cause "artifact named by assumption, never checked."

## Trajectory

`RUN-024.events.jsonl` is consistent (7 events): one `task_plan` validation retry, then execute→test(nc:0)→history. No self-reported passes lacking tool evidence; nothing contradicting the artifacts.

## Verdict

No blockers; three majors, all silent-metric-corruption risks in a research pipeline: torn `cp` of a live-overwritten checkpoint, off-by-one boundary label, and a "final" eval that reads the wrong file. Each has a one-line-class fix; minors are hardening. Ship only after the majors.

Verdict: PASS_WITH_MAJOR_FINDINGS
