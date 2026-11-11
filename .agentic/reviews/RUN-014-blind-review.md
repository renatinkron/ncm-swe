# Blind Review — RUN-014

Reviewed cold (no spec/task_plan): `src/ncm_swe/tokenizer/bpe.py`,
`configs/vqvae.yaml`, `configs/vqvae_ts.yaml`, `scripts/chain_ts.sh`,
`scripts/chain_ts2.sh`. Adjacent modules read for interface consistency only.

## Findings

| # | file | line | severity | issue |
|---|------|------|----------|-------|
| 1 | scripts/chain_ts.sh, scripts/chain_ts2.sh | 9 / 8 | **major** | Unguarded infinite wait: `while ! test -s artifacts/tokenizer_ts.pt; do sleep 60; done` has no timeout and no producer-liveness check. If the VQ-VAE trainer crashes without producing the artifact, the chain hangs forever with a single stale "waiting" line — a silent pipeline failure at the script's only real failure boundary. Add a max-wait and/or check the trainer process is still alive. |
| 2 | scripts/chain_ts.sh, scripts/chain_ts2.sh | 14-18+25 / 13-17+40 | **major** | v1 and v2 watchers share the identical trigger (`artifacts/tokenizer_ts.pt`) and identical outputs (`artifacts/codes_ts`, `artifacts/gen_ncm_ts`) with no lock/pidfile/sentinel. If the superseded v1 is still running when v2 starts, both fire on the same artifact: concurrent `np.save` to `train.npy` (np.save is not atomic → torn file possible) and two `train_generator` processes writing the same out dir (`metrics.jsonl` interleaves; `ckpt.pt` is last-writer-wins). Data-loss/nondeterminism risk from a plausible operational mistake. Also, either script re-run unconditionally re-encodes over `codes_ts` with no "already done" guard. |
| 3 | configs/vqvae_ts.yaml, configs/vqvae.yaml (+src/ncm_swe/vq/model.py:41-49, src/ncm_swe/vq/mini_msrq.py:24-25) | 22, 36-49 | **major** | Dead config under `quantizer_impl: mini`: `build_vqvae` forwards only `levels`, `vocab_size`, `embed_dim`, `commitment_cost` to `MiniMSRQ` — `decay` is a MiniMSRQ parameter that is never forwarded (config's `0.99` silently ignored; it only coincidentally equals the default), and the entire `quant_resi`/`learned_*`/`aux.fine_drop`/`aux.code_corruption` surface is dead weight. The two yamls differ exactly in these knobs (`fine_drop.prob` 1.0→0.5, `w_full`/`w_trunc`, `code_corruption.enabled` true→false), so any experiment intending to ablate them silently changes nothing. Silent failure of config to take effect. |
| 4 | scripts/chain_ts2.sh | 25 | minor | Diversity gate hardcodes `levels = [1,8,16,32,64,128]` while `encode_corpus.py` already writes the true value to `artifacts/codes_ts/meta.json` (`"levels"`, encode_corpus.py:64). If the tokenizer is rebuilt with the 8-level `vqvae.yaml` layout (`[1,8,16,32,48,64,96,128]`), the gate silently checks wrong column boundaries — could pass a collapsed model or kill a healthy run. Read `meta.json` instead of duplicating the constant. |
| 5 | scripts/chain_ts.sh, scripts/chain_ts2.sh | 6 / 5 | minor | Unguarded hardcoded `cd /home/kron/Documents/projects/ncm_swe` (no `|| exit 1`, no `set -e`). If the path is wrong/missing the script continues in the caller's cwd, `test -s artifacts/...` then polls a different tree — either an infinite wait or, worse, picking up an unrelated `artifacts/`. Portability: breaks on any other clone; `cd "$(dirname "$0")/.."` is the standard fix. |
| 6 | src/ncm_swe/tokenizer/bpe.py | 14-16, 56-58 | minor | `token_to_id("[PAD]"/"[EOS]"/"[BOS]")` returns `None` for missing tokens with no init-time validation; `pad()` would then emit `ids + [None]*n` — silent corruption that only surfaces far downstream as a cryptic numpy/torch dtype error. `build_segments` (segments.py:60-61) relies on `pad_token_id` too. Assert non-None in `__init__`. |
| 7 | configs/vqvae.yaml, configs/vqvae_ts.yaml | 1-3 | nit | `quantizer_impl: mini` is inserted between header comment lines, splitting the comment block and making the key look like part of it — easy to miss the fact that the quantizer implementation was swapped (which is what makes finding 3 surprising). Move below the comment. |
| 8 | src/ncm_swe/tokenizer/bpe.py | 70-72 | nit | `corpus_texts` does `shard.read_text()` — loads each whole shard into memory instead of streaming lines — and `json.loads(line)["text"]` raises bare `KeyError`/`JSONDecodeError` with no shard/line context on a malformed record. Same pattern exists in segments.py:17 (adjacent, not changed). |
| 9 | scripts/chain_ts.sh, scripts/chain_ts2.sh | 4, 5, 34 | nit | Shell hygiene: `set -u` only (no `-e`/`pipefail`), hardcoded `PY="$HOME/.venvs/ncm/bin/python"` (fails loudly, but non-portable — a `command -v` check or env override is cheap), and the `[ $? -ne 0 ] && exit 1` idiom in chain_ts2.sh:34 is brittle (returns nonzero on success path; would abort under any future `set -e`). `|| exit 1` matches the style used everywhere else in both files. |
| 10 | scripts/chain_ts2.sh | 24 | nit | `np.load("artifacts/codes_ts/train.npy")` loads the entire code array into RAM for a uniqueness check; `mmap_mode='r'` would bound memory on a large corpus. Fails closed if the file is missing/narrow (uniq→0 → gate abort), which is the right direction. |
| 11 | src/ncm_swe/vq/model.py (+vendor vqvae.py:50-51,75-77) | 21-50 | nit | Under `quantizer_impl: mini` the full upstream quantizer is still constructed inside `VQVAE.__init__` and then discarded — including `count_params`, so the printed "Total parameter count" reports the thrown-away upstream codebooks, not the model that actually trains/saves. Misleading log only; state_dict is consistent because replacement precedes checkpointing. |

## Trajectory

`.agentic/state/runs/RUN-014.events.jsonl` is coherent: `task_plan` →
three `execute`→`test(fail, nc 6→5→1)`→`remediate` cycles → `test(pass, nc 0)`
→ `history`. Early `note` about a missing execution_report artifact was logged
and the step re-ran. Skips carry reasons. No self-reported pass without a
verdict event, no dispatched-but-absent steps. Clean.

## Patterns

- **Watcher-stage robustness** (findings 1, 2, 4, 5, 9, 10): the chain scripts
  are built on file-existence polling with no timeouts, no mutual exclusion,
  and config constants duplicated into the script. The same
  trust-the-artifact-and-hope pattern recurs at every stage boundary.
- **Config keys that don't reach the code** (findings 3, 7): `quantizer_impl`
  silently narrows the effective config surface; dead knobs differ between the
  two yamls, and a MiniMSRQ parameter (`decay`) is never forwarded.

## Verdict

No blockers — the pipeline as wired plausibly runs and the diversity gate in
v2 fails closed. But the two watcher scripts share a trigger and outputs with
no exclusion (data-loss race if v1 is still alive), both wait forever on a
dead producer, and `quantizer_impl: mini` silently turns most of the
`vector_quantizer:` block — including the knobs that differentiate the two
configs — into a no-op. Secrets: none found.

findings: 11 (0 blocker, 3 major, 8 minor incl. 5 nit)
