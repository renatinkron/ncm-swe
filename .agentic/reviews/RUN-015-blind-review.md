# Blind Review — RUN-015

## Findings

| file | line | severity | issue |
|---|---|---|---|
| scripts/chain_ts3.sh | 36 | minor | Diversity gate hardcodes `levels = [1,8,16,32,64,128]` instead of reading `levels` from `artifacts/codes_ts2/meta.json` (written by `encode_corpus.py:64`) or `configs/vqvae_ts2.yaml`. On any level-layout drift the column offsets silently desync — numpy column slicing past the array width yields truncated slices without error (`X[:, 248:376]` on a narrower array does not raise), so `uniq` counts are computed over wrong bands and the gate can silently pass a collapsed codebook — the exact failure it exists to catch. |
| scripts/level_ablation.py | 77, 89 | minor | `model.decoder(z)` called outside `torch.no_grad()` — `z_partial` is decorated but its caller's decoder passes are not. `model.eval()` does not clear `requires_grad` on decoder weights, so an autograd graph is built for each of the 6+ k-passes over (n=512, 128, vocab) logits. Memory/compute waste only; results unaffected. |
| scripts/level_ablation.py | 58-62 | minor | `tempfile.mkdtemp(prefix="abl_corpus_")` corpus copies are never removed — every `--shards` run leaks a copy of up to N shards in /tmp. No `shutil.rmtree`/atexit/`TemporaryDirectory` cleanup. |
| configs/vqvae_ts2.yaml | 20-22, 25, 27 | minor | The NOTE claims "Only `levels.manual_levels` and `vqvae_vocab_size` still apply" — wrong in both directions: `commitment_cost: 0.25` IS forwarded to MiniMSRQ (`build_vqvae` reads `vcfg.get("commitment_cost", 0.25)`, model.py:49) while `decay: 0.99` is NOT (MiniMSRQ takes its constructor default — same value today, silently ignored if ever changed). A reader editing either knob gets the opposite of what the comment promises. |
| scripts/level_ablation.py | 27 | nit | `torch.zeros(B, vq.L_full, vq.embed_dim)` drops `device=idx.device` present in the mirrored `idx_to_z_q` (mini_msrq.py:151). Harmless on this CPU-only pipeline; breaks if idx/model ever move off-CPU. |
| scripts/chain_ts3.sh | 12 | nit | Stale-artifact skip: a pre-existing `artifacts/tokenizer_ts2.pt` bypasses the entire wait loop including producer-liveness — the chain encodes a possibly stale tokenizer silently. Same behavior as reviewed chain_ts2.sh; kept for parity. |
| scripts/chain_ts3.sh | 14 | nit | `pgrep -f "train_vqvae.py.*ts2"` matches ANY process whose cmdline contains `ts2` after `train_vqvae.py` (e.g. a producer writing `tokenizer_ts2_bak.pt`) → false liveness keeps the chain waiting the full 6h on an artifact that will never arrive. No self-match: this script's own cmdline (`bash scripts/chain_ts3.sh`) contains no `train_vqvae.py`. |
| scripts/chain_ts3.sh | 35 | nit | Unguarded `np.load("artifacts/codes_ts2/train.npy")` — missing/corrupt file produces a raw traceback. Outcome is still correct (non-zero exit → `[ $? -ne 0 ] && exit 1` aborts), just ungraceful. |
| scripts/chain_ts3.sh | 28-30, 4-5 | nit | Naming drift: log is `encode_ts3.log` while artifacts are `codes_ts2`/`gen_ncm_ts2`; hardcoded `cd /home/kron/...` and `PY=$HOME/.venvs/ncm/bin/python` — inherited house pattern from chain_ts2.sh, noted not re-litigated. |
| scripts/level_ablation.py | 56, 68-70 | nit | `import tempfile, shutil` mid-function (style); degenerate input unhandled — empty corpus/`--shards` with no `shard-*.jsonl` matches yields `ho.shape[0]==0` → `n=0` → `model.encode` on an empty batch crashes rather than exiting with a clear message. |

## Patterns

No 3+ findings share a single root cause. Heaviest cluster is latent-desync
hardcoding (gate level list; yaml comment/impl mismatch) — both correct today,
both silently wrong on a future config change.

## Trajectory

`.agentic/state/runs/RUN-015.events.jsonl` scanned: every emitted step has a
matching outcome event (task_plan, issue_create, execute, test×4, remediate×2,
history). Two remediation loops recorded with nc counts 2→1→0; execute took
~10h wall consistent with real training. No self-reported pass lacking
evidence, no silent expects, no missing dispatches. Clean.

## Verdict

Code is fundamentally correct: `z_partial` offset bookkeeping advances `ofs`
over all levels (matching `idx_to_z_q` semantics, so k=K reproduces full
reconstruction), the watcher fixed chain_ts2's save-and-exit race via the
sleep-10 re-check, `save_vqvae`'s atomic rename means `test -s` cannot see a
partial checkpoint, flock covers the whole script, and pgrep cannot self-match.
No blockers or majors — the findings are latent-desync hardcodes (diversity
gate level list, yaml comment claiming `commitment_cost` inert), resource
hygiene (no_grad scope, /tmp leak), and inherited cosmetic issues.

findings: 10 (0 blocker, 0 major, 4 minor, 6 nit)
