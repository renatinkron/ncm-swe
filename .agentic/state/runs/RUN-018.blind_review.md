# Blind Review — RUN-018

Cold review of `src/ncm_swe/model/generator.py`, `scripts/train_gen_official.py`,
`scripts/bench_official.py`; adjacent modules read: `vendor/ncm-upstream/models/quant.py`,
`vendor/ncm-upstream/models/vqvae.py`, `vendor/ncm-upstream/config/eval/*.yaml`,
`src/ncm_swe/model/common.py`, `warmstart.py`, `scripts/encode_official.py`.

## Findings

| file | line | severity | issue |
|---|---|---|---|
| scripts/bench_official.py + .agentic/state/runs/RUN-018.task_plan.md | 27-28 / task_plan:17 | **major** | Env interface can silently void the deferred 13h verdict. `GEN` takes a **file** path while `OUT_DIR`/`BENCH_OUT` take different granularities; nothing chains `OUT_DIR` → `GEN`. Two failure modes: (a) the task plan literally records `GEN=gen_ncm_v2` — a *directory* shorthand; if the armed chain passes that, `torch.load(GEN)` raises at bench time (and `os.chdir(ROOT)` at line 20 resolves relative paths inside `/repo/vendor/ncm-upstream`, so a bare name lands nowhere); (b) if `GEN` is left unset, the default `/work/artifacts/gen_ncm_official/final.pt` **exists** (RUN-017's scrambled-vqvae checkpoint) and bench silently evaluates the *old* model, writing `bench_v2.json` — a wrong verdict recorded as the deferred answer. No existence/format check guards GEN before the wait. |
| scripts/train_gen_official.py | 31, 68, 84-98 | **major** | `OUT_DIR` default is the *previous run's live output dir*. If the env is missing or misspelled — note the change description itself says "from env `OUT`" while the code reads `OUT_DIR`, so `OUT=gen_ncm_v2` would be **silently ignored** — the retrain appends to `gen_ncm_official/metrics.jsonl` (mode `"a"`, no run marker, line 68) and overwrites `final.pt`/`ckpt.pt` (lines 84-98), contaminating the baseline the bench compares against. A sane default would fail or require explicit `OUT_DIR`; today env-less reruns silently clobber. Verify the in-flight `ncm_train_v2` container actually received `OUT_DIR` (not `OUT`). |
| src/ncm_swe/model/generator.py | 89-91 | minor | Protection is positional, not structural. `self.vqvae` after `self.apply` is correct — `apply` only visits modules registered at call time, no generator submodule shares storage with vqvae (fresh `proj_in`/`head`; head is *not* tied to codebook) — and `load_state_dict` is key-based so old/new checkpoints interchange cleanly. But vqvae is still a registered child: **any future `model.apply(fn)` on the constructed generator recurses into the tokenizer and can re-scramble it**. No current caller does this (`warmstart.py` copies into `blocks` directly; scripts never call `apply`), but the invariant "init never touches vqvae" exists only as ctor ordering invisible to later code. `object.__setattr__`-style exclusion (or a comment contract in the module) would make it structural. |
| src/ncm_swe/model/generator.py | 145-146 | minor | "official: skip_level0" provenance is thinner than the comment claims. `skip_level0: true` exists only in `vendor/.../config/eval/lm1b-{small,medium,large}.yaml:124`; `tinystories.yaml` — the config both scripts actually load — has **no `code_corruption` block** (its `aux` block, lines 92-95, holds only `fine_drop`). So `--corrupt 1.0` transplants the lm1b recipe into a tinystories run. Worse, `code_corruption` sits under `vector_quantizer_config.aux` and **no consumer exists in the vendored tree** (grep: config-only hits) — whether upstream applied it to generator conditioning or tokenizer training is unverifiable from this checkout. Semantically the skip is defensible: level-0 codes seed z_hat for every later level (`idx_to_ncm_input`, quant.py:679-684 via `learned_up_l1`), so skipping l0 does leave downstream levels trained only on ground-truth l0 while generate() conditions them on *sampled* l0 — residual exposure bias, but it is exactly what the official flag encodes, and our q0-head l0 sampling approximates the official `level0_unigram` (q0 attends only to itself ≈ learned marginal). Report as design-faithful with a provenance gap, not a bug. |
| src/ncm_swe/model/generator.py | 158-162, 117-124 | minor | `_intra_add` codebook lookups run **outside** `torch.no_grad()`: the `no_grad` block covers only `idx_to_ncm_input` (158-160); `vq.embeddings[i+1](codes_i)` (124) executes with grad enabled. Safe today only because `train_gen_official.py:52-53` freezes vqvae and `bench_official.py` (which does *not* freeze, line 45) runs intra=False under `@torch.no_grad()`. An unfrozen vqvae on an intra training path would silently train the codebooks. Cheap fix: wrap the `ce` lookup in `no_grad` (mask_emb grads flow through `torch.where` regardless). |
| scripts/train_gen_official.py | 39 + generator.py:143 | minor | `--corrupt` is unvalidated: `argparse` accepts any float; `rates = rand * self.corrupt` with corrupt>1 gives rate>1 → deterministic total corruption of all conditioning (mask always true), corrupt<0 → none. Either is accepted silently. |
| src/ncm_swe/model/generator.py | 145, 91 | minor | Comment hygiene: line 145 ends mid-sentence — `— l0 is` and nothing; line 91 says "registered post-init" but means post-`apply` (it is still inside `__init__` — the distinction is the whole point of the fix). |
| scripts/bench_official.py | 47, 64 | minor | `json.dump(res, open(OUT, "w"))` — file object never closed (CPython refcount saves it, still sloppy) and silently overwrites any existing bench JSON; `torch.load(GEN, weights_only=False)` executes arbitrary pickle on an env-controlled path (container-scoped, acceptable but worth noting). |
| src/ncm_swe/model + scripts | — | minor | Carried-open RUN-017 items still unfixed in the touched code: checkpoint `config` stores only `{n_embd, n_layer, corrupt, levels}` — bench rebuilds arch from ctor defaults (`n_head_kv`/`mlp_hidden`/`shared_head`/`intra`) and never validates saved `levels` vs `vqvae.vq.levels`; `state_dict` embeds the entire tokenizer (bloat — and pre-fix checkpoints carry *scrambled* vqvae weights with no marker distinguishing them); held eval still runs under corruption so `held_loss` is the corrupted objective; `generate()` mask0 `torch.zeros(1,1,1,1)` is CPU-only. |

## Trajectory (`.agentic/state/runs/RUN-018.events.jsonl`)

- Events 5–20: **six consecutive `execute`→`test` cycles all reporting `nc=3`** in ~9 min, then `history`. Run-state explains nc=3 as "artifacts pending" (the 14h train is a wall-clock block, not a defect) and the compliance report closes at nc=0 after task-plan amendment. Process signal: the tester re-reported the same wall-clock wait six times instead of escalating once — loop burn on an unresolvable count; the escalation was correct but late.
- `RUN-018.json` `overall_status` remains `"planning"` while `history` holds a `"final"` entry and `summary` describes an in-flight run — bookkeeping drift between the state fields and the history trail.
- Neither is a code defect; both are process noise a spec-aware pass would also see, listed per the skill's trajectory step.

## Patterns

The two majors plus minor #8 share one root cause: **the container scripts grew ad-hoc env-var interfaces** (`CORPUS`, `SHARDS`, `OUT_DIR`, `GEN`, `BENCH_OUT`, `THREADS` — different names, different granularity, no chaining, defaults pointing at live previous-run artifacts, no pre-flight validation). Any single misspelled or unset variable degrades to silent reuse/overwrite of the prior run's outputs rather than a loud failure.

## Verdict

The generator fix itself is correct and complete for the reported defect: registering `vqvae` after `self.apply` puts it outside `apply`'s traversal with no shared-storage leakage, `load_state_dict` is unaffected by attribute order so old checkpoints still load, and `skip_level0` matches the official `lm1b` flag while level-0's z_hat contribution stays clean by design — the residual train/generate exposure bias on l0 is inherent to that recipe, not introduced by this diff. The real residual risk lives in the scripts: the env-var defaults point at live artifacts from the previous run, so a missing/misspelled `OUT_DIR` silently contaminates `gen_ncm_official` and a missing/shorthand `GEN` either crashes the armed bench or — worse — silently benches the stale scrambled-vqvae checkpoint and records it as the deferred verdict. Verify the in-flight container's env and make the bench fail loudly when `GEN` isn't an existing file.

blocker: 0, major: 2, minor: 7

## Disposition (orchestrator)
- Major 1 (GEN path): VERIFIED non-defect — armed chain passes
  GEN=/work/artifacts/gen_ncm_v2/final.pt (file). Hardened anyway:
  bench_official.py now asserts isfile(GEN) — fails loudly instead of
  benching stale ckpt.
- Major 2 (OUT_DIR): VERIFIED non-defect — container env shows
  OUT_DIR=/work/artifacts/gen_ncm_v2; gen_ncm_official/metrics.jsonl
  untouched (mtime pre-launch). Doc shorthand corrected in task_plan.
- Minor truncated comment generator.py:145 — completed.
- Minors rest: logged, no action (bookkeeping).
disposition: majors_verified_non_defect, hardening_applied
