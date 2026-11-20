# Blind Review — RUN-038

Scope: `scripts/ncm_packtrain.py` (new file, 171 lines). Reviewed cold against
`src/ncm_swe/model/generator.py`, `vendor/ncm-upstream/models/{vqvae,quant}.py`,
`vendor/ncm-upstream/tokenization/{vqvae,bpe}_tokenizer.py`,
`config/eval/tinystories.yaml`, and sibling scripts
(`sample_official.py`, `ncm_official_probe.py`, `train_gen_official.py`).

## Verified good (explicitly checked, no defect)

- Checkpoint safety: `torch.load(GEN, map_location="cpu")` is the only access to
  `final.pt`; FT happens on an in-memory `LevelGenerator` copy; no `torch.save`
  anywhere — the "checkpoint never touched" docstring claim is accurate.
  `load_state_dict(strict=True)` makes arch drift fail loudly.
- Seed/sampling fairness pre vs post: `sample_and_score` re-seeds
  `torch.manual_seed(SEED)` + `np.random.seed(SEED)` at the top of *each* call
  (L110-111), so pre and post runs consume identical multinomial uniforms —
  paired common-random-numbers comparison isolating the weight delta. Correct.
- Decode path matches the official one (`idx_to_zhats` → `zl[-1]` → `decoder` →
  argmax), same as `sample_official.py:187-189` and `decode_multiscales`;
  `z_levels[-1]` is the final accumulated z_hat — safe despite upstream
  `z_hats` list aliasing the mutated tensor (quant.py:652).
- `codes` shape (N, 641) satisfies `loss()`'s `(B, sum(levels))` contract
  (levels 1+8+16+…+128 = 641); `uint16` cast is safe for V=4096.
- `model.eval()` inside `sample_and_score` re-disables the VQVAE dropout before
  the post-FT decode, undoing `model.train()` — post-FT decoding is
  deterministic given the seed.
- `loss()` return contract `(total, per_level)` handled via `out[0]` (L143-144).

## Findings

### major

- **scripts/ncm_packtrain.py:51-52,61** — `REGEX_INDICATORS` counts raw
  punctuation/prose substrings, not regex signals: `"re."` fires on any word
  ending "…re." ("there.", "more.", "where.", "sure."), `"?"` on any question,
  `"*"`, `"+"`, `"["`, `"^"`, `"$"` on ordinary punctuation/math. `mean_hits`
  (the headline metric) therefore has a nonzero noise floor on plain
  TinyStories-style prose; a run where pre samples contain "more." or a
  question inflates the baseline and can flip the sign of the delta. This run
  observed PRE=0.00 by luck, so the reported conclusion stands, but the metric
  is fragile — indicators should be restricted to regex-context matches
  (inside `r'…'` literals / `re.` calls) or the ambiguous ones dropped.

### minor

- **scripts/ncm_packtrain.py:60-73** — `mean_hits` is not normalized by text
  length; longer post-FT outputs mechanically collect more indicator hits.
  The score conflates "more regex content" with "more text".
- **scripts/ncm_packtrain.py:53,63-65** — `PAT_RE`/`pat` extraction: only the
  FIRST regex match's first non-empty group is compiled — an invalid leading
  `r"…"` suppresses later compilable patterns (false negatives);
  `re\.compile\(\s*[\"']([^\"']+)` truncates the capture at either quote type,
  so a pattern containing the other quote yields a truncated fragment that can
  compile spuriously or fail spuriously; non-raw `"…"`/`'…'` string literals
  are never matched at all → systematic undercount of `compilable`.
- **scripts/ncm_packtrain.py:89-93** — checkpoint `config` fields silently
  dropped on rebuild: `corrupt_schedule`, `corrupt_hi`, `corrupt_lo`,
  `intra_mask`, `intra_passes` are all persisted by
  `train_gen_official.py:124-131` but not read here (sibling
  `sample_official.py:150-157` does read `intra_mask`/`intra_passes`). With a
  checkpoint trained under `schedule='linear'` or `intra_mask='random'`, the
  FT loop silently runs a different regime than the weights were trained
  under; `strict=True` passes because these keys carry no parameters.
- **scripts/ncm_packtrain.py:140,147** — frozen-tokenizer boundary held only
  by accident: `opt = AdamW(model.parameters())` and `clip_grad_norm_` span the
  registered `model.vqvae` submodule. No grads flow today because
  `idx_to_ncm_input` runs under `torch.no_grad()` (generator.py:217-219) — but
  in `intra=True` checkpoints `_intra_add` embeds codes OUTSIDE that guard
  (generator.py:141-143,220-221), so `vq.embeddings` weights would be
  silently fine-tuned/decayed too. `model.train()` (L139) also recursively
  flips VQVAE `Dropout(0.3)` modules to train mode; benign only because the
  FT path never calls encoder/decoder. Freeze should be explicit
  (`requires_grad_(False)` on `vqvae`, `vqvae.eval()` pinned).
- **scripts/ncm_packtrain.py:101** — silent data loss: `encode(...)[:CTX]`
  truncates `probe + "\n" + ideal_answer` at 128 BPE tokens with no counter or
  warning; long cards train on truncated targets and there's no record of how
  many cards were cut.
- **scripts/ncm_packtrain.py:142-148** — FT loop is unseeded: only the two
  sampling calls re-seed. With `corrupt>0` in the ck config (train default is
  `--corrupt 1.0`), `loss()` draws `torch.rand`/`randint` corruption masks
  every epoch from an unseeded stream → post metrics vary run to run despite
  the SEED env knob. A `torch.manual_seed` before the loop would make the
  whole run reproducible.
- **scripts/ncm_packtrain.py:130,152,162** — late/edge crashes after all work
  is done: `N=0` → `sum(hits)/len(hits)` ZeroDivisionError; `EPOCHS=0` →
  `ce_first`/`ce_last` stay `None` → `round(None,4)` TypeError at metrics
  write. Env inputs are never validated.

### info

- **scripts/ncm_packtrain.py:87** — `weights_only=False` pickle load.
  Project-wide convention and an own artifact; flagged for completeness.
- **scripts/ncm_packtrain.py:33-36** — container paths `/repo`, `/work`
  hardcoded with no host guard and `os.chdir(ROOT)` mutates global cwd.
  Identical convention in every sibling script — consistent, but the run
  fails with an opaque ImportError outside the container.
- **scripts/ncm_packtrain.py:20-31** — `hydra` module stub shadows a real
  hydra install; deliberate and consistent with siblings; only
  `hydra.utils.get_class` is exercised.
- **scripts/ncm_packtrain.py:56-57** — `pick_tokens` is plain argmax; sibling
  `sample_official.py:56-91` carries the upstream anti-repetition port
  (freq/mu/adj). Identical decode for pre/post keeps the comparison fair;
  noted as convention drift.
- **scripts/ncm_packtrain.py:13** — `import time` unused.
- **scripts/ncm_packtrain.py:143-144** — `isinstance(out, (tuple,list))`
  defensive branch is dead code: `loss()` always returns a 2-tuple; the
  `per_level` dict it also returns is discarded instead of logged.

### trajectory (`.agentic/state/runs/RUN-038.events.jsonl`)

- `execute` passed 19:11:25 → `test` 19:12:52–19:13:02 heldout 7/7, nc=0.
  The container run is evidenced by durable artifacts (compliance report H1-H7,
  mtimes in-window); the tester itself did not re-run the container
  (479 MB ckpt) — graded `artifact`, consistent.
- Doc typo: execution_report.md:10 and SESSION_HANDOFF.md:327 both cite
  `metrics.jsonl`; the script writes `metrics.json`. Cosmetic.
- Run artifacts live under gitignored `artifacts/` (.gitignore:2) — expected,
  but they cannot be re-verified after the fact; the compliance report is the
  only persistent evidence of the numeric results. Info.

## Patterns

Two themes recur (3+ related findings):

- **Silent-divergence from checkpoint/vqvae boundaries**: dropped config
  fields (L89-93), optimizer scope spanning `model.vqvae` (L140), recursive
  `model.train()` flipping tokenizer dropout (L139), conditional grad leak via
  `_intra_add` (generator.py:141-143). The script leans on accidental safety
  (strict load, no_grad guards, eval() re-entry) rather than explicit freezes —
  each individually benign for gen_ncm_v2's likely config, collectively fragile.
- **Metric fragility**: substring indicators on punctuation (L51-52), no
  length normalization (L61), first-match-only pattern extraction (L63-65).
  The headline `mean_hits`/`compilable` numbers are the run's deliverable; all
  three findings weaken their interpretability the same way.

## Verdict

A compact, mostly correct container experiment script: the important
properties hold — checkpoint untouched, paired-seed pre/post sampling, official
decode path, loud failure on arch mismatch. The headline weakness is metric
validity: punctuation-level indicators and first-match compilable detection
make `regex_score` noisy in both directions (this run's 0.00→0.38 delta
survived by observation, not by design). Secondary theme: the "frozen"
tokenizer boundary and the checkpoint's saved training regime are honored
only accidentally — fine for this specific `final.pt`, silently wrong for
intra/non-flat-schedule checkpoints. Reproducibility gap: FT RNG unseeded.

major: 1, minor: 7
