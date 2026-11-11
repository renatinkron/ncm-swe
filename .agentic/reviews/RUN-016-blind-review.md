# Blind Review — RUN-016

## Findings

| file | line | severity | issue |
|---|---|---|---|
| src/ncm_swe/model/generator.py | 211-213 | major | Train/generate conditioning mismatch: `_chunks_from_codes` downsamples `z_hat` with `F.interpolate(mode="area")` (mirrors vendored `quant.py:685-691`), but the training path uses `vq.idx_to_ncm_input`, which for MiniMSRQ (`quantizer_impl: mini`, all active configs) is `self._interp` = `mode="linear", align_corners=False` (mini_msrq.py:52-55,171). For 1-D downsampling these agree only at factor 2 and identity; at factor >=4, linear averages only the center-2 elements of each block while area averages the whole block. With levels [1,8,16,32,64,128], chunks predicting levels 8/16/32 (factors 16/8/4) are computed differently in generate vs train — every `generate()` call, intra AND parallel (`_forward_partial` is shared), feeds out-of-distribution inputs on 3 of 5 chunks. Silent quality degradation on all sampled output; plausible contributor to the "word-salad" bench outcome. |
| scripts/bench_generate.py | 37-44 | minor | `load_gen` omits `shared_head=cfg.get("shared_head", True)` — a ckpt trained with `shared_head: false` reconstructs `head` as a single Linear and `load_state_dict` fails on unexpected `head.{i}.weight` keys. Latent today (all generator configs use shared_head:true) but the reconstruction already forwards every other knob including the new intra flags. |
| scripts/bench_generate.py | 36 | minor | `load_gen` reads `ck["config"]`, but `save_ckpt` (checkpoint.py:17) serializes the config under `"cfg"` while `final.pt` (train_generator.py:152) uses `"config"` — benching a resumable `ckpt.pt` (written every `save_every` steps) crashes with KeyError. Two checkpoint schemas for the same artifact family; only final.pt works. |
| scripts/train_generator.py | 66-67 | minor | `cfg["intra"]`/`cfg["intra_passes"]` injected unconditionally from argparse defaults: a yaml-provided `intra: true`/`intra_passes` value is silently overridden (`store_true` can only turn on; default 32 stomps any config value — intra cannot be enabled via config file at all), and AR-baseline ckpts persist meaningless `intra`/`intra_passes` keys. |
| src/ncm_swe/model/generator.py | 216-241, 289-297 | minor | `_forward_partial` recomputes the entire prefix every intra pass, but the block-diagonal mask means earlier chunks can never influence level-k logits — their hidden states are pure dead compute (~2.5x position-forwards vs running [q0]+last-chunk with offset pos/rope indices). In a bench whose purpose is measuring NCM-vs-AR speed, this inflates NCM wall time and understates the reported speedup. |
| src/ncm_swe/model/generator.py | 57-59 | nit | `mask_emb[0]` is allocated but never read — level 0 is predicted from q0 and has no chunk (`_intra_add`/`_forward_partial` only index 1..K-1). Dead parameter that lands in the optimizer and every intra state_dict. |
| src/ncm_swe/model/generator.py | 34,289 | nit | `intra_passes` is a per-pass chunk WIDTH (positions revealed per forward), not a pass count — `--intra-passes 8` on L=128 yields 16 passes. Also `intra_passes <= 0` produces `range(0, Lk, 0)` ValueError mid-generate; no validation in ctor or CLI. |
| src/ncm_swe/model/generator.py | 266 | nit | `mask0 = torch.zeros(1,1,1,1)` pins CPU while the rest of the device state rides buffers/`q0.device` — same latent-device class flagged on level_ablation.py in RUN-015. Harmless on this CPU-only pipeline. |
| src/ncm_swe/model/generator.py | 165-168, 270-271 | nit | Level-0 paths hard-assume `levels[0]==1`: `loss` reshapes `logits_l0` (B,1,V) against `t0` (B*l0) → shape crash if l0>1; `generate` emits (B,1) level-0 codes → wrong `idx` layout. All current configs use l0=1 — latent. |
| scripts/bench_generate.py | 90 | nit | `rss0 = rss_mb()` assigned, never used — docstring claims "delta via /proc" but the report stores absolute peak RSS. Dead variable. |
| src/ncm_swe/model/generator.py | 117-124, 193 | nit | `arange_cache` dict rebuilt per `loss()` call (per-L arange could be a buffer); `_chunks_from_codes`' `upto` parameter equals `len(codes_by_level)` at every call site — redundant arg. |
| scripts/chain_ts3.sh | 14,35 | nit | Parity nits carried from RUN-015 (not re-litigated): `pgrep -f "train_vqvae.py.*ts2"` over-matches any ts2-suffixed producer; a pre-existing `tokenizer_ts2.pt` skips the liveness loop entirely; unguarded `np.load`/`json.load` in the gate → raw traceback on missing/corrupt files (outcome still correct: non-zero exit aborts). The RUN-015 majors are fixed: gate now reads `levels` from meta.json (matches encode_corpus.py:64) and the sleep-10+retest liveness recheck closes the save-and-exit race; `save_vqvae`'s atomic rename keeps `test -s` safe. |
| src/ncm_swe/model/generator.py | 109-110 | nit | `_intra_add` docstring claims `boundaries=None` "matches pure-parallel" — true only while `mask_emb` is at its zeros init; once trained, all-pending adds a learned nonzero vector. Semantics are right for intra mode; the comment overstates equivalence to a non-intra model. |

## Patterns

Checkpoint/config serialization drift — 3 findings share one root cause: the
train→ckpt→reconstruct contract is re-implemented by hand in every loader.
`bench_generate.load_gen` drops `shared_head`, reads `"config"` while
`save_ckpt` writes `"cfg"`, and `train_generator` stomps yaml keys with
argparse defaults. Same desync family as RUN-015's gate/comment hardcodes:
correct today, silently wrong on the next knob change. A single
"model_kwargs_from_cfg" helper would collapse all three.

## Trajectory

`.agentic/state/runs/RUN-016.events.jsonl` scanned: every emitted step has a
matching outcome event — task_plan, issue_create (initial validator rejection
recorded as a `note` with `step_failed`, then pass), execute (~4.5h wall,
consistent with real training), test nc=1, remediate, execute, test nc=0 with
the boundary-CE diagnostic artifact cited, history. No self-reported pass
lacking tool-call evidence, no silent expects, no absent dispatches. Clean.

## Verdict

Mask/leak semantics are sound: pending positions carry `mask_emb` (never the
true embedding — `where()` discards it), the block-diagonal mask isolates
level-k predictions from other chunks' intra conditioning so
`_forward_partial`'s last-chunk-only conditioning faithfully reproduces the
training-visible input, boundary convention (`arange < boundary` decided /
`>=` pending) is identical in train and generate, CE correctly excludes
decided positions (self-attention would make them trivial copies), per-level
boundary sampling is uniform 0..L with a div-zero clamp, pass counting is
ceil-based and correct (+1 for the q0 level-0 pass), and non-intra
checkpoints are untouched (`mask_emb` exists only under `intra`, cfg flag
round-trips through both save paths). One major: `_chunks_from_codes`
downsamples with `area` while MiniMSRQ training inputs use `linear` — every
generate call feeds out-of-distribution chunk inputs for levels below factor
2. The rest is config-plumbing drift, dead compute inflating the NCM side of
the bench, and latent-edge nits.

findings: 13 (0 blocker, 1 major, 4 minor, 8 nit)
