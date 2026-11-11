# Blind Review — RUN-017

## Findings

### blocker

- **generator.py:87** — `self.apply(self._init_weights)` recursively reinitializes **every submodule, including `self.vqvae`** (registered as a child module at line 33, before `apply`). `_init_weights` (lines 91–97) overwrites all `nn.Linear` and `nn.Embedding` weights in the pretrained VQ-VAE with N(0,0.02): `vq.embeddings` (all 12 codebooks), `encoder.wte/wpe`, decoder `c_attn/c_proj/c_fc/c_proj/output_layer`. Both call sites build the generator **after** loading the checkpoint (`train_gen_official.py:50-59`, `bench_official.py:44-51`), and `load_state_dict` in bench restores the *scrambled* weights saved at train time — nothing ever reloads the true tokenizer. Observable consequences match the run's own symptoms: (a) `idx_to_ncm_input` becomes a fixed random projection of codes (identity-preserving → explains why fine levels still learned); (b) `vqvae.decode`/`decode_text` produce argmax of a random output layer → token salad from the BPE vocab **regardless of generator quality** (8/8 word-salad is guaranteed by this alone). The "l0 bottleneck" verdict is confounded — decode could not have produced coherent text even with a perfect generator.

### major

- **generator.py:239-264, 300-304** — dead compute in `generate()` (non-intra): each pass for level *k* runs `_forward_partial` over q0 + chunks for levels 1..k, but `attn_mask` is block-diagonal (lines 79-85) so positions of the last chunk attend only to q0 (whose output is constant — it attends only itself) and their own block. Hidden states of earlier chunks are computed and then provably unused; only `h[:, -Lk:]` is read. Per-pass cost is `1+sum(levels[1..k])` (≈2567 position-passes total for the official levels) vs `1+L_k` minimal (≈651) — ~4× wasted transformer compute on every sample. This materially inflates `s_per_sample` and the "slower than AR" bench conclusion.
- **generator.py:143** — `_corrupt_codes` corrupts level-0 codes (loop `for j, L in enumerate(self.levels)` starts at j=0). The referenced official recipe sets `code_corruption.skip_level0: true` (vendor `config/eval/lm1b-*.yaml:124`); the docstring claims "official-NCM-style". Level-0 is the most influential conditioning code — it seeds the z_hat accumulation that feeds *every* chunk's input, so ~50% mean corruption there degrades all levels' conditioning, unlike other levels which affect only later chunks.

### minor

- **generator.py:289** — `mask0 = torch.zeros(1, 1, 1, 1)` hardcodes CPU; `blk(h, mask0, ...)` raises device mismatch if the model is ever moved to CUDA. Rest of `generate` uses `self.q0.device`.
- **generator.py:287-294** — `generate` structurally assumes `levels[0] == 1` (single q0 position emits the whole level-0). For `levels[0] > 1`, `codes[0]` would be `(B,1)` and `_chunks_from_codes`/`decode` would misalign; no assert guards it.
- **train_gen_official.py:84-88,94-98 + bench_official.py:48-51** — checkpoint `config` saves only `{n_embd, n_layer, corrupt, levels}`; bench reconstructs the model relying on ctor defaults for `n_head_q/n_head_kv/mlp_hidden/shared_head/intra`. Consistent today; a future non-default training config silently produces an arch that only fails at `load_state_dict` (loud, but late). Also `levels` is saved but never validated against `vqvae.vq.levels`.
- **generator.py:33** — `self.vqvae` registered as a submodule means `state_dict()` embeds the entire frozen VQ-VAE (encoder+decoder+codebooks+EMA buffers) in every `ckpt.pt`/`final.pt` — large duplicated payload; also the mechanism that made the blocker possible.
- **encode_official.py:30,63,98** — hardcoded `CTX = 128`, `(N,641)`, `"vocab": [4096]*len(lvls)` duplicate `vqvae.context_length`/`sum(vq.levels)`/`vqvae_vocab_sizes` instead of reading them. Today all three match (config 128 / Σ=641 / 12×4096); a config swap fails silently for CTX (wpe only errors when CTX > ctx_len).
- **encode_official.py:103, bench_official.py:64** — `json.dump(..., open(path, "w"))` leaves the handle unclosed (GC-dependent); **train_gen_official.py:68** `mlog` unclosed on exception.
- **train_gen_official.py:92-93** — held eval is `model.loss` under `corrupt=1.0` on 64 random items: it measures the corrupted-conditioning objective (not a clean teacher-forced loss) on a small noisy subset; the `held_loss` label invites wrong comparisons.
- **encode_official.py:37** — `shard.read_text().splitlines()` loads whole shards into memory; bounded at TinyStories scale, fragile at larger.

### trajectory

- **RUN-017.events.jsonl:2** — `t=issue_create` emitted but `run_state.github_issue` records `number: null, status: "unavailable"` — the issue-lifecycle gate logged an event with no artifact created; silent unavailability.
- Iterations 1→3 show plausible tool-call cadence (execute ≈40min/≈2h/≈13.5h, test nc 3→2→0 with remediate between); no step self-reports "passed" without supporting artifacts. No other anomalies.

## Patterns

- **Config values duplicated by hand instead of read from the loaded object** — `CTX=128`, `641`, `[4096]*12`, partial ckpt config (4 findings): each is correct today and silently wrong on any config change.
- Unclosed file handles appear 3× across the scripts — cosmetic at this scale.

## Verdict

One blocker invalidates the run's central conclusion: `LevelGenerator.__init__` calls `self.apply(self._init_weights)` *after* registering the pretrained `vqvae` as a submodule, re-randomizing every Linear/Embedding in the tokenizer — codebooks included. `idx_to_ncm_input` degrades to a fixed random projection (identity-preserving, which is why fine levels still trained), and `decode` is guaranteed word-salad, so the 8/8 failure and the "l0 bottleneck" diagnosis cannot be trusted. Additionally, `generate()` wastes ~4× transformer compute on provably dead earlier-chunk hidden states (the block-diagonal mask makes them unreachable), inflating the reported latency, and `_corrupt_codes` deviates from the referenced recipe by corrupting level-0. Minors are hardcoded config constants, CPU-only mask, `levels[0]==1` assumption, checkpoint bloat, and small resource/eval-hygiene nits. No label leakage and no RNG/determinism bugs found: targets stay clean, intra boundaries mask correctly, and all RNG is seeded.
