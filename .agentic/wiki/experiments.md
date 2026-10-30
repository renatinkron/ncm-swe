# What we measured — the experiment record

Honest log of this framework's own benchmark runs. Verdicts were applied as
pre-registered — including the NO-GOs. **[measured]** = has data in
`.agentic/state/runs/` and benchmark ledgers.

## RUN-016 / RUN-017 — first ROI pilots: INCONCLUSIVE

- Measured proxies (bytes read, artifact sizes) — not provider tokens.
  Contamination between arms; meter disagreement (ATIF cumulative vs /usage).
- **Lesson**: freeze the protocol *before* running; count real tokens via
  ATIF `final_metrics`, not filesystem bytes.

## RUN-018 — trust/escalation mechanics

- Terse-mode promotion latch + failure drop to guided. Regression tests pinned.
- Foundation for the lean emission model.

## RUN-019 — operational fixes

- Classify-hook deadlock fixed: only commands *beginning* with `agentloop` are
  exempt; `cd ... && agentloop` was blocked. Run `agentloop` bare.
- **Lesson**: hooks are sequencing layers; keep them dumb and explicit.

## RUN-020 — lean mode: SHIPPED

Byte-level cuts with zero governance weakening (verified: no `expect`/
`validateExpects`/queue/latch edits in the diff):

- `CLAUDE.md`: 16 duplicated RTK blocks → 1 (**82,996 → 5,881 bytes, −93%**);
  embedded copy ships to every `agentloop init`.
- Run digest (~2KB) printed only on *newly emitted* steps, not failure reprints.
- `trustUpgradeAt` 3→1: terse after first clean validation.
- ~380B "Gate" preamble in 16 skill/agent files → one self-contained line.
- Read-only demands (questions/reviews/surveys) route to `fast` regardless of
  breadth — still logged to `fast_log.jsonl`.

In-flight catch: first digest-dedup design (comparing event counts) could
never fire — every `next` appends an event. Replaced with emit-on-new-step
rule during blind review.

## RUN-021 — reconciled benchmark: NO-GO

18 controlled sessions (AgroTrade @ `da87251`, SWE-2 Max, plain vs lean
framework arm, ATIF token accounting, frozen protocol, ABBA order):

| Demand | plain | fw lean | Δ cost_units |
|---|---:|---:|---:|
| D1 read-only survey | 510.5k | 467.6k | **−8.4%** |
| D2 small rename | 29.3k | 39.3k | **+34.1%** |
| D3 MCP feature | 144.9k | 107.0k | **−26.2%** |

Blind-judged quality (sum of 3 reps, max 18): fw scored *below* plain on all
three (15 vs 17, 15 vs 16, 16 vs 18) — lean outputs are more compact and the
judge rewarded completeness. Frozen rule required quality equal-or-better →
**NO-GO**, applied verbatim.

Takeaways **[measured]**:
- Governance overhead vanished on substantial tasks (was ~3× in RUN-017).
- On ~30k-unit tasks the governance floor (~10k) dominates — expected loss.
- Blind review caught *our own* truth-anchor error (D2) and framework worktree
  churn left in deliverables (D3-r3). Process finding its own bugs = working.

## RUN-022 — akita research → lean-overhaul spec

Full distillation in [working-with-ai.md](working-with-ai.md). Decisions:
adopt ai-memory instead of building our own memory layer; ai-jail as external
sandbox boundary; benchmark accounting in real USD; reposition product as
audit trail (the trail is the product), not quality/economics booster.

## RUN-023 — lean overhaul implementation: SHIPPED

- Queues: standard 11→8, complex 14→11. `branch_strategy`/`contract` merged
  into `task_plan` (validated via new `file_contains` expect); `docs` absorbed
  into `close`; `pentest` complex-only. Legacy defs retained — RUN-023 itself
  completed on a legacy queue, proving compat live.
- `bench_reconciled.py`: `cache_hit_ratio` always, `cost_usd` only when the
  model has an explicit price entry (never a guessed zero).
- `agentloop global status`: read-only `ai-memory`/`ai-jail` detection via
  `LookPath`, works even when agentloop isn't installed.

## Cross-run lessons

1. **The product is the trail, not smarter agents.** Every benchmark so far
   says governance is cost-neutral-to-cheaper on substantial work and does
   not measurably improve blind-judged quality. Adopt for auditability.
2. **Freeze protocols before measuring**; pre-register verdict rules.
3. **Judge rubrics matter**: completeness vs verbosity changes verdicts.
   Next benchmark: ≥2 judges, completeness-scored rubric, real USD.
4. **Small tasks should stay fast-path**; governance floor is real.

## RUN-033/034/037/038 — self-grading leakage, rework, and the literature

[measured] Cold re-grade (RUN-033): 3/55 criteria (5.5%) falsely self-passed —
pattern: criterion narrowing. Replicated in findue at 5.4% (RUN-034); condohub
showed the narrowing move *up* to criteria-vs-spec coverage + a 20-criteria
"unverifiable" class (runtime claims with no durable evidence).
[measured] Rework scan (RUN-037): 28–45% of runs across 3 projects were rework
of prior claimed deliveries; dominant class = premature done.
[attributed] Literature (NeurIPS/ACL/ICLR 2024-25): self-preference is causal
(Panickssery et al.), external verifiers beat self-check (Xu; Stechly), panels
of disjoint models beat single judges (PoLL), judges see verification internals
→ gameable (SHADE-Arena/SpecBench), LLM judges on claims cap ~0.65 AUROC vs
state checks 0.83–0.95 (false-success study, unreplicated). Canonical failure
name: "premature termination / missing completion verification" (MAST).
Design consequence (RUN-038): Held-out Checks in task_plan, evidence_type
per criterion, audit-fast sampling, blind_review trajectory scan.
