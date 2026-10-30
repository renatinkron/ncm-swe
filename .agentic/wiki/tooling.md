# Companion tooling — adopt, don't rebuild

Tools that complement agentloop's audit trail. All optional — `agentloop
global status` reports which are present (read-only `LookPath` detection;
absence changes nothing). Install with `agentloop companions install
<name|all>` — verified GitHub-release download (sha256-checked) into
`~/.local/bin`; `global install --companions` covers ai-memory at install
time, and ai-jail needs the explicit `--with-ai-jail` opt-in (GPL-3.0).

## ai-memory — cross-harness session memory

- Repo: https://github.com/akitaonrails/ai-memory — **MIT**
- What it is: searchable ledger of agent sessions (messages, tool calls,
  compactions, git checkpoints) + distilled project wiki + handoffs. Local
  embeddings, FTS + entity graph, multi-vendor incl. Devin CLI support.
- Why adopt instead of build: mature (12.7k★, thousands of commits) [attributed:
  GitHub, 2026-10], solves exactly the "memory outside the agent" problem we
  once considered building.
- Usage when present: `ai-memory run` inside sessions; `ai-memory search
  <topic>` before research steps.
- **Devin wiring caveat**: upstream `ai-memory setup` writes
  `~/.devin/hooks.v1.json`, which current Devin CLI ignores at user level.
  `agentloop companions setup` runs setup **and** merges the hooks into
  `~/.config/devin/config.json` automatically.
- Boundary: ai-memory is *your project memory*; `.agentic/` is the
  *governance/audit trail*. Complementary, not overlapping.

## ai-jail — OS-level sandbox

- Repo: https://github.com/akitaonrails/ai-jail — **GPL-3.0**
- What it is: bwrap-based OS sandbox + snapshots for unattended/dangerous
  sessions. Contains damage at the filesystem level.
- **License boundary: invoke externally only, never vendor.** GPL-3.0 code
  must not be copied into this MIT-licensed project.
- Relationship to hooks: agentloop's pre-tool hooks do *sequencing*
  (classification gates); they are not containment. ai-jail is the outer
  damage boundary for YOLO-style runs.

## ghpending — pending PR/issue queue

- What it is: at-a-glance pending PRs/issues across repos.
- Use: complements `issue_lifecycle` for *visibility*; it does not replace
  create/claim/close governance.

## llm-coding-benchmark — cost methodology reference

- Repo: https://github.com/akitaonrails/llm-coding-benchmark
- What to copy: same tasks across harnesses, **real USD** accounting, and
  cache-hit reporting (prompt caching is the dominant cost lever — his data:
  $1.19 vs $6.33 for the same 11M-token run) [attributed: llm-coding-benchmark].
- Our `scripts/bench_reconciled.py` now emits `cost_usd` +
  `cache_hit_ratio` [measured: RUN-023]; next benchmark should use real prices
  and ≥2 blind judges (see experiments.md, RUN-021 lessons).

## Deliberately NOT adopted

- **Workflow-graph engines / agent orchestrators** (LangGraph-style): the
  evidence (his benchmark + ours) says orchestration plumbing eats gains.
  Our loop stays a thin state machine.
- **A custom memory/embedding layer**: solved problem — see ai-memory.
