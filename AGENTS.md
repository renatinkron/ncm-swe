# AGENTS


# STOP — READ THIS FIRST

This file is a reference map, not a starting point.

1. Every demand starts with `agentloop begin "<demand>"` — the binary sequences the
   loop step by step (`agentloop next` validates each step and prints the next one
   until DONE). `.skills/classify_demand.md` defines the classification rules the
   first emitted step applies.
2. `.agents/orchestrator.md` is the semantic reference for what each step means —
   the binary enforces the order and the artifacts each step must produce.
3. Do not execute any work from this file.
Se `.agentic/state/.classified` existe e `loop.json` indica um loop ativo (`done: false`),
     retome o passo emitido. Não execute `begin` nem `classify` novamente.
     Só inicie um loop novo quando não houver um loop classificado ativo.

## Loop pointer recovery

- `agentloop begin` refuses while a loop is active. If the pointer is wrong
  (`status` shows a run you did not intend — e.g. an orphan run created by a
  delegated call), run `agentloop resume <run-id>` to restore the intended run.
  It uses the `loop.<run>.json` snapshot written at displacement, else rebuilds
  the queue from the run's events log. Never start a duplicate run to "fix" a
  displaced pointer.
- `begin --force` abandons the active loop on purpose; the old loop is
  snapshotted and can still be resumed.
- If hooks block `agentloop`/`rtk agentloop` commands, the project's installed
  hooks are stale — run `agentloop update .` to refresh managed files.


---

Loop diagram, context isolation, and file index.  
**Full procedure, rules, enums, and model routing**: `.agents/orchestrator.md`

---

## Loop

```
demand ──→ classify_demand (haiku)
              │
              ├─ fast ──→ check reuse ──→ inline execute ──→ self-verify
              │              ──→ append fast_log.jsonl ──→ done
              │
              └─ standard/complex ──→ [research incl. reuse check? complex only]
                                          │
                                          ▼
                    architecture_evaluation? ──→ task_plan (branch + contract inside)
                                                                    │
              ┌── reconcile ◄── sync_history ◄── verify ◄── [security_gate_owasp?] ◄── execute
              │   (loop while findings > 0)
              ▼
         blind_review ──→ [complex & final_candidate: defensive_pentest/chaos_day?]
                                      ──→ docs report + index.json summary ──→ CLOSE
```

**Fast path** skips: task_plan, historian, documenter, blind_review — but still
appends one line to `.agentic/state/fast_log.jsonl`. Never fully untracked.
Read-only demands (questions, reviews, surveys — deliverable is an answer, not a
change) always classify `fast` regardless of breadth.  
Escalate to standard if scope grows during execution.

**`security_gate_owasp`** dispatches only when `change_domain == security` or changed
files touch auth/crypto/input-handling/access-control/dependency manifests — not on
every run.

**`defensive_pentest`** queues on the `complex` path only (standard-path changes are
almost never final candidates). A standard-path change that *is* release-bound should
either be reclassified `complex` or have the pentest run manually before release.

**`execution_mode` (phase dispatch)**: `spec_to_tasks` groups tasks into phases
(cohorts) and marks each `sequential` (default) or `parallel`. `parallel` is valid
only when the phase's tasks have disjoint file scopes and no inter-task dependencies;
the orchestrator fans out one executor per task and fans-in their reports before the
tester. Like model routing, `parallel` is **real** in Claude Code and **advisory**
(degrades to sequential) on Devin/Cursor/Copilot/Windsurf. See
`.agents/orchestrator.md → step 6` and `.skills/spec_to_tasks.md`.

---

## Context isolation

The orchestrator passes **only** the listed inputs to each agent/skill. No carry-over.

| Agent / Skill | Receives only | Must not see |
|---|---|---|
| **orchestrator** | everything | — |
| executor | task_plan (incl. `## Contract`) | spec, research, reports |
| tester | task_plan, {run}.heldout.md, execution_report, codebase | spec, research |
| remediator | compliance_report, task_plan, execution_report | spec, research |
| historian | run_state, execution_report, compliance_report, remediation_plan | spec |
| documenter | task_plan, compliance_report, history | research |
| **blind_review** *(skill)* | changed_files, adjacent_modules | spec, task_plan |
| **research** *(skill)* | topic, scope | codebase, prior agent context |
| **security_gate_owasp** *(skill)* | changed_files, task_plan | spec, research |
| **architecture_evaluation** *(skill)* | demand, scope, research_brief, codebase/diagram | task_plan, execution_report |
| **defensive_pentest** *(agent/skill)* | changed_files, adjacent_modules, task_plan, run_state, final_candidate, chaos_day, scope, codebase | spec, research, private prompts |
| **issue_lifecycle** *(skill)* | run_state, action | spec, task_plan |

---

## Agents

| File | Role |
|---|---|
| `.agents/orchestrator.md` | **Entry point** — loop owner, rules, enums, model routing |
| `.agents/executor.md` | Implements tasks |
| `.agents/tester.md` | Validates against contract |
| `.agents/remediator.md` | Fixes failures |
| `.agents/historian.md` | Appends iteration entries |
| `.agents/documenter.md` | Updates docs after compliance |

## Skills

| File | Output |
|---|---|
| `.skills/classify_demand.md` | `fast \| standard \| complex` |
| `.skills/research.md` | `.agentic/research/{slug}.md` |
| `.skills/spec_to_tasks.md` | task_plan |
| `.skills/execute_tasks_with_contract.md` | execution_report |
| `.skills/non_conformity_reconciliation.md` | remediation_plan |
| `.skills/implementation_history_sync.md` | history_entry |
| `.skills/blind_review.md` | `.agentic/reviews/{run_id}-blind-review.md` |
| `.skills/documentation_update_sync.md` | documentation_delta |
| `.skills/security_gate_owasp.md` | `.agentic/reviews/{run_id}-security-gate.md` — OWASP Top 10:2025 gate |
| `.skills/architecture_evaluation.md` | `.agentic/research/{slug}-architecture-evaluation.md` — state/flow/computation/failure/trust assessment |
| `.agents/defensive_pentest.md` + `.skills/defensive_pentest.md` | `.agentic/reviews/{run_id}-defensive-pentest.md` — bounded defensive pentest and optional chaos day |
| `.skills/caveman_default.md` | n/a — project communication-style default, not a loop step |
| `.skills/issue_lifecycle.md` | GitHub issue create/claim/close; gate: inescapable for standard/complex |

Every file above also exists as a directly invocable Claude Code Skill at
`.claude/skills/<name>/SKILL.md` (generated — see `framework.manifest.json → claude_skills`).

## Templates

| File | Used for |
|---|---|
| `.agentic/contracts/task_plan.template.md` | Task decomposition + `## Branch` + `## Contract` (executor–tester contract merged in) |
| `.agentic/contracts/implementation_contract.template.md` | Legacy — referenced by pre-merge runs only; new runs carry the contract inside task_plan |
| `.agentic/contracts/implementation_history.template.md` | Iteration history |
| `.agentic/contracts/documentation_update.template.md` | Doc sync report |
| `.agentic/state/run_state.template.json` | New run state |
| `.agentic/schemas/run_state.schema.json` | Validation |

---

## Data directories (never modified by installer)

| Path | Contents |
|---|---|
| `.agentic/state/runs/` | Run state files + `index.json` (each entry carries a one-line `summary`) |
| `.agentic/state/archive/` | Archived completed/old runs — moved here by `agentloop archive` |
| `.agentic/state/fast_log.jsonl` | One line per **fast**-path demand — the trace for demands too small for a full run |
| `.agentic/state/caveman.json` | Active project communication level (default: `full`) — set via `agentloop caveman <level>` |
| `.agentic/specs/` | Input spec documents |
| `.agentic/research/` | Research briefs |
| `.agentic/reviews/` | Blind review + security gate reports |
| `.agentic/wiki/` | Starter knowledge base — working-with-AI practices, our experiment record, companion tooling. Read `[.agentic/wiki/README.md]` first |

As runs accumulate, don't read every file in `.agentic/state/runs/` — read
`index.json` first. `agentloop` (installed executable) owns growth management:
`agentloop runs` lists active runs from the index, `agentloop archive [--days=N]`
moves terminal/old runs to `.agentic/state/archive/`, `agentloop reindex` rebuilds
`index.json` from disk if it drifts. The loop references these; it does not
reimplement them (see `.agents/orchestrator.md → Run-scale housekeeping`).

---

## Project-specific rules

*(Populated at runtime — see `.agents/orchestrator.md → Global rules → Project-memory
rule`. When execution/testing/remediation uncovers a project-specific constraint that
isn't already documented here, it gets appended as a bullet below before the run
closes, so the same mistake can't break the project twice.)*

- **Any `claude_subagents` entry whose canonical `.skills/*.md` file instructs it to
  save its own output ("Save to...", "Write report...") must include `Write` in its
  `tools` list in `framework.manifest.json`.** Found live (RUN-agentloop-self-update,
  2026-07-09): `research`, `blind-review`, and `security-gate-owasp` all had this
  mismatch — the skill's documented contract required a self-save the dispatched
  subagent couldn't perform, silently falling back to "return text, orchestrator
  persists it by hand." All three fixed. When adding a new dispatched skill that
  writes its own artifact, add `Write` to its tools up front — don't wait for a run
  to surface the gap.

- **`agentloop update` refuses to overwrite locally-modified canonical files**
  (guarded since RUN-007): files with uncommitted changes print `skip (locally
  modified — commit/stash, or --force)`, and a binary built from an ancestor
  commit warns that its embedded snapshot may be stale. Historical context:
  before the guard, a stale binary's `update` silently reverted uncommitted
  canonical edits (found live, RUN-002/RUN-006). If a needed file is skipped,
  commit the canonical edit first, then update.

- **`agentloop update` and `migrate-global` remove legacy `agentloopd`/agentloop-MCP
  server entries** from known project MCP configs (`.mcp.json`,
  `.devin/mcp_config*.json`, `.cursor/mcp.json`, `.claude/settings.local.json`,
  `.windsurf/mcp_config.json`, `.vscode/mcp.json`) — key or command/URL matching
  `agentloop*` under `mcpServers`/`servers`. Foreign servers are preserved,
  malformed files warn+skip, pre-eventlog runs are auto-compacted in the same
  pass (RUN-030/035). Sessions still bound to a removed server need a restart.

- **Devin CLI user-level hooks live in `~/.config/devin/config.json` under
  `"hooks"` — not `~/.devin/hooks.v1.json`.** Found live (RUN-026): ai-memory
  `setup` writes `~/.devin/hooks.v1.json`, which this CLI never loads at user
  level (project-level `.devin/hooks.v1.json` does work). Symptom: hooks
  registered but nothing captured. `agentloop companions setup` applies the
  fixup automatically; manually: merge the hooks object into the user config.

- **A `.claude/agents/*.md` tools-frontmatter fix does not hot-reload into an
  already-running Claude Code session.** Found live (RUN-publish-to-github,
  2026-07-09): the `research` subagent's `Write` fix (previous rule above) was
  confirmed on disk via `grep`, but a fresh dispatch of `research` in the *same*
  session still reported no `Write` tool. The subagent tool grant is loaded once
  per session, not re-read per dispatch. **Consequence**: after fixing a subagent's
  `tools` list mid-session, don't expect the fix to take effect until a new
  session — treat that session's dispatches of the affected subagent as still
  running the old grant, and keep the orchestrator-saves-the-file fallback in
  place until then. Don't misdiagnose this as "the fix didn't work."

---

## How each harness consumes this

Canonical behavior lives once in `.agents/*.md` and `.skills/*.md` — plain Markdown any
tool can read. On top of that, `agentloop` generates thin per-harness adapters so model
routing is real where the harness supports it.

| Harness | Entry point | Model routing |
|---|---|---|
| Claude Code | `CLAUDE.md` + generated `.claude/agents/*.md` (model pinned in frontmatter) | **Real** — dispatch via Task tool, each unit runs on its assigned model |
| GitHub Copilot | `.github/copilot-instructions.md` → `AGENTS.md` | Advisory (single model) |
| Cursor | `.cursor/rules/agents.mdc` → `AGENTS.md` | Advisory (single model) |
| Windsurf | `.windsurfrules` → `AGENTS.md` | Advisory (single model) |
| Devin / OpenCode / other | `AGENTS.md` directly | Advisory (single model) |

`.claude/agents/*.md` are **generated** — never hand-edit them. Change the canonical
`.agents/`/`.skills/` file or the routing in `framework.manifest.json → claude_subagents`,
then run `agentloop update`. See `.agents/orchestrator.md → Dispatch`.

**Why `.claude/agents/` has more files than `.agents/`:** four dispatched units
(`classify-demand`, `research`, `spec-to-tasks`, `blind-review`) are pure skills — their
canonical source is only `.skills/*.md`, they have no `.agents/*.md` counterpart. Every
name in `framework.manifest.json → claude_subagents` gets one generated file in
`.claude/agents/`, whether its canonical source is `.agents/`, `.skills/`, or both. This
is expected, not drift — verify with `diff <(jq -r '.claude_subagents[].name' framework.manifest.json | sort) <(ls .claude/agents | sed 's/\.md$//' | sort)`.

Likewise, `.claude/skills/*/SKILL.md` are **generated** adapters for `.skills/*.md`
(same mechanism, see `framework.manifest.json → claude_skills`) — this is what makes
research/classify-demand/etc. invocable as native Claude Code Skills, not just
Task-dispatched subagents. Edit `.skills/*.md`, then `agentloop update`.

---

## Install footprint

`agentloop install` (or `agentloop global install [--harness=...]`) puts the binary
under the per-user bin dir and stores the canonical `.agents/`/`.skills/` source once
under the user config directory, with thin adapters per detected harness. A project
then needs no `.agents/`/`.skills/` copies — `agentloop begin` creates only
`.agentic/` run state and reads instructions from the user-level source. Project-local
`agentloop init` remains valid for team-shared setups.

`agentloop migrate-global <dir>` previews (default) or applies (`--apply`) safe
removal of project-local framework files once the global install is verified
complete — it preserves edited/unknown content and reports skips; `--force` is the
explicit opt-in to discard local edits. `agentloop global status` shows the installed
version; `agentloop global remove --dry-run` previews cleanup.

---

## Companions (optional)

`agentloop global status` reports which of these are installed; absence changes
nothing — they are never required. Install via `agentloop companions install
<name|all>` (verified GitHub-release download into `~/.local/bin`); ai-jail
additionally needs `--with-ai-jail` (GPL-3.0 opt-in). `agentloop global install
--companions` bundles ai-memory into the framework install.

- **ai-memory** (MIT, https://github.com/akitaonrails/ai-memory): cross-harness
  session memory — searchable session ledger + distilled project wiki. When
  present, run sessions via `ai-memory run` and query `ai-memory search <topic>`
  before research steps. It is *project memory*; `.agentic/` remains the
  *governance/audit trail* — complementary, not overlapping.
- **ai-jail** (GPL-3.0, https://github.com/akitaonrails/ai-jail): OS-level
  sandbox for unattended/dangerous sessions. Invoke externally only — never
  vendor GPL code into this MIT project. Our hooks sequence work; ai-jail is
  the outer damage boundary.

See `.agentic/wiki/tooling.md` for the full companion list.

---

## Commands

```
RESEARCH       topic="<topic>" scope=[<file|url|query>]
PLAN_FROM_SPEC spec_path=<path> project_context={env:dev, domain:backend}
EXECUTE_PLAN   plan=<path>                       # plan carries its ## Contract section
VERIFY         plan=<path> execution_report=<path>
SECURITY_GATE  changed_files=[...] plan=<path>   # only when domain=security or auth/crypto/input surface touched
RECONCILE      compliance_report=<path> run_state=<path>
BLIND_REVIEW   run_id=<id> changed_files=[...] adjacent_modules=[...]
CLOSE_RUN      run_state=<path>
```

## Run event log — token-cheap history

Every standard/complex run appends one JSON line per transition to
`.agentic/state/runs/RUN-NNN.events.jsonl` — this is the persistent,
machine-readable history (the binary is the single writer; agents append only
via the command below). Use it instead of reading every artifact file:

```
agentloop session-start                # SessionStart hook entry: notice + version check + guarded auto-update
agentloop context [RUN-NNN]            # ≤500-token digest: demand, steps done, nc, last event
agentloop report  [RUN-NNN]            # render the log as readable markdown (stdout only)
agentloop event   [RUN-NNN] [t=note] k=v ...   # append a structured event
agentloop compact [RUN-NNN | --all]    # migrate pre-eventlog runs to events.jsonl (keeps .md)
```

Rule of thumb: reach for `context` before opening `.md` artifacts — the digest
usually answers "where is this run" in ~1% of the tokens. Since RUN-006, `next`
prints this digest as a footer on each newly emitted step (failure re-prints
already carry the failing expects), and each skill/agent
contract ends with a **Structured result** line: append your verdict via
`agentloop event t=<step> k=v` instead of expanding the `.md` artifact with
machine-destined prose (the `.md` stays the validator; events are the history).

---

## Code graph — use it before grep- exploring

If the project was indexed (`agentloop index`), prefer graph queries over
repeated grep/read sweeps — they are O(1) and return verified impact sets:

```
agentloop index [dir] [--full]                  # build/refresh (SCIP + inferred fallback)
agentloop graph search  <name> [--top N]        # resolve a name → canonical symbol key (ranked)
agentloop graph grep    <regex> [-i] [--top N]  # content search over indexed files only
agentloop graph callers|callees <sym> [depth]   # who depends on / depends-from a symbol
agentloop graph impact  <sym> [depth]           # transitive dependents — blast radius
agentloop graph diff-impact <git-ref>           # impacted symbols for changed files
agentloop graph path <a> <b>                    # shortest call path
agentloop graph imports <file> | stats | drift | registry
```

- **When to index**: once per project before heavy exploration, and re-run
  after pulling or large edits (unchanged files skip by content hash).
- **Coverage model** (RUN-013): ecosystem markers are detected recursively —
  `front/package.json` under a Go root gets its own `scip-typescript` pass —
  and a regex/ctags **fallback tier** ingests languages without a SCIP
  indexer (`.vue`/`.svelte` script blocks, Dart, Kotlin, Swift, Python,
  Java, C, shell…). Two trust levels, never mixed: `exact_compiler` edges
  come from real indexers; fallback edges are `inferred`. The index report
  carries `failures:[{lang,root,cmd,error}]` and `missing_indexers` — check
  them before trusting coverage.
- **Trust metadata**: every result carries `coverage` (compiler-exact edge
  fraction — inferred edges don't count toward it), `stale` (HEAD ≠ indexed
  commit), `truncated` (caps hit). Treat results as a verified lower bound —
  fall back to grep when `coverage` is low or the graph is absent (DB lives
  at `~/.cache/agentloop/graph/<hash>.db`; it is a derived cache, safe to
  delete).
- **Content search** (RUN-015): `graph grep <regex> [-i] [--top N]` searches
  file CONTENTS restricted to indexed `file:` nodes — gitignore-aware at index
  time, so no node_modules/build noise vs raw grep. Hits carry `score`
  (IDF-weighted term diversity), `count`, `terms`, and ≤8 snippet lines.
  Benchmarked: D3 recall .21→.61 at ≤100 candidates (union-of-keywords .86);
  D2 recall 1.00. `graph search` now ranks exact>prefix>substring, symbol>file.
  Files whose lines exceed 1MB (minified bundles) are silently skipped. The
  residual gap is semantic: files containing zero query terms (auth wiring
  named differently) need `graph semantic` or neighbor traversal — content
  search can't see them. Use `grep` for text, `search` for names, `impact`/
  `callers` for structure — complementary, not substitutes.
- **`adjacent_modules`**: when a run needs it (blind_review, defensive_pentest),
  derive it from `graph impact <changed-symbol>` output instead of guessing by
  directory proximity.
- **MCP**: `graph_callers|graph_callees|graph_impact` tools are exposed by
  `agentloop mcp` for harnesses that register it as an MCP server; the CLI is
  the portable equivalent.

### Semantic layer (embeddings — opt-in, needs a provider)

`agentloop embed` computes embeddings into the same DB (incremental via content
hash). **Default experts are `deps`+`docs`** — cheap, and they carry most of the
cross-project signal (field-validated RUN-003). The `code` expert is opt-in:
`--experts code,deps,docs` — on CPU-only providers cap it with `--max N`.
Requires an embedding endpoint: default is local Ollama (`ollama serve` +
`ollama pull nomic-embed-text`), or any OpenAI-compatible endpoint via
`AGENTLOOP_EMBED_URL`/`AGENTLOOP_EMBED_MODEL`/`AGENTLOOP_EMBED_KEY`/
`AGENTLOOP_EMBED_PROVIDER=openai`.

`agentloop graph semantic "<query>" [--expert code|docs|deps] [--k N] [--global]`
is MoE-routed: a deterministic gate sends the query to expert spaces (code
symbols / docs / external deps) and merges weighted results. Every hit carries
`score`, `expert`, `gate_weight` — **scores are fuzzy similarity, not verified
facts**; follow a promising hit with `graph impact`/`callers` for exact
relationships. `--global` fans out over all projects in `graph registry` —
the cross-project knowledge check (find who already solved this). MCP:
`graph_semantic` tool, same semantics.
