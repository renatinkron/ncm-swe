# Research Skill

**Goal**: Produce a concise brief before spec writing. Save once; downstream agents load the file — do not re-search.  
**Model**: `claude-sonnet-4-6`

> **Context isolation**: receives only `{topic, scope}` — no codebase, no prior agent outputs.

## Gate — do not proceed without classified state

Requires `.agentic/state/.classified` to exist and the active `run_state` loaded in context; if either check fails, output `BLOCKED: dispatch classify_demand` or `BLOCKED: run not initialized` and stop — no tools before the gate passes.

---

## Inputs
`topic` — what to research  
`scope` — list of files, URLs, or queries to explore

## Output
`.agentic/research/{topic-slug}.md`

```
# {topic}
## Context      — 2-4 lines on why this matters
## Constraints  — key limits discovered
## Options      — brief list with tradeoffs (one line each)
## Recommendation — chosen approach + rationale
## Open Questions — unresolved items for spec author
```

## Steps

1. Search codebase and external sources within `scope` only
2. **Check for reuse first**: is there an existing library, package, or in-repo utility
   that already solves this? Note candidates and why each is or isn't a fit — do not
   default to building new without checking
   - **Cross-project check** (when `agentloop` + graph cache exist): run
     `agentloop graph registry` for indexed projects and
     `agentloop graph semantic "<key concepts>" --global` — if another indexed
     project already implemented something similar, cite it in the brief as a
     reuse precedent (with `repo` + `score`). No output/empty registry = skip
     silently; never block research on it.
3. Filter findings to only what influences a decision
4. Write brief using output format — target ≤ 1 page
5. Save to `.agentic/research/{topic-slug}.md`
6. Return path to orchestrator

## Conditional fan-out (broad scopes only)

When `scope` contains **multiple independent areas** (e.g. distinct subsystems with
no overlap), the research may **fan-out**: dispatch one sub-researcher per area in
parallel, each producing a partial brief, then **consolidate** them into the single
output file. This is **opt-in**, not the default:

- **Use fan-out when**: the scope is broad (3+ independent areas) and a single pass
  would either exceed a reasonable context budget or leave areas shallow. Parallel
  research improves coverage on independent areas (industry data: +90% on parallel
  research tasks).
- **Do NOT use fan-out when**: the scope is small, the areas overlap (shared context
  matters), or a single concise brief already fits ≤ 1 page. Fan-out on small scopes
  is pure overhead.
- **Consolidation is mandatory**: regardless of how many sub-researchers ran, the
  output is always **one** `.agentic/research/{topic-slug}.md` file. Merge partial
  briefs into it; downstream agents load that one file and never re-search.
- On advisory harnesses that can't dispatch parallel units, run the areas
  sequentially in one pass — fan-out degrades to serial, same as `execution_mode`.

## Rules

- Omit findings that don't change the recommendation
- Cite files or sources inline — no separate refs section
- If more context is needed, note it in Open Questions; do not expand inline

## Contract


```contract
inputs: topic, scope
outputs: .agentic/research/{topic-slug}.md (written_by: self)
requires_tools: Read, Write, Grep, Glob, WebFetch, WebSearch
```

**Structured result**: on completion, append your facts to the run log — `agentloop event t=research <k>=<v> ...` (e.g. verdict/nc/paths). The `.md` artifact stays the validator; the event is the cheap, persistent record. Never edit `events.jsonl` directly — the binary is the single writer.
