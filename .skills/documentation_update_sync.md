# Documentation Update Sync Skill

**Goal**: Sync docs with final validated behavior after zero findings.  
**Model**: `claude-haiku-4-5-20251001` — structured write from given inputs, no reasoning required.

## Gate — do not proceed without classified state

Requires `.agentic/state/.classified` to exist and the active `run_state` loaded in context; if either check fails, output `BLOCKED: dispatch classify_demand` or `BLOCKED: run not initialized` and stop — no tools before the gate passes.

---

## Inputs
`task_plan`, `compliance_report`, `implementation_history`, `project_docs_map`

## Outputs
`documentation_delta`, `docs_update_report`

## Steps

1. Identify docs impacted by each completed task
2. Apply minimal accurate updates from validated behavior
3. Mark unchanged docs explicitly
4. Produce `docs_update_report` (see `.agentic/contracts/documentation_update.template.md`)

## Rules

- Every done task must have doc impact status (`update_required` | `no_change`)
- Docs reflect final behavior only — no speculation
- Prioritize README, architecture, and agents docs for infra or script changes

## Contract


```contract
inputs: task_plan, compliance_report, history
outputs: return-only
requires_tools: Read, Write, Edit, Grep, Glob
```

**Structured result**: on completion, append your facts to the run log — `agentloop event t=docs <k>=<v> ...` (e.g. verdict/nc/paths). The `.md` artifact stays the validator; the event is the cheap, persistent record. Never edit `events.jsonl` directly — the binary is the single writer.
