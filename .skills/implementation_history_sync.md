# Implementation History Sync Skill

**Goal**: Append one factual entry per iteration.  
**Model**: `claude-haiku-4-5-20251001` — structured write only.

## Gate — do not proceed without classified state

Requires `.agentic/state/.classified` to exist and the active `run_state` loaded in context; if either check fails, output `BLOCKED: dispatch classify_demand` or `BLOCKED: run not initialized` and stop — no tools before the gate passes.

---

## Inputs
`run_state`, `execution_report`, `compliance_report`, `remediation_plan`

## Outputs
`history_entry`, updated `.agentic/state/runs/{run_id}.json`

## Steps

1. Collect task results from `execution_report`
2. Collect findings and tester verdict from `compliance_report`
3. Collect fix scope from `remediation_plan` (if any)
4. Write entry (fields: see `.agentic/contracts/implementation_history.template.md`)
5. Update `run_state` history pointer and persist to `.agentic/state/runs/{run_id}.json`
6. Write a one-line `summary` (≤ 15 words, factual — what changed, not why) into that
   same entry, and mirror it into `.agentic/state/runs/index.json` for this run's
   entry (`summary` field) — this is what lets other agents skim `index.json` instead
   of opening every run file as the number of runs grows

## `index.json` entry schema (exact field names — do not improvise)

The field names here differ from `run_state.json`'s own top-level names
(`run_id`→`id`, `overall_status`→`status`, `change_domain`→`domain`,
`target_environment`→`environment`) — this mismatch caused a real bug once
(`RUN-agentloop-self-update`: historian wrote `run_id` instead of `id`, which broke
`agentloop runs`'s display since the CLI's `RunSummary` struct expects `id`). Use
exactly these keys, nothing else, when creating or updating an entry:

```json
{
  "id": "<run_id>",
  "status": "<overall_status>",
  "domain": "<change_domain>",
  "environment": "<target_environment>",
  "summary": "<one-line summary, ≤15 words>",
  "created_at": "<RFC3339, from this run's first history entry timestamp>",
  "updated_at": "<RFC3339, from this run's latest history entry timestamp>"
}
```

`archived` is never set by historian — only `agentloop archive` sets it.

## Rules

- No empty entries
- Include task ids and evidence references
- Include tester verdict snapshot
- The `summary` field is mandatory on every entry — never leave it blank
- When writing/updating the matching `index.json` entry, use the exact field names
  in the schema above — do not reuse `run_state.json`'s field names

## Contract


```contract
inputs: run_state, execution_report, compliance_report, remediation_plan
outputs: .agentic/state/runs/{run_id}.json (written_by: self), return-only
requires_tools: Read, Write, Edit
```

**Structured result**: on completion, append your facts to the run log — `agentloop event t=history <k>=<v> ...` (e.g. verdict/nc/paths). The `.md` artifact stays the validator; the event is the cheap, persistent record. Never edit `events.jsonl` directly — the binary is the single writer.
