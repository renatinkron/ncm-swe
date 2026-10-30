# Non-Conformity Reconciliation Skill

**Goal**: Resolve all non-conformities with minimal changes; return scope for re-execution.  
**Model**: `claude-sonnet-4-6`

## Gate — do not proceed without classified state

Requires `.agentic/state/.classified` to exist and the active `run_state` loaded in context; if either check fails, output `BLOCKED: dispatch classify_demand` or `BLOCKED: run not initialized` and stop — no tools before the gate passes.

---

## Inputs
`compliance_report`, `task_plan` (its `## Contract` section), `run_state`

## Outputs
`remediation_plan`, `reexecution_scope`

## Steps

1. Group findings by root cause
2. Define minimal corrective change per group
3. Prioritize by severity (see `.agents/orchestrator.md → Severity policy`)
4. Mark impacted tasks as `rework`; preserve compliant tasks
5. No scope spillover outside environment without approval
6. Emit `remediation_plan` + impacted task list
7. Return control to orchestrator

## Done when
`non_conformities_count == 0` and previously-compliant tasks still pass

## Contract


```contract
inputs: compliance_report, task_plan, execution_report
outputs: return-only
requires_tools: Read, Grep, Glob
```

**Structured result**: on completion, append your facts to the run log — `agentloop event t=reconcile <k>=<v> ...` (e.g. verdict/nc/paths). The `.md` artifact stays the validator; the event is the cheap, persistent record. Never edit `events.jsonl` directly — the binary is the single writer.
