# Execute Tasks with Contract Skill

**Goal**: Execute tasks sequentially under contract; emit evidence per task.  
**Model**: inherits from executor (`sonnet`)

## Gate — do not proceed without classified state

Requires `.agentic/state/.classified` to exist and the active `run_state` loaded in context; if either check fails, output `BLOCKED: dispatch classify_demand` or `BLOCKED: run not initialized` and stop — no tools before the gate passes.

---

## Inputs
`task_plan` — the contract is its `## Contract` section

## Output
`execution_report`

## Steps

1. Verify `branch_strategy_decision` is confirmed — emit blocker and stop if missing
2. Pick next ready task by dependency order
3. Implement minimal changes within task scope
4. Apply project guardrails before tests
5. Run `mandatory_tests` from the plan's `## Contract` section
6. Collect evidence: `files_touched`, `commands`, `outputs`, `criteria_evidence`
7. Mark task `done` or `blocked`
8. Repeat from step 2 until all tasks attempted
9. Emit `execution_report` with handoff package per task

## Failure policy

| Event | Action |
|---|---|
| Missing branch gate | Emit blocker, stop |
| Guardrail violation | Emit blocker, stop task |
| Task failure | Mark blocked, include diagnostics, continue non-dependent tasks if contract allows |

## Contract


```contract
inputs: task_plan
outputs: return-only
requires_tools: Read, Write, Edit, Bash, Grep, Glob
```

**Structured result**: on completion, append your facts to the run log — `agentloop event t=execute <k>=<v> ...` (e.g. verdict/nc/paths). The `.md` artifact stays the validator; the event is the cheap, persistent record. Never edit `events.jsonl` directly — the binary is the single writer.
