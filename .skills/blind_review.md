# Blind Review Skill

**Goal**: Review delivered code without seeing spec or tasks. Surface issues purely from what the code shows.  
**Model**: `claude-sonnet-4-6`

> **Context isolation**: receives only `changed_files` and `adjacent_modules`.  
> Must not open spec, task_plan, or contract during this review.

## Gate — do not proceed without classified state

Requires `.agentic/state/.classified` to exist and the active `run_state` loaded in context; if either check fails, output `BLOCKED: dispatch classify_demand` or `BLOCKED: run not initialized` and stop — no tools before the gate passes.

---

## Inputs
`changed_files` — list from `execution_report`  
`adjacent_modules` — paths to neighboring modules (for boundary checks)

## Output
`.agentic/reviews/{run_id}-blind-review.md`

```
# Blind Review — {run_id}
## Findings
[{file, line, severity, issue}]
## Patterns
recurring themes — only if 3+ findings share a root cause
## Verdict
one-paragraph summary
```

## Steps

1. Read each changed file cold — do not open spec or task_plan
2. Flag any of: dead code, unclear names, security risk, missing error handling, boundary violation, style drift
3. Check adjacent modules for interface consistency
4. Classify each finding: `info` | `minor` | `major` | `blocker`
5. Group into patterns when 3+ findings share the same root cause
6. Trajectory scan: read `.agentic/state/runs/{run}.events.jsonl` — flag process
   signals the artifacts hide (CoT-monitoring finding): a step self-reporting
   "passed" with no tool-call evidence, expects failing silently, or dispatches
   claimed but absent from the log. List as findings under `trajectory`.
7. Write report; save to `.agentic/reviews/{run_id}-blind-review.md`
8. Return path to orchestrator

## Rules

- Report only what the code shows — do not infer intent
- A `blocker` here escalates the same way as a tester blocker (see `.agents/orchestrator.md → Severity policy`)

## Contract


```contract
inputs: changed_files, adjacent_modules
outputs: .agentic/reviews/{run_id}-blind-review.md (written_by: self)
requires_tools: Read, Write, Grep, Glob
```

**Structured result**: on completion, append your facts to the run log — `agentloop event t=blind_review <k>=<v> ...` (e.g. verdict/nc/paths). The `.md` artifact stays the validator; the event is the cheap, persistent record. Never edit `events.jsonl` directly — the binary is the single writer.
