# Spec to Tasks Skill

**Goal**: Break a spec into atomic, independently verifiable tasks.  
**Model**: inherits from orchestrator (`sonnet`)

## Gate — do not proceed without classified state

Requires `.agentic/state/.classified` to exist and the active `run_state` loaded in context; if either check fails, output `BLOCKED: dispatch classify_demand` or `BLOCKED: run not initialized` and stop — no tools before the gate passes.

---

## Inputs
`spec_path`, `project_context`

## Output
`task_plan` — follows `.agentic/contracts/task_plan.template.md`. The plan is
the single planning artifact: it must include the `## Branch` decision and a
`## Contract` section (scope, quality gates, acceptance criteria) — the former
standalone `implementation_contract.{run}.md` was merged into it.

## Steps

1. Load `.agentic/research/{slug}.md` if one exists for this topic, then run
   `agentloop recall "<subsystem/area>"` once (fail-open): prior runs publish
   `runs/RUN-NNN.md` wiki pages — known failure classes in this area are the
   best source of held-out checks below.
2. Extract requirements and constraints from spec
3. Classify each requirement by domain and environment (see `.agents/orchestrator.md → Enums`)
4. Identify architecture impacts
5. Split into atomic tasks — one concern per task (atomicity still applies; it lives
   *within* a phase, not against it)
6. Set dependencies between tasks
7. **Group coesive tasks into phases (cohorts)** — a cohort is a set of tasks that
   share context or target overlapping files. Grouping avoids flat over-decomposition:
   the executor keeps coherent state within a phase, and the orchestrator can
   dispatch independent phases in parallel (see step 8). Populate the `## Phases`
   table in the task_plan: `Phase | Execution Mode | Task IDs | Rationale`
8. **Classify each phase's `execution_mode`**:
   - `sequential` (default) — tasks share files, have inter-task dependencies, or
     need a coherent step-by-step state (e.g. debugging, interdependent refactors).
     A single executor processes the phase's tasks in dependency order.
   - `parallel` — **only** when ALL of the following hold:
     (a) the phase's tasks have **disjoint file scopes** (no two tasks write the same
     file), AND (b) the tasks have **no inter-task dependencies**. The orchestrator
     dispatches one executor per task; each receives only its task_plan subset.
   - When in doubt, use `sequential`. Over-parallelizing interdependent tasks
     degrades quality (up to -70% on sequential work) and risks write conflicts.
9. Define `acceptance_criteria` and `mandatory_tests` per task
10. Assign `risk_level` and `execution_mode` per task (task-level mode must agree with
    its phase's mode)
11. Fill the `## Branch` decision and the `## Contract` section (scope in/out,
    quality gates with commands/evidence, acceptance criteria) — the tester
    validates against this section. Write intent-level acceptance checks the
    executor should NOT see to `.agentic/state/runs/{run}.heldout.md` — a
    separate tester-only file, NOT inside task_plan.md (things easy to fake if
    known: "no sibling call sites left", "deployed bundle actually serves the
    new content", "all N call sites use the helper"); keep guidance-level gates
    visible.
12. Emit `task_plan`

## Rules

- Each task maps to one or more specific files or modules
- Do not mix unrelated concerns in one task
- For new scripts include a sync task (locals/main/user_data when applicable)
- Prefer fewer, coesive phases over many single-task phases — over-decomposition
  inflates planning cost and the tester/remediation cycles. A phase should hold
  tasks that belong together; only split a phase when its tasks are genuinely
  independent (then it can become `parallel`)
- `parallel` is opt-in and gated on disjoint scopes + no dependencies. If you cannot
  prove both, mark `sequential`

## Contract


```contract
inputs: spec_path, project_context
outputs: return-only
requires_tools: Read, Write, Grep, Glob
```

**Structured result**: on completion, append your facts to the run log — `agentloop event t=task_plan <k>=<v> ...` (e.g. verdict/nc/paths). The `.md` artifact stays the validator; the event is the cheap, persistent record. Never edit `events.jsonl` directly — the binary is the single writer.
