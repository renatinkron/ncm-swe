# Classify Demand Skill

**Goal**: Determine complexity path before any work begins. Called first on every demand.  
**Model**: `claude-haiku-4-5-20251001` — classification only, no implementation.

> **Local models are not suitable here.** Tested: `qwen2.5-coder:1.5b` returned "standard" for every input regardless of complexity (typo fix → standard, OAuth impl → standard). Use Haiku via API unconditionally.

> **Context isolation**: receives only the raw demand text.  
> Does not read codebase, prior context, or run state.

## Gate — this is the first step of every demand

1. Check `.agentic/state/.classified`. If it exists for the current demand, stop — do not reclassify.
2. Classify the demand below using the rules.
3. Commit the result by running: `agentloop classify fast|standard|complex`
   (the binary stamps `.classified` and builds the step queue — for
   `standard`/`complex` it also creates the run_state file).
4. Then follow `agentloop next` until it prints DONE. The binary sequences the
   loop; `.agents/orchestrator.md` remains the semantic spec for each step.

---

## Input
`demand` — user's request verbatim

## Output

```
path:             fast | standard | complex
rationale:        one line
estimated_files:  N
```

## Rules

**fast** — ALL must be true:
- Scope ≤ 5 files, no new files required — the files must be coesive (one concern);
  if they span unrelated concerns, this is not fast
- Risk: low — reversible in < 1 min, no data loss, no security impact
- No architecture decisions, no new dependencies
- No research needed
- Effort: ≤ 15 min

**Read-only demands are always fast**: questions, explanations, reviews, surveys,
status checks — anything whose deliverable is an *answer*, not a change — classify
`fast` regardless of breadth (a survey may legitimately read many files). The
`execute_fast` step still appends its JSON line to `fast_log.jsonl`, so the audit
trail is preserved without the full run queue.
- Any intent to create, modify, or delete files/state — including "just fix it"
  phrasing inside a question — disqualifies read-only routing: classify
  `standard` or `complex` instead.

**complex** — ANY of:
- Research needed before writing spec
- Multiple domains or environments involved
- Architecture decisions required
- High risk or irreversible actions
- New external dependencies

**standard** — everything else (default)

## Issue lifecycle

For every `standard` or `complex` demand, the orchestrator must create and claim a
GitHub issue (via `.skills/issue_lifecycle.md`) before any work begins, and close it
before the run is completed. `fast` demands only create/claim if the repository has a
GitHub remote; otherwise they append to `fast_log.jsonl`.

## Escalation

If during **fast** execution the scope grows beyond the initial estimate,
immediately escalate to **standard** and notify the user before continuing.

## Contract


```contract
inputs: demand
outputs: return-only
requires_tools: Read
```
