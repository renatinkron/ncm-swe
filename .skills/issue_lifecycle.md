# Issue Lifecycle Skill

**Goal**: Make repository issue tracking inescapable for every standard/complex demand. Create, claim, track, and close a GitHub issue that mirrors the run from start to finish.
**Model**: `claude-sonnet-4-6` for command parsing; `claude-haiku-4-5` if only updating fields.

## Gate

1. `.agentic/state/.classified` must exist. If not, output `BLOCKED: dispatch classify_demand` and stop.
2. `run_state` must be loaded. If not, output `BLOCKED: run not initialized` and stop.
3. Only then proceed.

---

## Inputs

- `run_state` — the current run state
- `action` — `create` | `update` | `close`

## Output

- Updated `run_state.github_issue` (number, url, status)
- GitHub issue created, claimed, or closed via `gh`

## When to call

The orchestrator must call this skill:

1. `create` — immediately after `classify_demand` returns `standard` or `complex` (and before any other work).
2. `update` — after every major phase change: `task_plan`, `execution_report`, `compliance_report`, `blind_review`.
3. `close` — after all exit criteria are met, right before the run is marked `completed`.

A fast-path demand may skip this skill if the repo has no GitHub remote; otherwise it runs inline.

## Procedure

### 0. Detect GitHub availability

```bash
gh auth status 2>&1 | head -3        # must succeed and show logged in
git remote get-url origin 2>&1 | grep -q github.com
```

- If `gh` is not authenticated or `origin` is not GitHub: record `run_state.github_issue = {"status": "unavailable", "reason": "no-github"}` and return.
- If GitHub is available, continue.

### 1. `action: create`

1. If `run_state.github_issue.number` already exists, skip to `update`.
2. Create the issue:
   ```bash
   gh issue create --title "agentic: <short demand summary>" \
                   --body "Run: \`{run_id}\`\n\nDemand: {demand}\n\nTracking issue created by the agentic loop." \
                   --label agentic
   ```
3. Capture the issue number from the URL printed by `gh` (e.g. `https://github.com/.../issues/42` → `42`).
4. Claim it:
   ```bash
   gh issue edit <number> --add-assignee @me --add-label in-progress
   ```
5. Record:
   ```json
   "github_issue": {
     "number": 42,
     "url": "https://github.com/{owner}/{repo}/issues/42",
     "status": "in-progress"
   }
   ```
6. If `gh issue create` or `gh issue edit` fails, output `BLOCKED: issue not created` and stop.

### 2. `action: update`

1. If `run_state.github_issue.number` does not exist, return.
2. If `run_state.github_issue.status` is `unavailable`, return.
3. Build a compact status comment from the current run state (≤ 2000 chars):
   - Iteration number
   - `overall_status`
   - Done / in_progress / blocked / pending tasks
   - Any blocker/critical findings
4. Post:
   ```bash
   gh issue comment <number> --body "..."
   ```

### 3. `action: close`

1. If `run_state.github_issue.number` does not exist or is `unavailable`, return.
2. If `compliance_report.non_conformities > 0` or any `blocker`/`critical` finding exists, do **not** close. Post a comment explaining why and output `BLOCKED: issue cannot be closed with open findings`.
3. Close the issue:
   ```bash
   gh issue close <number> --comment "Run {run_id} completed and validated. Closing."
   ```
4. Update labels:
   ```bash
   gh issue edit <number> --remove-label in-progress --add-label done
   ```
5. Record `run_state.github_issue.status = "closed"`.

## Inescapability rule

When GitHub is available, the run is **not allowed to proceed** without a tracked issue. If `run_state.github_issue` is missing, incomplete, or `status` is not `in-progress` or `closed` when required, the orchestrator must output `BLOCKED: issue not tracked` and stop.

Do not skip this skill with excuses such as "small change", "no time", or "will create later". The issue is created before work begins.

## Contract

```contract
inputs: run_state, action
outputs: .agentic/state/runs/{run_id}.json (written_by: self), return-only
requires_tools: Read, Write, Edit, Bash, Grep, Glob
```

**Structured result**: on completion, append your facts to the run log — `agentloop event t=issue <k>=<v> ...` (e.g. verdict/nc/paths). The `.md` artifact stays the validator; the event is the cheap, persistent record. Never edit `events.jsonl` directly — the binary is the single writer.
