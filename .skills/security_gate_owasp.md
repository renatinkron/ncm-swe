# Security Gate — OWASP Top 10:2025 Skill

**Goal**: Gate security-relevant changes against the official OWASP Top 10:2025
categories before a run can close. Source: https://owasp.org/Top10/2025/
**Model**: `claude-sonnet-4-6` — escalate to `claude-opus-4-8` per orchestrator routing
when `risk_level` is high or the spec is ambiguous and high-stakes.

> **Context isolation**: receives only `changed_files` (from `execution_report`) and
> `task_plan` §Contract. Does not need spec or task rationale — this is a code-level
> gate, not a design review.

## Gate — do not proceed without classified state

Requires `.agentic/state/.classified` to exist and the active `run_state` loaded in context; if either check fails, output `BLOCKED: dispatch classify_demand` or `BLOCKED: run not initialized` and stop — no tools before the gate passes.

---

## When this runs

Orchestrator dispatches this skill (in addition to `tester`, before `blind_review`) when
**any** of:
- `change_domain == security`
- Changed files touch: authentication, authorization/access-control, session/token
  handling, cryptography, secrets/credentials, input parsing or deserialization,
  file upload/download, SQL/NoSQL/shell/template execution, outbound HTTP calls
  (SSRF surface), dependency manifests (`package.json`, `go.mod`, `requirements.txt`,
  lockfiles), CI/CD pipeline config, or logging/error-handling middleware
- `risk_level` on any task is `high`

If none apply, skip this skill entirely — do not run it on every demand.

## Inputs
`changed_files` — from `execution_report`
`task_plan` §Contract — for declared security requirements/gate commands

## Output
`.agentic/reviews/{run_id}-security-gate.md`

```
# Security Gate — {run_id} — OWASP Top 10:2025
## Findings
[{file, line, category (A01-A10), severity, issue, fix}]
## Category checklist
A01 Broken Access Control        — pass | fail | not_applicable
A02 Security Misconfiguration    — pass | fail | not_applicable
A03 Software Supply Chain Fail.  — pass | fail | not_applicable
A04 Cryptographic Failures       — pass | fail | not_applicable
A05 Injection                    — pass | fail | not_applicable
A06 Insecure Design               — pass | fail | not_applicable
A07 Authentication Failures      — pass | fail | not_applicable
A08 Software/Data Integrity Fail.— pass | fail | not_applicable
A09 Logging & Alerting Failures  — pass | fail | not_applicable
A10 Mishandling of Exceptions    — pass | fail | not_applicable
## Verdict
one-paragraph summary + gate_status (passed | failed)
```

## Category checks (OWASP Top 10:2025)

| # | Category | What to check in the diff |
|---|---|---|
| A01 | Broken Access Control | Every new/changed endpoint or function enforces the intended authorization check; no missing ownership/tenant checks; no SSRF via unvalidated user-supplied URLs (folded into A01 for 2025) |
| A02 | Security Misconfiguration | No insecure defaults added (debug mode, permissive CORS, verbose stack traces to clients, default credentials, open S3/bucket-style ACLs, unnecessary features enabled) |
| A03 | Software Supply Chain Failures | New/updated dependencies are pinned, from a legitimate source, and not typosquats; lockfiles updated alongside manifests; no unreviewed transitive privilege (postinstall scripts, etc.) |
| A04 | Cryptographic Failures | No hardcoded keys/secrets; no weak/legacy algorithms (MD5/SHA1 for security use, ECB mode, short keys); sensitive data encrypted in transit and at rest; secure random source for tokens |
| A05 | Injection | All external input reaches SQL/NoSQL/OS/LDAP/template sinks only via parameterized queries or safe APIs — never string-concatenated |
| A06 | Insecure Design | Change reflects a deliberate threat-aware design for the feature (rate limiting, least privilege, fail-secure defaults) rather than a patch bolted onto an unsafe structure |
| A07 | Authentication Failures | No weakened session/credential handling (no reduced token entropy, no auth bypass paths, no missing re-auth for sensitive actions, no credential logging) |
| A08 | Software or Data Integrity Failures | Updates/artifacts/CI steps are verified (checksums/signatures) before use; deserialization of untrusted data uses safe formats/allow-lists, not unsafe unpickling/eval-style loaders |
| A09 | Security Logging and Alerting Failures | Security-relevant events (auth failures, access-control denials, admin actions) are logged with enough context to investigate, without logging secrets/PII |
| A10 | Mishandling of Exceptional Conditions | Errors fail closed, don't leak stack traces/internal paths to users, and don't leave partial/inconsistent state on the failure path |

## Steps

1. Read each file in `changed_files` — no spec, no task_plan
2. Walk the category table above; mark each `pass`, `fail`, or `not_applicable`
3. For every `fail`, record `{file, line, category, severity, issue, fix}`
4. Severity uses the same policy as `tester` (see `.agents/orchestrator.md → Severity policy`) —
   any A01/A04/A05/A07 finding that allows unauthorized access or data exposure is at
   least `major`, and a directly exploitable one is `blocker`
5. Save report to `.agentic/reviews/{run_id}-security-gate.md`; return path + verdict

## Rules

- Report only what the diff shows — do not speculate about unrelated code
- `not_applicable` requires a one-line reason (e.g. "no auth surface touched")
- A `blocker` here escalates exactly like a tester blocker — fix before closing the run

## Contract


```contract
inputs: changed_files, task_plan
outputs: .agentic/reviews/{run_id}-security-gate.md (written_by: self)
requires_tools: Read, Write, Grep, Glob
```

**Structured result**: on completion, append your facts to the run log — `agentloop event t=security_gate <k>=<v> ...` (e.g. verdict/nc/paths). The `.md` artifact stays the validator; the event is the cheap, persistent record. Never edit `events.jsonl` directly — the binary is the single writer.
