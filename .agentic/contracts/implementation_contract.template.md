# Implementation Contract

**Contract ID**: CONTRACT-001  
**Plan Ref**: —  
**Spec Ref**: —  
**Target Environment**: dev | hom | prod  
**Created At**: —

## Parties

| Role | Agent |
|---|---|
| Executor | executor |
| Tester | tester |
| Overseer | orchestrator |

## Branch Strategy

| Field | Value |
|---|---|
| Decision | current_branch \| new_branch |
| Confirmed At | — |
| Initial Branch | — |
| Selected Branch | — |

## Scope

**Included Tasks**: []  
**Excluded Tasks**: []

## Quality Gates

| Gate | Required | Command |
|---|---|---|
| branch_strategy_decision | required | `git rev-parse --abbrev-ref HEAD`, `git status --short` |
| worktree_state_start | required | `git status --short` |
| static_analysis | backend, frontend, infra, automation, mixed | — |
| unit_tests | backend, frontend, automation, data, mixed | — |
| integration_tests | backend, frontend, infra, data, mixed | — |
| security_checks | backend, frontend, infra, security, mixed | — |

## Architecture Rules

_none_

## Coding Standards

_none_

## Task Assertions

### T001

**Required Evidence**: branch_strategy_evidence, changed_files, acceptance_criteria_checklist, test_output_summary  
**Pass Conditions**: all_acceptance_criteria_met, mandatory_tests_passed, no_blocking_findings
