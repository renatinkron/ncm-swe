# Blind Review — RUN-005

## Findings

[{file="scripts/build_pyramid_codebook.py", line=80, severity="info",
  issue="variable `mse` computed but unused — cosmetic only"},
 {file="scripts/train_level_heads.py", line=99, severity="info",
  issue="cond[k] input is k*d_code — head_k sees only codes <k; no
        teacher-forcing leak; verified"},
 {file="scripts/train_level_heads.py", line=200, severity="info",
  issue="loss uses last position only — deliberate smoke simplification;
        flagged for full training"},
 {file=".agentic/research/rag/rag.py", line=48, severity="info",
  issue="ranking loop prints empty headers before chunk text — cosmetic"},
 {file="scripts/train_level_heads.py", line=43, severity="info",
  issue="all backbone params requires_grad=False; safetensors local,
        no new deps"}]

## Patterns

n/a — fewer than 3 findings share a root cause.

## Verdict

no blockers, no majors — new scripts are additive; existing pipeline untouched.
