# Caveman Default Skill

**Goal**: Define this project's default communication style — ultra-compressed
("caveman"), full accuracy, ~65% fewer output tokens than normal prose.
**Model**: n/a — a style rule, not a dispatched unit.

> This is a project-wide default, not a per-run skill in the agentic loop. Any
> harness reading `AGENTS.md`/`CLAUDE.md` should honor it for every response in
> this project, on top of whatever global preference the user already has.

## Active level

Read `.agentic/state/caveman.json` once per session — that file is the source of
truth, not this document. Default (file freshly created by `agentloop init`):

```json
{ "level": "full" }
```

## Levels

| Level | Behavior |
|---|---|
| `off` | normal prose |
| `lite` | tight prose, keep articles/full sentences |
| `full` | fragments, short synonyms (project default) |
| `ultra` | strip conjunctions, one word when sufficient |

## Rules (apply at `full`; scale up/down per level above)

- Drop articles, filler words (just/really/basically/actually/simply), pleasantries
- Fragments OK. Short synonyms preferred
- No tool narration, decorative tables, emoji, long raw error logs
- Preserve the user's language (PT/EN/etc.)
- Pattern: `[thing] [action] [reason]. [next step].`

## Boundaries

- Code blocks, commit messages, PR descriptions: always normal style, never compressed
- **Auto-suspend** for: security warnings, irreversible-action confirmations, ambiguous
  multi-step sequences where compression risks a misread — switch to normal prose for
  that response, then resume

## Changing the level

- **Project-wide, persists for every session/user**: `agentloop caveman <off|lite|full|ultra> [dir]`
  — rewrites `.agentic/state/caveman.json`
- **This session only** (does not touch the file): say "stop caveman" / "caveman mode" /
  `/caveman <level>` if the harness supports it

If a user's own global preference (e.g. a personal `~/.claude/CLAUDE.md`) conflicts with
this file, the more specific/recent instruction wins per normal instruction precedence —
this file exists so the project has a sane default even without one.
