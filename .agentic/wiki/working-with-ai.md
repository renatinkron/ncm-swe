# Working with AI coding agents — what experienced practice looks like

Distilled from Fabio Akita's public work (akitaonrails.com, AI Lair, repos)
as of 2026-08. Each claim links to its source; our own measurements are in
[experiments.md](experiments.md).

## The operating model **[attributed]**

Akita runs **one agent pane per project** (tmux, ~40 repos) and drives each
with terse skill names: `run pr-audit and iss-audit, then github-resolution`.
The "loop" lives in the operator's head + sharp skills — not in a workflow
engine. Every morning's routine is itself a loop
(`pr-audit → iss-audit → resolution → release → kaizen`), just an *implicit* one.

Philosophy: **"Agile Vibe Coding" = XP + LLM**. Tests, clean code, CI,
pair-programming discipline. "The idea is 10% of the work; the other 90% is
normal engineering." Software *emerges* — knowledge is distilled *from* work
(wiki/ledger), not prescribed *before* it (big specs).

Source: https://akitaonrails.com/en/ + https://ailair.akitaonrails.com/

## What he calls bullshit **[attributed]**

Post of 2026-08-18 ("Loop Engineering, Graph Engineering, Spec-Driven
Development Are Bullshit") argues:

1. **"Loop with verification" is a 50-year-old idea**: tests and code review.
   Renaming it doesn't add rigor — running the suite does.
2. **Chained agent steps fail multiplicatively**: 0.9^10 ≈ 35% end-to-end.
   In his llm-coding-benchmark, MiniMax M3 under an orchestrator scored 24
   points; the same model un-orchestrated scored 91. "When the plumbing
   dominates the result, you stopped measuring the model."
3. **Prescribed artifacts rot**: specs written before work diverge from what
   the work discovers. Wikis distilled after work stay true.
4. **"It works for me" never proved the ceremony caused the outcome.**

Source: https://akitaonrails.com/en/2026/08/18/hot-take-harness-loop-engineering-graph-engineering-are-bullshit/

## Memory lives outside the agent **[attributed]**

`ai-memory` (https://github.com/akitaonrails/ai-memory, MIT): a searchable
ledger of sessions — messages, tool calls, compactions, git checkpoints —
plus a distilled project wiki. Multi-vendor, local embeddings, team support.
This is why he can switch Claude Code → Codex → Kimi without losing context.
**Takeaway**: don't build your own cross-harness memory layer; the problem is
solved. See [tooling.md](tooling.md).

## Safety = OS sandbox, not prompt rules **[attributed]**

`ai-jail` (https://github.com/akitaonrails/ai-jail, **GPL-3.0**) sandboxes the
agent at the OS level (bwrap and friends), plus BTRFS snapshots + restic for
rollback. He runs YOLO mode because the *filesystem* is contained, not because
prompts forbid damage. Prompt-level guardrails are sequencing aids, not a
damage boundary. **Never vendor GPL code — invoke externally.**

## Cost discipline: measure dollars, cache is the lever **[attributed + measured]**

His `llm-coding-benchmark` runs the *same* tasks across harnesses and reports
real USD. Key finding: prompt caching dominates cost (Grok 4.6 run: $1.19
cached vs $6.33 uncached for the same 11M tokens). Instruction files that are
stable-prefix/variable-suffix are cache-friendly; churned context is not.

## Where we differ — deliberately

His context is a solo expert with 30 years of repertoire and his own repos.
This framework targets *teams/auditability*: when nobody watches each step,
the **trail** (issue lifecycle, contract-validated execution, compliance
report, event log) is the product. His critique shaped our lean direction —
fewer mandated artifacts, advisory graphs, docs distilled from results —
without pretending a solo expert's discipline scales to a team unattended.

## Practical rules of thumb

- Write tests first; let them, not prose, verify the agent. **[attributed]**
- Keep instruction context stable-prefixed; put volatile state at the end.
  **[measured indirectly — cache economics]**
- Prefer grep/built tools for small edits; reach for the code graph on
  large blast-radius changes. **[measured — our D2/D3 runs]**
- Don't chain 10 agent steps when 3 suffice. **[attributed]**
- Record what failed, not just what shipped — negative results are the
  most cited pages here. **[ours]**
