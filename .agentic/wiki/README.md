# Wiki — Working with AI Agents

Starter knowledge base shipped with agentloop. It distills two sources:

- **External research** — how experienced practitioners run AI coding agents
  (notably Fabio Akita's body of work: akitaonrails.com, AI Lair)
- **Our own experiments** — measured results from this framework's benchmark
  runs (RUN-016 through RUN-023), including the ones that failed

Read order:

| Page | What it covers |
|---|---|
| [working-with-ai.md](working-with-ai.md) | The model that works: lean loops, tests as verification, memory outside the agent, sandbox not prompts |
| [experiments.md](experiments.md) | What we actually measured — governance cost/quality data, honest verdicts |
| [tooling.md](tooling.md) | Companion tools worth adopting (ai-memory, ai-jail, ghpending, llm-coding-benchmark) |
| [migration.md](migration.md) | Upgrading an older install + optional past-history → ai-memory bridge |

## Ground rules

- These pages record *evidence and attributed opinion*. Claims tagged
  **[measured]** have data behind them; **[attributed]** means a source said it;
  everything else is design rationale.
- This wiki is *advisory*. Governance/audit state lives in `.agentic/state/`;
  companion memory (if installed) lives in `ai-memory`. This directory is just
  documentation — it ships with the framework and is never mutated at runtime.
- Contributions: keep pages short, cite sources, record negative results too.
