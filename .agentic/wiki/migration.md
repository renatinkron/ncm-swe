# Migrating an older agentic-framework install

Short answer first: **`.agentic/` state does not need conversion.** It is the
audit trail — keep it as-is. OKF (Open Knowledge Format v0.2) is ai-memory's
wiki bundle format; converting run history into memory pages is *optional* and
mechanical — zero LLM tokens.

## Upgrade path (all mechanical, no LLM calls)

```bash
agentloop update            # refreshes managed files incl. .agentic/wiki/
agentloop migrate-global .  # preview; --apply removes local .agents/.skills once global install verified
agentloop compact --all     # pre-eventlog runs → events.jsonl
```

Legacy queue states (`branch_strategy`/`contract`/`docs` as steps) keep
validating — RUN-023/024 ran on old queues under the new binary unchanged.

## Getting past knowledge into ai-memory (optional)

Cheapest first:

```bash
ai-memory backfill          # imports existing local harness session
                            # transcripts — zero LLM, one-time
ai-memory doctor            # verify capture coverage + leftover config
```

Then, if you want run digests searchable in the wiki (wiki pages are
OKF-native — `ai-memory export-okf` bundles them):

```bash
for run in $(agentloop runs --format ids); do
  agentloop report "$run" | ai-memory write-page --path "runs/$run.md" --body -
done
```

`agentloop report` renders each run's event log as markdown; `write-page`
indexes it. No LLM, ~no tokens — the event log is already the compressed
history.

`ai-memory bootstrap` exists (LLM-summarises git log/docs into seed pages)
but costs tokens — skip unless the project genuinely needs semantic recall
over pre-history.

## What NOT to do

- Don't bulk-convert `.agentic/state/runs/*.json` into wiki prose — the
  structured state is the audit source of truth; the bridge above covers
  recall needs.
- Don't run `bootstrap` by reflex — measure first whether recall over old
  work is actually needed.

## agentloopd/MCP legado

`agentloop update` remove automaticamente entradas de servidor MCP que referenciam agentloop/agentloopd nos arquivos conhecidos (`.mcp.json`, `.devin/mcp_config*.json`, `.cursor/mcp.json`, `.claude/settings.local.json`, `.windsurf/mcp_config.json`, `.vscode/mcp.json`). Servidores de terceiros são preservados; arquivos malformados são pulados com aviso. Runs pré-eventlog são migradas para `events.jsonl` na mesma passada (idempotente).
