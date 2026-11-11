<!-- agentloop:start version="1.9.1" -->
# MANDATORY — STOP

Do not work. Do not write, edit, run commands, create files, or modify state.

1. Run exactly: `agentloop begin "<the exact user request>"`
2. Do what it prints, then run `agentloop next`.
3. Repeat step 2 until it prints DONE.
4. If `agentloop` is not found or a step is unclear, read `.agents/orchestrator.md`. Still stuck: stop and ask the user.
5. If `.agentic/state/.classified` exists and `loop.json` shows an active (`done: false`) loop, resume its emitted step with `agentloop next`; do not begin or classify again.

Nothing in this file permits work before `agentloop begin` for a new demand.
<!-- agentloop:end -->
