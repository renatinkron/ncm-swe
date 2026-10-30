#!/usr/bin/env node
// Stop hook — blocks the agent from ending its turn while an agentloop
// sequencing loop is active (.agentic/state/loop.json with done != true).
// This is the enforcement primitive text alone can't provide: a model that
// decides to stop mid-loop is told exactly which step is pending and sent back
// to `agentloop next`.
//
// Both Claude Code and Devin honor {"decision":"block","reason":...} on stdout
// for Stop hooks. Never throws — a malformed state file fails open (the hook
// must not wedge the harness).
//
// Guard against infinite blocking: if the emitted step has failed validation
// many times in a row (loop truly stuck), let the stop through so the model
// can surface the stall to the user instead of looping forever.
const fs = require('fs')
const path = require('path')
const { projectRoot, logHookEvent } = require('./lib/classify-common')

let input
try {
  input = JSON.parse(fs.readFileSync(0, 'utf8'))
} catch {
  process.stdout.write('{}')
  process.exit(0)
}

function out(obj) {
  process.stdout.write(JSON.stringify(obj))
  process.exit(0)
}

try {
  const p = path.join(projectRoot(), '.agentic', 'state', 'loop.json')
  const loop = JSON.parse(fs.readFileSync(p, 'utf8'))
  if (!loop || loop.done === true) {
    out({})
  }
  const stuck = (loop.trust && loop.trust.failures >= 10)
  if (stuck) {
    logHookEvent('stop_allow_stuck', `step=${loop.emitted} failures=${loop.trust.failures}`)
    out({})
  }
  const step = loop.emitted || 'classify'
  logHookEvent('stop_block', `step=${step}`)
  out({
    decision: 'block',
    reason: `agentloop run still active at step "${step}". Run: agentloop next — repeat until it prints DONE before stopping.`,
  })
} catch {
  out({})
}
