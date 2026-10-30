#!/usr/bin/env node
// PreToolUse hook — blocks state-changing tool calls when classify-demand has
// not been dispatched this turn (no .agentic/state/.classified flag).
//
// Dual-harness: Claude Code registers this on matcher Edit|Write|PowerShell
// (Bash is merged into rtk-wrap.js — see its top comment for why). Devin
// registers it via .devin/hooks.v1.json on lowercase tool names:
//   edit|write|apply_patch|notebook_edit  → unconditional block when unflagged
//   exec|powershell                       → read-only allowlist decides
//   (Bash/exec)                           → agentloop commands are exempt in
//                                         classify-common (the binary is the
//                                         gate's own mechanism)
//
// Block mechanism: exit code 2 with the reason on stderr. This is the
// documented, version-stable way for a Claude Code PreToolUse hook to stop a
// tool call and surface feedback to the model, and Devin's command hooks honor
// the same exit-2 convention (JSON {"decision":"block"} also emitted on stdout
// for harnesses that prefer it).
const fs = require('fs')
const { isStateChangingBash, isStateChangingPowerShell, isFlagged, logHookEvent } = require('./lib/classify-common')

let input
try {
  input = JSON.parse(fs.readFileSync(0, 'utf8'))
} catch {
  // Malformed/empty stdin must never brick the harness. Fail open.
  process.stdout.write('{}')
  process.exit(0)
}

const REASON = 'Blocked: run "agentloop begin \\"<demand>\\"" then "agentloop classify <level>" before state-changing calls.'

function allow() {
  process.stdout.write(JSON.stringify(input))
  process.exit(0)
}

function block() {
  logHookEvent('pretool_block', `${input.tool_name}`)
  process.stderr.write(REASON + '\n')
  // JSON block form for harnesses that parse stdout (Devin honors both).
  process.stdout.write(JSON.stringify({ decision: 'block', reason: REASON }))
  process.exit(2)
}

if (isFlagged()) {
  allow()
}

const name = (input.tool_name || '').toLowerCase()

// File-writing tools never have a read-only mode — unconditional when unflagged.
if (['edit', 'write', 'apply_patch', 'notebook_edit'].includes(name)) {
  block()
}

// Shell tools: read-only allowlist decides; agentloop commands are exempt
// inside isStateChanging* (framework tooling is the gate's own mechanism).
if (name === 'exec' || name === 'bash') {
  const cmd = input.tool_input?.command || ''
  if (isStateChangingBash(cmd)) {
    block()
  }
  allow()
}

if (name === 'powershell') {
  const cmd = input.tool_input?.command || ''
  if (isStateChangingPowerShell(cmd)) {
    block()
  }
  allow()
}

allow()
