#!/usr/bin/env node
// PreToolUse hook — wraps verbose Bash commands with `rtk` when available.
//
// MERGE NOTE (classify-demand enforcement, T004): this script also runs the
// classify-demand gate check for the Bash matcher. Reason: it could not be
// verified in this environment whether Claude Code executes *multiple* hook
// entries registered under the same PreToolUse matcher (both running, block
// from either winning) versus honoring only one. No bundled docs/source were
// available locally to confirm (the installed CLI is a compiled binary with
// no accessible hook-dispatch source), and no network access was available
// to check current docs. Per the implementation contract's explicit
// fallback ("if genuinely undeterminable, default to the safe
// merge-into-rtk-wrap.js path"), the classify-enforce.js Bash-matcher check
// is merged into this script rather than registered as a second, possibly
// silently-dropped hook entry on the same matcher. See classify-enforce.js
// for the Edit/Write-matcher version of the same gate (registered as its
// own hook entry there, since Edit/Write only has one hook each, so no
// merge ambiguity applies there).
const { execSync } = require('child_process')
const fs = require('fs')
const { isStateChangingBash, isFlagged } = require('./lib/classify-common')

let input
try {
  input = JSON.parse(fs.readFileSync(0, 'utf8'))
} catch {
  // Malformed/empty stdin must never brick the harness. Fail open.
  process.stdout.write('{}')
  process.exit(0)
}

if (input.tool_name !== 'Bash') {
  process.stdout.write(JSON.stringify(input))
  process.exit(0)
}

const CLASSIFY_REASON = 'Blocked: dispatch classify-demand before state-changing calls (CLAUDE.md).'

if (!isFlagged() && isStateChangingBash(input.tool_input?.command)) {
  process.stderr.write(CLASSIFY_REASON + '\n')
  process.exit(2)
}

let rtkAvailable = false
try {
  execSync('rtk --version', { stdio: 'pipe' })
  rtkAvailable = true
} catch { }

if (!rtkAvailable) {
  process.stdout.write(JSON.stringify(input))
  process.exit(0)
}

const cmd = (input.tool_input?.command || '').trimStart()

const SKIP_PREFIXES = [
  'rtk ', 'cd ', 'mkdir', 'rm ', 'cp ', 'mv ', 'echo ', 'export ',
  'set ', 'source ', 'kill ', 'pkill', 'sleep', 'exit', 'true', 'false',
  'touch ', 'chmod ', 'chown ', 'ln ', 'which ', 'type ',
]

const skip = SKIP_PREFIXES.some(p => cmd.startsWith(p))
  || cmd.includes(' | ')
  || cmd.includes('rtk ')
  || cmd.length < 20

if (!skip) {
  input.tool_input.command = `rtk ${cmd}`
}

process.stdout.write(JSON.stringify(input))
