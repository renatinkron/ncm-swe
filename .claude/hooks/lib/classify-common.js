#!/usr/bin/env node
// Shared classify-demand gate logic, required by both classify-enforce.js
// (Edit|Write|PowerShell matcher) and rtk-wrap.js (Bash matcher, merged --
// see rtk-wrap.js top comment for why). Single source of truth so a fix
// here can't silently diverge between the two enforcement points again.
const fs = require('fs')
const path = require('path')

// Answers: is this command confidently read-only end-to-end, and therefore
// exempt from the classify-demand gate? Default-deny: anything not
// confidently read-only requires classify-demand to have run this turn.
// Distinct from rtk-wrap's SKIP_PREFIXES (rtk-prefix-worthiness, unrelated
// concern -- that list is left untouched).
//
// Patterns are anchored ^...$ against the WHOLE trimmed command, not just
// a prefix -- a multi-line or chained command must match in its entirety
// to be considered read-only. Combined with the disallowed-character
// reject-list below (which includes \n and \r), this closes the
// newline-separated-command bypass: "git status\nrm -rf x" is rejected
// because it is not, in full, one of these patterns.
// `git branch` alone only lists branches (read-only). Any argument turns it
// into create/delete/rename/set-upstream -- a real repo mutation -- so it
// must NOT be lumped into the general `git branch(\s+\S+)*` pattern used for
// the other read-only subcommands. Listed separately, with only list-mode
// flags allowed as arguments.
const GIT_BRANCH_READ_ONLY = /^git\s+branch(\s+(-a|-v|-vv|-r|--list))*$/

const READ_ONLY_BASH = [
  /^git\s+(status|log|diff|show|rev-parse)(\s+\S+)*$/,
  GIT_BRANCH_READ_ONLY,
  /^cat\s+\S+$/, /^ls(\s+\S+)*$/, /^dir(\s+\S+)*$/, /^type\s+\S+$/,
  /^pwd$/, /^cd(\s+\S+)?$/,
  /^which\s+\S+$/, /^where\s+\S+$/, /^echo\s+\S.*$/,
  /^node\s+(--version|-v)$/, /^go\s+version$/,
]

// PowerShell equivalents of the same read-only intents (Get-ChildItem,
// Test-Path, etc.) -- without these, PowerShell read-only commands fail
// safe (blocked pending classify-demand) but with unnecessary friction.
const READ_ONLY_POWERSHELL = [
  /^git\s+(status|log|diff|show|rev-parse)(\s+\S+)*$/,
  GIT_BRANCH_READ_ONLY,
  /^Get-Content\s+\S+$/, /^cat\s+\S+$/,
  /^Get-ChildItem(\s+\S+)*$/, /^ls(\s+\S+)*$/, /^dir(\s+\S+)*$/,
  /^Test-Path\s+\S+$/,
  /^Get-Location$/, /^pwd$/,
  /^Select-String\s+\S.*$/,
  /^node\s+(--version|-v)$/, /^go\s+version$/,
]

// Any of these anywhere in the command rejects read-only classification
// outright, regardless of what the command otherwise looks like --
// redirection, chaining, piping, substitution, or a newline/CR that Bash
// or PowerShell would treat as a command separator.
const DISALLOWED_CHARS = /[><|;&`$(){}\n\r]/

function isReadOnly(cmd, patterns) {
  const c = (cmd || '').trim()
  if (!c) return false
  if (DISALLOWED_CHARS.test(c)) return false
  return patterns.some(re => re.test(c))
}

function isReadOnlyBash(cmd) {
  return isReadOnly(cmd, READ_ONLY_BASH)
}

function isReadOnlyPowerShell(cmd) {
  return isReadOnly(cmd, READ_ONLY_POWERSHELL)
}

// agentloop binary commands are the gate's own mechanism — `begin`/`classify`/
// `next`/`skip` precede and drive classification, so blocking them deadlocks
// the loop. Sequencing inside the loop is enforced by the binary's `expects`
// validation, not by this hook.
// RTK (and `rtk proxy`) is a read-only output filter wrappers many users
// prefix every command with — without this the gate deadlocked the loop by
// blocking `rtk agentloop next` (agro-trade RUN-080 trace, RUN-044).
// The prefix alone is not enough: `rtk agentloop next; rm -rf x` must stay
// blocked, so the exemption applies only when the remainder carries no
// unquoted shell metachars. Command substitution ($ / `) is rejected even
// inside double quotes; single-quoted content stays literal.
const FRAMEWORK_PREFIX = /^(?:rtk(?:\.exe)?(?:\s+proxy)?\s+)?agentloop(?:\.exe)?/

function isFrameworkCmd(c) {
  const m = c.match(FRAMEWORK_PREFIX)
  if (!m) return false
  const rest = c.slice(m[0].length)
  if (rest !== '' && !/^\s/.test(rest)) return false // agentloopx, agentloop-foo
  let sq = false, dq = false
  for (const ch of rest) {
    if (sq) { if (ch === "'") sq = false; continue }
    if (dq) {
      if (ch === '"') { dq = false; continue }
      if (ch === '$' || ch === '`' || ch === '\\') return false
      continue
    }
    if (ch === "'") { sq = true; continue }
    if (ch === '"') { dq = true; continue }
    if (';&|<>$`(){}[]\\\r\n'.includes(ch)) return false
  }
  return !sq && !dq
}

function isStateChangingBash(cmd) {
  const c = (cmd || '').trim()
  if (isFrameworkCmd(c)) return false
  return !isReadOnlyBash(c)
}

function isStateChangingPowerShell(cmd) {
  const c = (cmd || '').trim()
  if (isFrameworkCmd(c)) return false
  return !isReadOnlyPowerShell(c)
}

// Hooks may run with cwd set to the project root (Claude Code) or elsewhere
// (Devin sets DEVIN_PROJECT_DIR; newer Claude Code sets CLAUDE_PROJECT_DIR).
// Resolve the flag against the project root, not the process cwd.
function projectRoot() {
  return process.env.DEVIN_PROJECT_DIR || process.env.CLAUDE_PROJECT_DIR || '.'
}

function isFlagged() {
  try {
    return fs.existsSync(path.join(projectRoot(), '.agentic', 'state', '.classified'))
  } catch {
    return false
  }
}

// logHookEvent appends one JSONL line to .agentic/state/hook_log.jsonl.
// Telemetry only — feeds the adaptive trust scoring and the benchmark. Never
// throws; a hook must not fail because logging failed.
function logHookEvent(event, detail) {
  try {
    const p = path.join(projectRoot(), '.agentic', 'state', 'hook_log.jsonl')
    fs.mkdirSync(path.dirname(p), { recursive: true })
    fs.appendFileSync(p, JSON.stringify({
      at: new Date().toISOString(), event, detail: detail || '',
    }) + '\n')
  } catch { }
}

module.exports = {
  isReadOnlyBash,
  isReadOnlyPowerShell,
  isStateChangingBash,
  isStateChangingPowerShell,
  isFlagged,
  projectRoot,
  logHookEvent,
}
