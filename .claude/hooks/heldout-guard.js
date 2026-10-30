#!/usr/bin/env node
// PreToolUse hook — keeps {run}.heldout.md tester-only.
//
// spec_to_tasks writes held-out acceptance checks during the task_plan step;
// the tester reads them during test. Any other step (notably execute — the
// executor's window) must not see or modify them: SHADE-Arena/SpecBench show
// agents saturate visible monitors while deviating from intent. The file
// stays on disk for the post-run audit record; this hook is the read/write
// boundary the filesystem alone can't express (RUN-039 blind review: the
// in-plan block was readable by any executor — procedural, not enforced).
//
// Applies to direct file tools (read/write/edit/grep/glob families) and to
// shell commands that reference a heldout path. Malformed stdin fails open;
// an absent/unknown loop state means "no legit window" → block.
const fs = require('fs')
const path = require('path')
const { projectRoot, logHookEvent, isReadOnlyBash, isReadOnlyPowerShell } = require('./lib/classify-common')

const HELDOUT_RE = /\.heldout\.md([\\/"'`)\s]|$)|\*\.heldout\.md/

let input
try {
  input = JSON.parse(fs.readFileSync(0, 'utf8'))
} catch {
  process.stdout.write('{}')
  process.exit(0)
}

function allow() {
  process.stdout.write(JSON.stringify(input))
  process.exit(0)
}

const REASON = 'Blocked: *.heldout.md is tester-only. Held-out acceptance checks are hidden from the executor on purpose (SHADE-Arena/SpecBench) — do not read or edit them.'

function block(what) {
  logHookEvent('heldout_block', `${input.tool_name}:${what}`)
  process.stderr.write(REASON + '\n')
  process.stdout.write(JSON.stringify({ decision: 'block', reason: REASON }))
  process.exit(2)
}

function emittedStep() {
  try {
    const l = JSON.parse(fs.readFileSync(path.join(projectRoot(), '.agentic', 'state', 'loop.json'), 'utf8'))
    if (l.done) return ''
    return l.emitted || ''
  } catch {
    return ''
  }
}

// Does this call touch a heldout file? Check every path-ish field the file
// tools use, plus shell commands that name one.
function heldoutTarget() {
  const ti = input.tool_input || {}
  for (const k of ['file_path', 'notebook_path', 'path', 'pattern', 'glob']) {
    const v = ti[k]
    if (typeof v === 'string' && HELDOUT_RE.test(v)) return v
  }
  const cmd = ti.command || ti.text_input || ''
  if (typeof cmd === 'string' && HELDOUT_RE.test(cmd)) return cmd.slice(0, 120)
  return ''
}

const target = heldoutTarget()
if (!target) {
  allow()
}

const step = emittedStep()
// Legit windows are asymmetric: task_plan allows read AND write (spec_to_tasks
// authors the file); test allows READ ONLY (the tester consumes it — a write
// window here would let mid-test tampering leave no git trace, since the file
// is .gitignore'd). Everything else — execute, remediate, blind_review,
// close, no-loop — is a block.
const name = (input.tool_name || '').toLowerCase()
const WRITERS = ['edit', 'write', 'apply_patch', 'notebook_edit', 'multiedit']
const SHELLS = ['exec', 'bash', 'powershell', 'write_to_process']
if (step === 'task_plan') {
  allow()
}
if (step === 'test' && !WRITERS.includes(name)) {
  if (SHELLS.includes(name)) {
    const cmd = (input.tool_input?.command || input.tool_input?.text_input || '')
    if (isReadOnlyBash(cmd) || isReadOnlyPowerShell(cmd)) {
      allow()
    }
    // a shell line touching heldout that isn't provably read-only → block
  } else {
    allow()
  }
}
block(target)
