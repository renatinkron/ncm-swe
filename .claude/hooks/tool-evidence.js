#!/usr/bin/env node
// PostToolUse hook — durable evidence capture (RUN-040).
//
// Every tool call appends one JSONL line to .agentic/state/hook_log.jsonl:
// {at, event:"tool_use", tool, session, ok, detail}. The blind_review
// trajectory scan reads this trail — a step claiming "passed" with no
// tool-call evidence in the log is the false-delivery pattern this exists
// to catch.
//
// NEVER blocks: parse failures, write failures, and dedupe failures all
// exit 0 — evidence capture must not break the harness.
//
// Devin imports .claude/settings.json hooks in addition to .devin/
// hooks.v1.json, so the same PostToolUse event can arrive twice. Dedupe by
// atomic mkdir on a content hash of the raw payload — identical payload =
// same event = logged once.
const fs = require('fs')
const path = require('path')
const crypto = require('crypto')
const { projectRoot } = require('./lib/classify-common')

let input
let raw = ''
try {
  raw = fs.readFileSync(0, 'utf8')
  input = JSON.parse(raw)
} catch {
  process.stdout.write('{}')
  process.exit(0)
}

function done() {
  process.stdout.write('{}')
  process.exit(0)
}

try {
  const root = projectRoot()
  const stateDir = path.join(root, '.agentic', 'state')
  const markerDir = path.join(stateDir, 'hook_markers')
  fs.mkdirSync(markerDir, { recursive: true })

  const hash = crypto.createHash('sha256').update(raw).digest('hex').slice(0, 24)
  try {
    fs.mkdirSync(path.join(markerDir, hash + '.lock'))
  } catch {
    done() // already recorded (double-fired import)
  }

  // Bound marker growth — the dedupe window only needs to outlive the
  // double-fire (seconds); prune anything older than 10 minutes.
  try {
    const cutoff = Date.now() - 10 * 60 * 1000
    for (const f of fs.readdirSync(markerDir)) {
      try {
        const st = fs.statSync(path.join(markerDir, f))
        if (st.mtimeMs < cutoff) fs.rmdirSync(path.join(markerDir, f))
      } catch { }
    }
  } catch { }

  const ti = input.tool_input || {}
  const tr = input.tool_response || {}
  const detail = (ti.file_path || ti.notebook_path || ti.path || ti.command || '').toString().slice(0, 200)
  const ok = tr.success !== false && !tr.error ? 'ok' : 'err'
  const line = JSON.stringify({
    at: new Date().toISOString(),
    event: 'tool_use',
    tool_name: input.tool_name || '',
    ok,
    detail,
  })
  fs.appendFileSync(path.join(stateDir, 'hook_log.jsonl'), line + '\n')
} catch { }

done()
