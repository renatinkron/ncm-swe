#!/usr/bin/env node
// UserPromptSubmit hook — clears the classify flag on a new user message unless
// a recent, classified loop is active, so follow-up turns do not deadlock.
// Never blocks the prompt. Always re-emits the input JSON unchanged and exits 0.
const fs = require('fs')
const path = require('path')
const { projectRoot } = require('./lib/classify-common')

let input
try {
  input = JSON.parse(fs.readFileSync(0, 'utf8'))
} catch {
  // Malformed/empty stdin must never brick the harness. Fail open.
  process.stdout.write('{}')
  process.exit(0)
}

try {
  // If an active classified loop is still in progress (done === false and a
  // classification has been committed), keep .classified intact across
  // follow-up turns ("continue", answers to questions, etc.) — wiping it
  // mid-run deadlocks the next tool call even though the demand is already
  // classified (RUN-042).
  let activeClassifiedLoop = false
  try {
    const loopPath = path.join(projectRoot(), '.agentic', 'state', 'loop.json')
    // Staleness bound: an abandoned mid-run loop must not keep satisfying the
    // per-turn classify gate forever. If loop.json hasn't been touched in
    // 24h, the loop is stale and the flag must be cleared (RUN-042 review).
    const mtime = fs.statSync(loopPath).mtimeMs
    if (Date.now() - mtime < 24 * 60 * 60 * 1000) {
      const l = JSON.parse(fs.readFileSync(loopPath, 'utf8'))
      activeClassifiedLoop = l && l.done === false && typeof l.classification === 'string' && l.classification !== ''
    }
  } catch { }
  if (!activeClassifiedLoop) {
    const flagPath = path.join(projectRoot(), '.agentic', 'state', '.classified')
    if (fs.existsSync(flagPath)) {
      fs.unlinkSync(flagPath)
    }
  }
} catch (e) {
  // Surface cleanup failures (e.g. Windows file lock) rather than swallowing
  // them. This hook never blocks the prompt, so emit a warning and continue.
  process.stderr.write(`classify-guard: failed to clear stale flag: ${e.message}\n`)
}

process.stdout.write(JSON.stringify(input))
process.exit(0)
