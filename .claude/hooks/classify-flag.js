#!/usr/bin/env node
// PreToolUse hook — stamps .agentic/state/.classified when classify-demand is
// dispatched. Never blocks. Always re-emits the input JSON unchanged, exit 0.
//
// Tool-name portability: the subagent-dispatch tool is "Task" in the standard
// Claude Code CLI but "Agent" in SDK-based sessions — so this keys off
// tool_input.subagent_type, which only appears on dispatch calls regardless of
// the tool's name. Devin's `skill` tool uses tool_input.skill and
// `run_subagent` uses profile/title, so those are checked too. Note the
// `agentloop classify` binary command also stamps the flag directly (Go side)
// — this hook covers the harness-native dispatch paths.
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
  const ti = input.tool_input || {}
  const isClassify =
    ti.subagent_type === 'classify-demand' || // Claude Code Task/Agent dispatch
    ti.skill === 'classify-demand' ||         // Devin/native skill invocation
    ti.skill === 'classify_demand' ||
    (typeof ti.title === 'string' && /classify/i.test(ti.title)) // run_subagent
  if (isClassify) {
    const dir = path.join(projectRoot(), '.agentic', 'state')
    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(path.join(dir, '.classified'), new Date().toISOString())
  }
} catch { }

process.stdout.write(JSON.stringify(input))
process.exit(0)
