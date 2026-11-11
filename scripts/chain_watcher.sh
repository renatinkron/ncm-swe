#!/bin/bash
# Robust chain watcher — watches /proc/PID (no pgrep self-match possible).
# Chain 1: bench_471m -> eval_471m
# Chain 2: heads train -> mark done (adaptive bench TBD, no script yet)
cd /home/kron/Documents/projects/ncm_swe
PY=/tmp/llmenv/bin/python
LOG=artifacts/queue_471m.log
BENCH_PID=$1
HEADS_PID=$2

alive() { [ -n "$1" ] && [ -d "/proc/$1" ]; }

bench_done=0; heads_done=0
# artifact-first: if process already died but output exists, chain immediately
[ -f artifacts/bench_471m.json ] && bench_done=2
[ -f artifacts/level_heads_qwen05/final.pt ] && heads_done=2

while :; do
    # --- chain 1: bench -> eval ---
    if [ "$bench_done" = 0 ] && ! alive "$BENCH_PID"; then
        if [ -f artifacts/bench_471m.json ]; then bench_done=2; else bench_done=1; fi
    fi
    if [ "$bench_done" = 2 ] && [ ! -f artifacts/eval_471m.json ]; then
        echo "[q] bench_471m done $(date) -> eval" >> "$LOG"
        $PY -u scripts/eval_samples.py --bench artifacts/bench_471m.json \
            --out artifacts/eval_471m.json >> artifacts/bench_471m.log 2>&1
        echo "[q] Q471 DONE $(date)" >> "$LOG"
        bench_done=3
    elif [ "$bench_done" = 1 ]; then
        echo "[q] bench_471m died WITHOUT json $(date) — check bench_471m.log" >> "$LOG"
        bench_done=3
    fi

    # --- chain 2: heads ---
    if [ "$heads_done" = 0 ] && ! alive "$HEADS_PID"; then
        if [ -f artifacts/level_heads_qwen05/final.pt ]; then heads_done=2; else heads_done=1; fi
    fi
    if [ "$heads_done" = 2 ]; then
        echo "[q] heads done $(date) — tail:" >> "$LOG"
        tail -3 artifacts/heads_train.log >> "$LOG"
        echo "[q] HEADS DONE $(date)" >> "$LOG"
        heads_done=3
    elif [ "$heads_done" = 1 ]; then
        echo "[q] heads died WITHOUT final.pt $(date) — check heads_train.log" >> "$LOG"
        heads_done=3
    fi

    [ "$bench_done" = 3 ] && [ "$heads_done" = 3 ] && break
    sleep 120
done
echo "[q] watcher exit $(date)" >> "$LOG"
