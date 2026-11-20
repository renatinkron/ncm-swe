#!/usr/bin/env python3
"""Live view of the RUN-036 teacher<->student conversation.
Run: ~/.venvs/ncm/bin/python scripts/watch_convdistill.py
"""
import json
import os
import time
from pathlib import Path

T = Path("artifacts/convdistill/transcript.jsonl")
M = Path("artifacts/convdistill/metrics.jsonl")
seen = 0
print("watching", T, "(ctrl+c to stop)\n")
while True:
    if T.exists():
        rows = [json.loads(l) for l in open(T) if l.strip()]
        for r in rows[seen:]:
            seen += 1
            if "error" in r:
                print(f"  ! round {r['round']}: {r['error']}\n")
                continue
            t = r["teacher"]
            print(f"=== ROUND {r['round']} | skill: {t['skill']}")
            print(f"prof strategy: {t['strategy_note']}")
            print(f"PROBE> {t['probe']}")
            print(f"30B ideal>\n{t['ideal_answer']}")
            print(f"0.5B attempt>\n{r['student_attempt'][:600]}")
            print()
    time.sleep(3)
