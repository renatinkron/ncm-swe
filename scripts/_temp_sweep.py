#!/usr/bin/env python
"""Sweep temp em s=16 nos 5 prompts — argmax vs sampling em difusão."""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dream_render import render
from _dream_curve import PROMPTS, code_of, exec_ok

rows = []
for temp in [float(x) for x in sys.argv[1].split(",")]:
    for pr in PROMPTS:
        r = render(pr["p"], 160, 16, 8, "entropy_exit", 1, temp, 42)
        code = code_of(r["text"])
        try:
            ast.parse(code)
            a = True
        except Exception:
            a = False
        ok = a and exec_ok(code, pr["test"])
        rows.append({"prompt": pr["id"], "s": 16, "temp": temp,
                     "wall_s": r["wall_s"], "passes": r["steps_used"],
                     "n_tok": r["n_tokens"], "tok_s": r["tok_s"],
                     "ast": a, "pass1": ok, "code": code})
        print(f"[t={temp}] {pr['id']}: {r['wall_s']}s ast={a} p@1={ok}",
              flush=True)
Path("/tmp/curve_temps.json").write_text(json.dumps(rows, indent=1))
print("done")
