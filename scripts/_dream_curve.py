#!/usr/bin/env python
"""Micro-curva passes×qualidade do Dream-7B (diffuse-cpp) em CPU.

Grid: 5 prompts de código × s∈{8,16,24,32}. Mede wall, passes reais,
AST-parse e pass@1 (execução com asserts, timeout 2s).

  python scripts/_dream_curve.py --out artifacts/dream_curve.json
"""
import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dream_render import render

PROMPTS = [
    {"id": "is_prime",
     "p": "Write a Python function `def is_prime(n)` returning True if n is prime. Code only.",
     "test": "assert is_prime(2)==True\nassert is_prime(7)==True\nassert is_prime(8)==False\nassert is_prime(1)==False"},
    {"id": "binary_search",
     "p": "Write a Python function `def binary_search(arr, target)` returning index of target in sorted arr or -1. Code only.",
     "test": "assert binary_search([1,3,5,7],5)==2\nassert binary_search([1,3,5],4)==-1\nassert binary_search([],1)==-1"},
    {"id": "factorial",
     "p": "Write a Python function `def factorial(n)` returning n!. Code only.",
     "test": "assert factorial(0)==1\nassert factorial(5)==120\nassert factorial(3)==6"},
    {"id": "is_palindrome",
     "p": "Write a Python function `def is_palindrome(s)` returning True if s reads the same backwards. Code only.",
     "test": "assert is_palindrome('arara')==True\nassert is_palindrome('casa')==False\nassert is_palindrome('')==True"},
    {"id": "sort_list",
     "p": "Write a Python function `def sort_list(lst)` returning lst sorted ascending. Code only.",
     "test": "assert sort_list([3,1,2])==[1,2,3]\nassert sort_list([])==[]\nassert sort_list([5])==[5]"},
]
STEPS = [8, 16, 24, 32]


def code_of(text: str) -> str:
    t = text.split("<|")[0]
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("python"):
                p = p[6:].strip()
            if p.startswith("def ") or p.startswith("import"):
                return p
    return t.strip()


def exec_ok(code: str, test: str) -> bool:
    src = code + "\n\n" + test
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=2.0)
        return r.returncode == 0
    except Exception:
        return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/dream_curve.json")
    ap.add_argument("-t", type=int, default=8)
    ap.add_argument("-n", type=int, default=160)
    args = ap.parse_args()

    rows = []
    for pr in PROMPTS:
        for s in STEPS:
            print(f"[curve] {pr['id']} s={s} ...", flush=True)
            r = render(pr["p"], args.n, s, args.t, "entropy_exit", 1,
                       0.0, 42)
            code = code_of(r["text"])
            try:
                ast.parse(code)
                ast_ok = True
            except Exception:
                ast_ok = False
            ok = ast_ok and exec_ok(code, pr["test"])
            rows.append({"prompt": pr["id"], "s": s,
                         "wall_s": r["wall_s"], "passes": r["steps_used"],
                         "n_tok": r["n_tokens"], "tok_s": r["tok_s"],
                         "ast": ast_ok, "pass1": ok, "code": code})
            print(f"  -> {r['wall_s']}s {r['steps_used']}p "
                  f"{r['tok_s']}t/s ast={ast_ok} pass1={ok}", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    n_ok = sum(r["pass1"] for r in rows)
    print(f"[curve] pass@1 total {n_ok}/{len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
