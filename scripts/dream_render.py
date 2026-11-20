#!/usr/bin/env python
"""Render de código via Dream-7B + diffuse-cpp (podman, CPU).

  python scripts/dream_render.py "Write def fib(n) returning the n-th Fibonacci" \
      -n 192 -s 24 -t 8
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ENGINE = "/home/kron/Documents/projects/diffuse-cpp"
MODEL = "/models/dream-7b-q4km.gguf"
MODELS_DIR = "/home/kron/models/diffusion"
IMAGE = "localhost/llbuild:latest"
END_ID = 151645  # <|im_end|>


def tokenize(prompt: str) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    msgs = [{"role": "user", "content": prompt}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_dict=False)
    return out["input_ids"] if hasattr(out, "keys") else out


def render(prompt: str, n: int, steps: int, threads: int, remasking: str,
           keep_active: int, temp: float, seed: int) -> dict:
    ids = tokenize(prompt)
    ids_str = ",".join(map(str, ids))
    cmd = [
        "podman", "run", "--rm", "--network", "none",
        "-v", f"{ENGINE}:/src", "-v", f"{MODELS_DIR}:/models", "-w", "/src",
        IMAGE, "./build/diffuse-cli",
        "-m", MODEL, "--tokens", ids_str,
        "-n", str(n), "-s", str(steps), "-t", str(threads),
        "--remasking", remasking,
        "--cache-keep-active", str(keep_active),
        "--temp", str(temp), "--seed", str(seed),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0
    out = res.stdout + res.stderr

    m = re.search(r"Generated token IDs:.*?(\n[\d,\s]+)", out, re.S)
    gen_ids = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
    real_steps = len(re.findall(r"step \d+/\d+", out))
    if END_ID in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(END_ID)]
    text = ""
    if gen_ids:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
        text = tok.decode(gen_ids)
    return {"wall_s": round(wall, 1), "steps_used": real_steps,
            "n_tokens": len(gen_ids),
            "tok_s": round(len(gen_ids) / max(wall, 1e-9), 2),
            "text": text, "rc": res.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-n", type=int, default=192)
    ap.add_argument("-s", type=int, default=24)
    ap.add_argument("-t", type=int, default=8)
    ap.add_argument("--remasking", default="entropy_exit")
    ap.add_argument("--keep-active", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    r = render(args.prompt, args.n, args.s, args.t, args.remasking,
               args.keep_active, args.temp, args.seed)
    print(f"[render] {r['wall_s']}s · {r['steps_used']} passes · "
          f"{r['n_tokens']} tok · {r['tok_s']} tok/s")
    print("─" * 50)
    print(r["text"])
    if args.out:
        import json
        Path(args.out).write_text(__import__("json").dumps(r, indent=1))


if __name__ == "__main__":
    main()
