#!/usr/bin/env python
"""Servidor OpenAI-compatible minimo pro Dream-7B via diffuse-cpp.

  python scripts/dream_server.py [--port 8095]

Endpoints: GET /v1/models, POST /v1/chat/completions (nao-streaming).
Cada request: chat_template -> podman diffuse-cli -> decode -> resposta.
Latencia tipica nesta maquina: ~45-90s por resposta.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dream_preview as dp

PORT = 8095
GEN_N = 256
STEPS = 16
THREADS = 8
TEMP = 0.3
REMASKING = "entropy_exit"


def generate(messages, n=GEN_N, temp=TEMP, seed=-1):
    ids = dp.tokenizer().apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    plen = len(ids)
    import random
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    cmd = [
        "podman", "run", "--rm", "-i", "--network", "none",
        "-v", f"{dp.ENGINE}:/src", "-v", f"{dp.MODELS_DIR}:/models",
        "-w", "/src", dp.IMAGE, "./build/diffuse-cli",
        "-m", "/models/dream-7b-q4km.gguf", "--tokens",
        ",".join(map(str, ids)),
        "-n", str(n), "-s", str(STEPS), "-t", str(THREADS),
        "--remasking", REMASKING, "--temp", str(temp),
        "--seed", str(seed), "--cache-keep-active", "1",
    ]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=900).stdout.strip()
    wall = time.time() - t0
    # stdout do diffuse-cli = regiao gerada APENAS (sem prompt)
    gen_ids = [int(x) for x in out.split(",")
               if x.strip().lstrip("-").isdigit()]
    txt = dp.tokenizer().decode(gen_ids, skip_special_tokens=True)
    return txt, wall, len(gen_ids)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{
                "id": "dream-7b", "object": "model",
                "created": int(time.time()), "owned_by": "diffuse-cpp"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", 0))))
            messages = req.get("messages", [])
            n = min(int(req.get("max_tokens", GEN_N) or GEN_N), 512)
            temp = req.get("temperature")
            temp = TEMP if temp is None else float(temp)
            print(f"[req] {len(messages)} msgs · n={n} T={temp}",
                  flush=True)
            t0 = time.time()
            txt, wall, ntok = generate(messages, n=n, temp=temp)
            print(f"[done] {wall:.0f}s · {ntok} tok", flush=True)
            self._json(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", "dream-7b"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": txt},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0, "completion_tokens": ntok,
                    "total_tokens": ntok,
                },
            })
        except Exception as e:
            self._json(500, {"error": str(e)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    print(f"dream-server em http://localhost:{args.port}/v1 "
          f"(s={STEPS} n={GEN_N} T={TEMP} {REMASKING})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
