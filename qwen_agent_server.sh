#!/usr/bin/env bash
# Sobe o servidor do agente de codigo local (Qwen2.5-Coder-7B).
# IMPORTANTE: NAO usar --cache-type-k/v q4_0 nesse build — corrompe a saida.
export LD_LIBRARY_PATH="$HOME/.local/opt/llama-master-nomp:$LD_LIBRARY_PATH"
exec "$HOME/.local/opt/llama-master-nomp/llama-server" \
  -m "$HOME/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf" \
  -t 8 -c 32768 -b 512 -ub 256 -fa on --load-mode mlock \
  --host 127.0.0.1 --port 8085 --jinja
