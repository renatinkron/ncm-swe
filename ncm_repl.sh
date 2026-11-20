#!/usr/bin/env bash
# REPL do NCM oficial (TinyStories) — a visão "renderizador" ao vivo.
# Uso: ./ncm_repl.sh           (4 threads, não briga com o treino)
#       THREADS=8 ./ncm_repl.sh (mais rápido, com CPU livre)
#
# Comandos dentro: 'g' ou Enter gera · :preview on mostra o refinamento
# por nível (12 passes) · :temp :topk :topp :cfg :freq :mu :l0 :seed
set -euo pipefail
cd "$(dirname "$0")"
exec podman run --rm -it --network none \
  -v "$PWD:/repo:ro" -v "$PWD/artifacts:/work/artifacts" \
  -e THREADS="${THREADS:-4}" -e NCM_CKPT=/work/artifacts/ncm_official \
  localhost/lcm-infer:latest nice -n 15 python3 /repo/scripts/_ncm_repl.py
