#!/usr/bin/env bash
# REPL do gen_bpe_curr.pt — ./repl.sh [CKPT_NAME]
CKPT="${1:-gen_bpe_curr.pt}"
exec podman run --rm -it --network none \
  -v "$HOME/Documents/projects/ncm_swe:/repo" \
  -v "$HOME/Documents/projects/ncm_swe/artifacts:/work/artifacts" \
  -e CTX=256 -e THREADS=8 -e CKPT_NAME="$CKPT" \
  localhost/lcm-infer:latest python3 -u /repo/scripts/_repl.py
