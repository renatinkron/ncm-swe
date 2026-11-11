#!/usr/bin/env bash
# watch_v4.sh — monitora artifacts/gen_ncm_v4/metrics.jsonl; a cada nova
# fronteira de save (500 steps locais) dispara eval_ckpt_v4.sh.
# Para quando final.pt existir (avalia o final) ou após MAXSEC.
cd "$(dirname "$0")/.."
set -u
OFFSET=6499
FIRST=6998   # primeira fronteira avaliável (local 499 + offset)
LOG=.agentic/state/v4_watch.log
LAST_DONE=0
MAXSEC=$((60*60*72))   # 72h teto
T0=$(date +%s)

echo "[watch] start $(date -Is)" >>"$LOG"
while :; do
  if [ -f artifacts/gen_ncm_v4/final.pt ]; then
    sleep 60   # espera o save completar
    scripts/eval_ckpt_v4.sh final >>"$LOG" 2>&1 || true
    echo "[watch] final avaliado; fim $(date -Is)" >>"$LOG"
    break
  fi
  LAST=$(tail -1 artifacts/gen_ncm_v4/metrics.jsonl 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["step"])' 2>/dev/null || echo 0)
  LOCAL=$((LAST - OFFSET))
  BOUNDARY=$(( (LOCAL + 1) / 500 * 500 - 1 + OFFSET ))
  if [ "$BOUNDARY" -gt "$LAST_DONE" ] && [ "$BOUNDARY" -ge "$FIRST" ]; then
    if scripts/eval_ckpt_v4.sh "$BOUNDARY" >>"$LOG" 2>&1; then
      LAST_DONE=$BOUNDARY
      echo "[watch] avaliado ckpt ~$BOUNDARY $(date -Is)" >>"$LOG"
    else
      echo "[watch] eval $BOUNDARY falhou $(date -Is)" >>"$LOG"
    fi
  fi
  [ $(( $(date +%s) - T0 )) -gt $MAXSEC ] && { echo "[watch] timeout" >>"$LOG"; break; }
  sleep 120
done
