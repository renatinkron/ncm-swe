#!/usr/bin/env bash
# watch_v3.sh — monitora artifacts/gen_ncm_v3/metrics.jsonl; quando o ckpt
# rolling cruza uma fronteira de 500 steps nova, dispara eval_ckpt_v3.sh.
# Para quando o treino terminar (final.pt existir) ou após MAXSEC.
cd "$(dirname "$0")/.."
LOG=.agentic/state/v3_watch.log
LAST_DONE=0
MAXSEC=$((60*60*60))   # 60h teto
T0=$(date +%s)

echo "[watch] start $(date -Is)" >>"$LOG"
while :; do
  if [ -f artifacts/gen_ncm_v3/final.pt ]; then
    LAST=$(tail -1 artifacts/gen_ncm_v3/metrics.jsonl | python3 -c 'import json,sys;print(json.load(sys.stdin)["step"])')
    scripts/eval_ckpt_v3.sh "final" >>"$LOG" 2>&1 || true
    echo "[watch] final avaliado; fim $(date -Is)" >>"$LOG"
    break
  fi
  LAST=$(tail -1 artifacts/gen_ncm_v3/metrics.jsonl 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["step"])' 2>/dev/null || echo 0)
  # ckpts a cada 500 locais → metrics steps 3499? não: ckpt em step%500==499 local.
  # local = metrics - 3000 → fronteiras metrics: 4499,4999,5499,... (local 1499,1999,...)
  BOUNDARY=$(( (LAST - 3000) / 500 * 500 - 1 + 3000 ))
  if [ "$BOUNDARY" -gt "$LAST_DONE" ] && [ "$BOUNDARY" -ge 4499 ]; then
    # espera 30s p/ fsync do save e confirma que metrics avançou além do save
    sleep 30
    scripts/eval_ckpt_v3.sh "$BOUNDARY" >>"$LOG" 2>&1 \
      && LAST_DONE=$BOUNDARY \
      && echo "[watch] avaliado ckpt ~$BOUNDARY $(date -Is)" >>"$LOG"
  fi
  [ $(( $(date +%s) - T0 )) -gt $MAXSEC ] && { echo "[watch] timeout" >>"$LOG"; break; }
  sleep 120
done
