#!/usr/bin/env bash
# eval_ckpt_v3.sh — avalia o checkpoint do braço v3.
# Uso: scripts/eval_ckpt_v3.sh [label|final]
# Snapshot do ckpt (o trainer sobrescreve ckpt.pt a cada 500 steps locais,
# save em step%500==499; metrics step = local + OFFSET=3000), depois
# sample_official.py --l0 unigram na receita oficial (T=1.0, top-k 0,
# n=8, seed 0) + sonda exposure-bias → artifacts/v3_eval/ckpt_step<N>.json
set -euo pipefail
cd "$(dirname "$0")/.."

OFFSET=3000
LABEL="${1:-auto}"
EVALDIR=artifacts/v3_eval
mkdir -p "$EVALDIR"

case "$LABEL" in
  final) CKPT=artifacts/gen_ncm_v3/final.pt; NAME=final ;;
  auto|[0-9]*) CKPT=artifacts/gen_ncm_v3/ckpt.pt ;;
  *) echo "label inválido: $LABEL (use número ou 'final')" >&2; exit 2 ;;
esac
[ -f "$CKPT" ] || { echo "sem $CKPT"; exit 1; }

# guard anti-torn-snapshot: exige mtime estável (>120s sem escrita) —
# torch.save escreve in-place; copiar durante o save corromperia o snapshot.
NOW=$(date +%s)
MT=$(stat -c %Y "$CKPT")
if [ $((NOW - MT)) -lt 120 ]; then
  echo "ckpt recém-escrito (mtime ${MT}); aguardando estabilizar" >&2
  sleep $((130 - (NOW - MT)))
fi

LAST=$(tail -1 artifacts/gen_ncm_v3/metrics.jsonl | python3 -c 'import json,sys; print(json.load(sys.stdin)["step"])')
LOCAL=$((LAST - OFFSET))
if [ "$LABEL" = "final" ]; then
  NAME=final
elif [ "$LABEL" = "auto" ]; then
  # última fronteira de save atingida: local%500==499 → ((local+1)/500)*500-1
  NAME=$(( (LOCAL + 1) / 500 * 500 - 1 + OFFSET ))
else
  NAME="$LABEL"
fi
SNAP="$EVALDIR/ckpt_step${NAME}.pt"
OUT="$EVALDIR/ckpt_step${NAME}.json"

if [ -f "$OUT" ]; then echo "já avaliado: $OUT"; exit 0; fi
cp "$CKPT" "$SNAP"
echo "snapshot $SNAP (metrics step ~$LAST)"

podman run --rm -i --network none \
  -v "$PWD:/repo" -v "$PWD/artifacts:/work/artifacts" -w /repo \
  localhost/lcm-infer \
  python3 scripts/sample_official.py \
    --gen "/repo/$SNAP" \
    --l0 unigram \
    --temp 1.0 --top-k 0 --top-p 1.0 \
    --n 8 --seed 0 \
    --out "/repo/$OUT"

echo "== $OUT =="
python3 -c "import json;d=json.load(open('$OUT'));print({k:d[k] for k in ['dup_rate','distinct_ratio','s_per_sample','passes']});print('self',d['ce_self_gen']['total'],'gt',d['ce_held_gt']['total'])"
