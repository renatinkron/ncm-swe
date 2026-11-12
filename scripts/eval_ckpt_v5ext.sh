#!/usr/bin/env bash
# eval_ckpt_v5.sh — avalia o checkpoint do braço v5 (intra + mask-predict).
# Uso: scripts/eval_ckpt_v5.sh [label|final]
# Snapshot do ckpt (save em step%500==499; metrics step = local + OFFSET=12000)
# depois sample_official.py --l0 unigram na receita oficial (T=1.0, top-k 0,
# n=8, seed 0) + refinement opcional via env REFINE_ITERS/REFINE_FRAC/
# REFINE_FROM → artifacts/v5_eval/ckpt_step<N>[_refI].json
set -euo pipefail
cd "$(dirname "$0")/.."

OFFSET=12000
LABEL="${1:-auto}"
EVALDIR=artifacts/v5ext_eval
REFINE_ITERS="${REFINE_ITERS:-0}"
REFINE_FRAC="${REFINE_FRAC:-0.3}"
REFINE_FROM="${REFINE_FROM:-7}"
mkdir -p "$EVALDIR"

case "$LABEL" in
  final) CKPT=artifacts/gen_ncm_v5ext/final.pt; NAME=final ;;
  auto|[0-9]*) CKPT=artifacts/gen_ncm_v5ext/ckpt.pt ;;
  *) echo "label inválido: $LABEL (use número ou 'final')" >&2; exit 2 ;;
esac
[ -f "$CKPT" ] || { echo "sem $CKPT"; exit 1; }

# guard anti-torn-snapshot: exige mtime estável >=120s, re-stat em loop
for _ in 1 2 3 4 5; do
  NOW=$(date +%s); MT=$(stat -c %Y "$CKPT")
  [ $((NOW - MT)) -ge 120 ] && break
  echo "ckpt recém-escrito (mtime ${MT}); aguardando estabilizar" >&2
  sleep $((130 - (NOW - MT)))
done

LAST=$(tail -1 artifacts/gen_ncm_v5ext/metrics.jsonl | python3 -c 'import json,sys; print(json.load(sys.stdin)["step"])')
LOCAL=$((LAST - OFFSET))
if [ "$LABEL" = "final" ]; then
  NAME=final
elif [ "$LABEL" = "auto" ]; then
  NAME=$(( (LOCAL + 1) / 500 * 500 - 1 + OFFSET ))
else
  NAME="$LABEL"
fi
SUFFIX=""
[ "$REFINE_ITERS" != "0" ] && SUFFIX="_ref${REFINE_ITERS}f${REFINE_FRAC}l${REFINE_FROM}"
SNAP="$EVALDIR/ckpt_step${NAME}${SUFFIX}.pt"
OUT="$EVALDIR/ckpt_step${NAME}${SUFFIX}.json"

if [ -f "$OUT" ]; then echo "já avaliado: $OUT"; exit 0; fi
cp "$CKPT" "$SNAP"
echo "snapshot $SNAP (metrics step ~$LAST, refine_iters=$REFINE_ITERS)"

podman run --rm -i --network none \
  -v "$PWD:/repo" -v "$PWD/artifacts:/work/artifacts" -w /repo \
  localhost/lcm-infer \
  python3 scripts/sample_official.py \
    --gen "/repo/$SNAP" \
    --l0 unigram \
    --temp 1.0 --top-k 0 --top-p 1.0 \
    --n 8 --seed 0 \
    --refine-iters "$REFINE_ITERS" \
    --refine-frac "$REFINE_FRAC" \
    --refine-from "$REFINE_FROM" \
    --out "/repo/$OUT"

echo "== $OUT =="
python3 -c "import json;d=json.load(open('$OUT'));print({k:d[k] for k in ['dup_rate','distinct_ratio','s_per_sample','passes']});print('self',d['ce_self_gen']['total'],'gt',d['ce_held_gt']['total'])"
