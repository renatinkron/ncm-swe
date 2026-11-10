#!/usr/bin/env bash
# eval_ckpt_v3.sh — avalia o ckpt.pt corrente do braço v3.
# Uso: scripts/eval_ckpt_v3.sh [label]
# Snapshot do ckpt (evita overwrite pelo próximo save a cada 500 steps),
# depois sample_official.py --l0 unigram com a receita oficial (T=1.0,
# top-k 0) + sonda exposure-bias, mesma seed/n do baseline v2.
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-}"
CKPT=artifacts/gen_ncm_v3/ckpt.pt
EVALDIR=artifacts/v3_eval
mkdir -p "$EVALDIR"

# infere o step do ckpt: último metrics step, floor ao múltiplo de 500 - 1
# (trainer salva em step%500==499 local; offset 3000 → metrics step)
LAST=$(tail -1 artifacts/gen_ncm_v3/metrics.jsonl | python3 -c 'import json,sys; print(json.load(sys.stdin)["step"])')
CKPT_STEP=$(( (LAST - 3000) / 500 * 500 - 1 + 3000 ))
[ -n "$LABEL" ] && CKPT_STEP="$LABEL"
SNAP="$EVALDIR/ckpt_step${CKPT_STEP}.pt"
OUT="$EVALDIR/ckpt_step${CKPT_STEP}.json"

if [ -f "$OUT" ]; then echo "já avaliado: $OUT"; exit 0; fi
cp "$CKPT" "$SNAP"
echo "snapshot $SNAP (metrics step ~$LAST, ckpt ~$CKPT_STEP)"

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
