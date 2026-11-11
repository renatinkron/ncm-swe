#!/bin/bash
# RUN-014 chain v2: VQ-VAE(mini) done -> encode -> DIVERSITY GATE -> NCM train.
set -u
PY="$HOME/.venvs/ncm/bin/python"
cd /home/kron/Documents/projects/ncm_swe

# mutual exclusion: only one chain may own the codes_ts/gen_ncm_ts pipeline
exec 9>artifacts/.chain_ts.lock
flock -n 9 || { echo "[chain2] another chain instance holds the lock; exit"; exit 1; }

# wait with timeout (6h) AND producer liveness: bail if producer dead
WAIT_MAX=21600; waited=0
echo "[chain2] waiting for artifacts/tokenizer_ts.pt ..."
while ! test -s artifacts/tokenizer_ts.pt; do
  sleep 60; waited=$((waited+60))
  if ! pgrep -f "train_vqvae.py.*tokenizer_ts" >/dev/null; then
    echo "[chain2] producer train_vqvae not running and no artifact; abort"; exit 1
  fi
  [ "$waited" -ge "$WAIT_MAX" ] && { echo "[chain2] wait timeout ${WAIT_MAX}s; abort"; exit 1; }
done
sleep 15
echo "[chain2] tokenizer ready; encoding corpus -> artifacts/codes_ts"

OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 "$PY" scripts/encode_corpus.py \
  --tokenizer artifacts/tokenizer_ts.pt \
  --bpe artifacts/bpe_ts.json \
  --corpus-dir data/corpus_ts_vq \
  --out-dir artifacts/codes_ts \
  --threads 10 --batch-size 64 \
  > artifacts/encode_ts2.log 2>&1 || { echo "[chain2] encode FAILED"; exit 1; }

echo "[chain2] codes ready; checking diversity gate"

"$PY" - <<'GATE'
import numpy as np, sys
X = np.load("artifacts/codes_ts/train.npy")
levels = [1,8,16,32,64,128]
ofs = 0; uniq = []
for L in levels:
    uniq.append(len(np.unique(X[:, ofs:ofs+L]))); ofs += L
print("[chain2] uniq/level:", uniq)
if min(uniq) < 10:
    print("[chain2] GATE FAIL: collapsed codes, aborting NCM launch")
    sys.exit(1)
GATE
[ $? -ne 0 ] && exit 1

echo "[chain2] diversity OK; launching NCM generator train"

OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 "$PY" scripts/train_generator.py \
  --model ncm --config configs/generator.yaml \
  --vqvae artifacts/tokenizer_ts.pt --codes artifacts/codes_ts \
  --out artifacts/gen_ncm_ts --steps 3000 --batch-size 16 \
  > artifacts/gen_ncm_ts_train.log 2>&1 || { echo "[chain2] ncm train FAILED"; exit 1; }

echo "[chain2] all done"
