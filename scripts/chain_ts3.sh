#!/bin/bash
# RUN-015 chain v3: tapered VQ-VAE done -> encode -> DIVERSITY GATE -> NCM train.
set -u
PY="$HOME/.venvs/ncm/bin/python"
cd /home/kron/Documents/projects/ncm_swe

exec 9>artifacts/.chain_ts3.lock
flock -n 9 || { echo "[chain3] another chain holds the lock; exit"; exit 1; }

WAIT_MAX=21600; waited=0
echo "[chain3] waiting for artifacts/tokenizer_ts2.pt ..."
while ! test -s artifacts/tokenizer_ts2.pt; do
  sleep 60; waited=$((waited+60))
  if ! pgrep -f "train_vqvae.py.*ts2" >/dev/null; then
    sleep 10  # producer may have saved+exited inside our check window
    test -s artifacts/tokenizer_ts2.pt && break
    echo "[chain3] producer dead and no artifact; abort"; exit 1
  fi
  [ "$waited" -ge "$WAIT_MAX" ] && { echo "[chain3] timeout; abort"; exit 1; }
done
sleep 15
echo "[chain3] tokenizer ready; encoding -> artifacts/codes_ts2"

OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 "$PY" scripts/encode_corpus.py \
  --tokenizer artifacts/tokenizer_ts2.pt \
  --bpe artifacts/bpe_ts.json \
  --corpus-dir data/corpus_ts_vq \
  --out-dir artifacts/codes_ts2 \
  --threads 10 --batch-size 64 \
  > artifacts/encode_ts3.log 2>&1 || { echo "[chain3] encode FAILED"; exit 1; }

echo "[chain3] codes ready; diversity gate"
"$PY" - <<'GATE'
import numpy as np, json, sys
X = np.load("artifacts/codes_ts2/train.npy")
levels = json.load(open("artifacts/codes_ts2/meta.json"))["levels"]
ofs = 0; uniq = []
for L in levels:
    uniq.append(len(np.unique(X[:, ofs:ofs+L]))); ofs += L
print("[chain3] uniq/level:", uniq)
if min(uniq) < 10:
    print("[chain3] GATE FAIL"); sys.exit(1)
GATE
[ $? -ne 0 ] && exit 1

echo "[chain3] diversity OK; launching NCM"
OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 "$PY" scripts/train_generator.py \
  --model ncm --config configs/generator.yaml \
  --vqvae artifacts/tokenizer_ts2.pt --codes artifacts/codes_ts2 \
  --out artifacts/gen_ncm_ts2 --steps 3000 --batch-size 16 \
  > artifacts/gen_ncm_ts2_train.log 2>&1 || { echo "[chain3] ncm FAILED"; exit 1; }

echo "[chain3] all done"
