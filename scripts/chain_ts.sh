#!/bin/bash
# RUN-014 chain: VQ-VAE done -> encode_corpus -> NCM LevelGenerator train.
# Signal for stage completion is the artifact file each stage writes.
set -u
PY="$HOME/.venvs/ncm/bin/python"
cd /home/kron/Documents/projects/ncm_swe

echo "[chain] waiting for artifacts/tokenizer_ts.pt ..."
while ! test -s artifacts/tokenizer_ts.pt; do sleep 60; done
sleep 15
echo "[chain] tokenizer ready; encoding corpus -> artifacts/codes_ts"

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" scripts/encode_corpus.py \
  --tokenizer artifacts/tokenizer_ts.pt \
  --bpe artifacts/bpe_ts.json \
  --corpus-dir data/corpus_ts_vq \
  --out-dir artifacts/codes_ts \
  --threads 8 --batch-size 64 \
  > artifacts/encode_ts.log 2>&1 || { echo "[chain] encode FAILED"; exit 1; }

echo "[chain] codes ready; launching NCM generator train"

OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 "$PY" scripts/train_generator.py \
  --model ncm --config configs/generator.yaml \
  --vqvae artifacts/tokenizer_ts.pt --codes artifacts/codes_ts \
  --out artifacts/gen_ncm_ts --steps 3000 --batch-size 16 \
  > artifacts/gen_ncm_ts_train.log 2>&1 || { echo "[chain] ncm train FAILED"; exit 1; }

echo "[chain] all done"
