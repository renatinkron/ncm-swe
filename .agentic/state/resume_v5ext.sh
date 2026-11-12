#!/usr/bin/env bash
# Retomada do treino v5-ext pausado em RUN-034 (checkpoint @13499).
# Perdeu-se os steps 13500-13800 (~1.3h) na pausa — aceito pelo usuario.
# Uso: bash .agentic/state/resume_v5ext.sh
set -e
cd /home/kron/Documents/projects/ncm_swe
podman run -d --name ncm_train_v5ext2 --network none \
  -v /home/kron/Documents/projects/ncm_swe:/repo \
  -v /home/kron/Documents/projects/ncm_swe/artifacts:/work/artifacts \
  -w /repo localhost/lcm-infer-gcc:latest \
  python3 scripts/train_gen_official.py \
    --init-from /work/artifacts/gen_ncm_v5ext/ckpt.pt \
    --offset 13499 --steps 1001 --batch 16 --lr 3e-4 \
    --corrupt 1.0 --corrupt-schedule linear --corrupt-hi 0.75 --corrupt-lo 0.25 \
    --intra --intra-mask mix --intra-passes 32 --compile-trunk
echo "retomado: rotulando steps 13499..14500 (ckpt boundaries ~13998/14498)"
echo "watcher antigo segue valendo (detecta boundaries novos em metrics.jsonl)"
