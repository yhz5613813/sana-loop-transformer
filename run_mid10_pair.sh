#!/usr/bin/env bash
set -euo pipefail
cd /cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
SANA_OUTPUT_DIR=/tmp/sana_finetune2m/mid10_hybrid4_results \
torchrun --standalone --nproc_per_node=8 train_sana_mid10_interleaved.py \
  --root /cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop --anchor-every 4 \
  --steps 6000 --val-every 200 --generate-every 2000 \
  > logs/mid10_hybrid4_train.log 2>&1
SANA_OUTPUT_DIR=/tmp/sana_finetune2m/mid10_pure_results \
torchrun --standalone --nproc_per_node=8 train_sana_mid10_interleaved.py \
  --root /cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop --anchor-every 0 \
  --steps 6000 --val-every 200 --generate-every 2000 \
  > logs/mid10_pure_train.log 2>&1
echo PAIR_TRAIN_COMPLETE > logs/mid10_pair_status.txt
