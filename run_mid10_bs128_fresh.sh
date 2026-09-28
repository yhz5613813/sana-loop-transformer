#!/usr/bin/env bash
set -euo pipefail
cd /cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
test ! -e /tmp/sana_finetune2m/mid10_bs128_fresh_results/latest.pt
torchrun --standalone --nproc_per_node=8 train_sana_mid10_bs128_fresh.py \
 --root /cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop \
 --batch-per-device 16 --epochs 2 --val-every 50 \
 --output /tmp/sana_finetune2m/mid10_bs128_fresh_results \
 >logs/mid10_bs128_fresh_train.log 2>&1
mkdir -p results_finetune2m_mid10_bs128_fresh
cp -a /tmp/sana_finetune2m/mid10_bs128_fresh_results/. results_finetune2m_mid10_bs128_fresh/
python summarize_mid10_bs128_fresh.py >logs/mid10_bs128_fresh_summary.log 2>&1
sha256sum /tmp/sana_finetune2m/mid10_bs128_fresh_results/latest.pt results_finetune2m_mid10_bs128_fresh/latest.pt >logs/mid10_bs128_fresh_checkpoint_sha256.txt
echo BS128_FRESH_SYNC_COMPLETE >logs/mid10_bs128_fresh_sync.log
