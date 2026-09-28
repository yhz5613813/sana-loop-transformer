#!/usr/bin/env bash
set -euo pipefail
cd /cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
test -f /tmp/sana_finetune2m/mid10_bs128_smoke_v2/SMOKE_COMPLETE.json
torchrun --standalone --nproc_per_node=8 train_sana_mid10_bs128.py \
 --root /cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop \
 --resume-from /tmp/sana_finetune2m/mid10_hybrid4_results/latest.pt \
 --batch-per-device 16 --epochs 2 --val-every 50 \
 --output /tmp/sana_finetune2m/mid10_bs128_results \
 >logs/mid10_bs128_train.log 2>&1
mkdir -p results_finetune2m_mid10_bs128 results_finetune2m_mid10_hybrid4
cp -a /tmp/sana_finetune2m/mid10_hybrid4_results/. results_finetune2m_mid10_hybrid4/
cp -a /tmp/sana_finetune2m/mid10_bs128_results/. results_finetune2m_mid10_bs128/
python summarize_mid10_bs128.py >logs/mid10_bs128_summary.log 2>&1
sha256sum /tmp/sana_finetune2m/mid10_bs128_results/latest.pt results_finetune2m_mid10_bs128/latest.pt >logs/mid10_bs128_checkpoint_sha256.txt
echo BS128_SYNC_COMPLETE >logs/mid10_bs128_sync.log
