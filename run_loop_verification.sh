#!/usr/bin/env bash
set -euo pipefail
cd /cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
for variant in control4 sequence4 sequence4_bridge; do
  fast="/tmp/sana_finetune2m/loop_verify_${variant}"
  target="results_loop_verify_${variant}"
  test ! -e "${fast}/latest.pt"
  torchrun --standalone --nproc_per_node=8 train_sana_loop_sequence.py \
    --root /cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop \
    --variant "$variant" --epochs 2 --val-every 100 --output "$fast" \
    >"logs/loop_verify_${variant}.log" 2>&1
  mkdir -p "$target"
  cp -a "${fast}/." "$target/"
  sha256sum "${fast}/latest.pt" "${target}/latest.pt" >"logs/loop_verify_${variant}_sha256.txt"
  python summarize_loop_verification.py --partial >logs/loop_verify_summary.log 2>&1
done
python summarize_loop_verification.py >logs/loop_verify_summary.log 2>&1
echo LOOP_VERIFICATION_SYNC_COMPLETE >logs/loop_verify_sync.log
