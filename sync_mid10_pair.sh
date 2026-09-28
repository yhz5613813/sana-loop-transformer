#!/usr/bin/env bash
set -euo pipefail
cd /cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop
for attempt in $(seq 1 720); do
  if [[ -f /tmp/sana_finetune2m/mid10_hybrid4_results/PILOT_COMPLETE.json && -f /tmp/sana_finetune2m/mid10_pure_results/PILOT_COMPLETE.json ]]; then
    break
  fi
  sleep 10
done
test -f /tmp/sana_finetune2m/mid10_hybrid4_results/PILOT_COMPLETE.json
test -f /tmp/sana_finetune2m/mid10_pure_results/PILOT_COMPLETE.json
mkdir -p results_finetune2m_mid10_hybrid4 results_finetune2m_mid10_pure
cp -a /tmp/sana_finetune2m/mid10_hybrid4_results/. results_finetune2m_mid10_hybrid4/
cp -a /tmp/sana_finetune2m/mid10_pure_results/. results_finetune2m_mid10_pure/
python summarize_mid10_pair.py > logs/mid10_pair_summary.log 2>&1
python make_mid10_pair_comparison.py >> logs/mid10_pair_summary.log 2>&1
sha256sum /tmp/sana_finetune2m/mid10_hybrid4_results/latest.pt results_finetune2m_mid10_hybrid4/latest.pt \
  /tmp/sana_finetune2m/mid10_pure_results/latest.pt results_finetune2m_mid10_pure/latest.pt \
  > logs/mid10_pair_checkpoint_sha256.txt
echo PAIR_SYNC_COMPLETE
