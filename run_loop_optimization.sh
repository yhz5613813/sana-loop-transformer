#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
mkdir -p logs
exec 9>logs/loop_optimization.lock
flock -n 9 || { echo "Optimization queue already running"; exit 1; }
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
trap 'code=$?; printf "FAILED exit=%s line=%s\n" "$code" "$LINENO" >logs/loop_optimization_status.txt; exit "$code"' ERR
printf 'RUNNING reference\n' >logs/loop_optimization_status.txt
torchrun --standalone --nproc_per_node=8 evaluate_loop_optimization_reference.py --root "$root" >logs/loop_opt_reference.log 2>&1
for variant in cache_bridge distill_bridge rollout_distill_bridge cache_rollout_distill; do
  fast="/tmp/sana_finetune2m/loop_opt_${variant}"
  target="$root/results_loop_opt_${variant}"
  test ! -e "$fast/latest.pt"
  test ! -e "$target/latest.pt"
  printf 'RUNNING %s\n' "$variant" >logs/loop_optimization_status.txt
  torchrun --standalone --nproc_per_node=8 train_sana_loop_optimized.py \
    --root "$root" --variant "$variant" --epochs 2 --val-every 200 --output "$fast" \
    >"logs/loop_opt_${variant}.log" 2>&1
  test -f "$fast/PILOT_COMPLETE.json"
  mkdir -p "$target"
  cp -a "$fast/." "$target/"
  sha256sum "$fast/latest.pt" "$target/latest.pt" >"logs/loop_opt_${variant}_sha256.txt"
  python summarize_loop_optimization.py --partial >logs/loop_opt_summary.log 2>&1
done
python summarize_loop_optimization.py >logs/loop_opt_summary.log 2>&1
printf 'COMPLETE four experiments\n' >logs/loop_optimization_status.txt
