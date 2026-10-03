#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/Argoverse2_Motion_Forecasting [output_dir]" >&2
  exit 2
fi

av2_root="$1"
output_root="${2:-results/smoke_test}"

python scripts/train.py \
  --dataset AV2 \
  --av2-root "$av2_root" \
  --av2-cache-root data/av2_joint_cache \
  --model flow \
  --use-memory \
  --device cpu \
  --no-amp \
  --epochs 1 \
  --batch-size 4 \
  --hidden-dim 32 \
  --layers 1 \
  --heads 2 \
  --num-samples 2 \
  --sample-chunk-size 2 \
  --flow-steps 2 \
  --memory-k 2 \
  --av2-train-size 32 \
  --av2-val-size 16 \
  --av2-test-size 16 \
  --limit-train 16 \
  --limit-val 8 \
  --limit-test 8 \
  --mask-mode mixed \
  --mask-ratio 0.5 \
  --mask-seed 42 \
  --seed 42 \
  --output-root "$output_root"
