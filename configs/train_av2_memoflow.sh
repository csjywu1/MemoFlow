#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/Argoverse2_Motion_Forecasting [output_dir]" >&2
  exit 2
fi

av2_root="$1"
output_root="${2:-results/av2_memoflow}"

python scripts/train.py \
  --dataset AV2 \
  --av2-root "$av2_root" \
  --av2-cache-root data/av2_joint_cache \
  --model variational_flow \
  --use-memory \
  --device cuda \
  --amp \
  --epochs 30 \
  --batch-size 8 \
  --grad-accum-steps 16 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --hidden-dim 64 \
  --layers 2 \
  --heads 4 \
  --latent-dim 16 \
  --variational-dim 64 \
  --residual-loss-weight 0.1 \
  --kl-weight 0.001 \
  --memory-k 20 \
  --memory-temperature 0.1 \
  --memory-source-ratio 1 \
  --ot-temperature 0.03 \
  --ot-fde-weight 0.5 \
  --endpoint-loss-weight 4 \
  --source-noise 0.05 \
  --eval-source-noise 0 \
  --memory-flow-strength 0.5 \
  --diversity-mode endpoint \
  --diversity-relevance 0.2 \
  --num-samples 12 \
  --sample-chunk-size 12 \
  --flow-steps 12 \
  --patience 6 \
  --av2-train-size 8192 \
  --av2-val-size 1024 \
  --av2-test-size 1024 \
  --av2-selection-seed 2026 \
  --mask-mode mixed \
  --mask-ratio 0.5 \
  --mask-seed 42 \
  --seed 42 \
  --output-root "$output_root"
