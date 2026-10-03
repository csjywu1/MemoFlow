# MemoFlow

Official PyTorch implementation of **MemoFlow: Memory-Anchored Flow Matching for Joint Trajectory Imputation and Forecasting**.

MemoFlow retrieves complete trajectories from a training-only memory, aligns them with the visible history, and refines the retrieved anchors with conditional flow matching. The same model reconstructs missing historical coordinates and generates multiple future trajectories.

## Repository layout

```text
MemoFlow/
├── configs/                 # Reproducible training and evaluation commands
├── data/                    # Dataset setup instructions (raw data is not committed)
├── scripts/                 # User-facing entry points
├── src/memoflow/
│   ├── model.py             # Memory, flow, residual adapter, and CVAE variant
│   ├── data.py              # ETH/UCY and common trajectory data utilities
│   ├── av2_data.py          # Argoverse 2 loading, splits, and missingness masks
│   ├── train.py             # Training and evaluation pipeline
│   └── prediction_baseline.py
└── tests/                   # Data, sampling, and result-summary tests
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/csjywu1/MemoFlow.git
cd MemoFlow
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

## Data

Dataset files are not distributed in this repository. See [`data/README.md`](data/README.md) for the expected AV2 and ETH/UCY directory structures.

## Training

Run the full AV2 model with:

```bash
bash configs/train_av2_memoflow.sh /path/to/Argoverse2_Motion_Forecasting
```

For a quick CPU pipeline check:

```bash
bash configs/smoke_test.sh /path/to/Argoverse2_Motion_Forecasting
```

The direct Python entry point is:

```bash
python scripts/train.py --help
```

## Evaluation

Evaluate a saved checkpoint without training:

```bash
python scripts/train.py \
  --dataset AV2 \
  --av2-root /path/to/Argoverse2_Motion_Forecasting \
  --model variational_flow \
  --use-memory \
  --eval-only \
  --eval-split test \
  --checkpoint /path/to/best.pt \
  --output-root results/evaluation
```

Classical and neural forecasting baselines are available through:

```bash
python scripts/train_baseline.py --help
```

## Tests

```bash
pytest -q
```

## Notes

- Retrieval memory is constructed from the training split only.
- Raw datasets, cached tensors, checkpoints, and experiment outputs are excluded from version control.
- AV2 missingness masks are generated deterministically from the scenario ID, mask type, ratio, and seed.
