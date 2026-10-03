# MemoFlow

Official PyTorch implementation of **MemoFlow: Memory-Anchored Flow Matching for Joint Trajectory Imputation and Forecasting**.

MemoFlow retrieves complete trajectories from a training-only memory, aligns the retrieved anchors with the visible history, and refines them using conditional flow matching. A variational residual adapter produces query-specific sources, while hard observation projection preserves every available historical coordinate.

## Repository structure

```text
MemoFlow/
├── configs/                 # Dataset-specific experiment settings
├── data/                    # Raw-data and processed-cache locations
├── models/
│   └── memoflow.py          # Memory retrieval, anchor adaptation, flow, and CVAE
├── src/
│   ├── data.py              # ETH/UCY loading and trajectory preprocessing
│   ├── av2_data.py          # Argoverse 2 loading, splits, and missingness masks
│   ├── train.py             # MemoFlow training and validation
│   ├── evaluate.py          # Checkpoint evaluation
│   ├── baselines.py         # Forecasting baselines
│   └── audit_av2.py         # AV2 split and memory audit
├── scripts/                 # Direct command-line entry points
├── tests/                   # Data, model-sampling, and reporting tests
└── outputs/                 # Experiment outputs
```

## Installation

```bash
git clone https://github.com/csjywu1/MemoFlow.git
cd MemoFlow
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python 3.10 or newer is required. CUDA is used when `device` is set to `cuda`.

## Data

The datasets are not redistributed. Place raw files under `data/raw/` and generated caches under `data/processed/`. The expected Argoverse 2 and ETH/UCY layouts are documented in [`data/README.md`](data/README.md).

## Training

Configuration files are provided for Argoverse 2 and all five ETH/UCY scenes.

```bash
python -m src.train --config configs/av2.json
```

Train on an ETH/UCY scene by selecting its configuration:

```bash
python -m src.train --config configs/zara1.json
```

Command-line values override the JSON configuration. For example:

```bash
python -m src.train \
  --config configs/av2.json \
  --seed 1 \
  --output-root outputs/av2/seed1
```

## Evaluation

```bash
python -m src.evaluate \
  --config configs/av2.json \
  --checkpoint outputs/av2/seed42/best.pt \
  --eval-split test \
  --output-root outputs/av2/seed42_test
```

The evaluator reports minADE, minFDE, missing-history minADE, meanADE, endpoint miss rate, and per-object-type metrics when labels are available.

## Baselines

```bash
python -m src.baselines --help
```

The baseline module contains constant-velocity, recurrent, Transformer, temporal-convolution, equivariant-MLP, destination-refinement, and CVAE predictors.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover deterministic missingness masks, disjoint AV2 splits, native-validity handling, serial/chunked flow equivalence, memory-source sampling, and run-metadata traceability.

## License

The source code is released under the [MIT License](LICENSE). Each dataset remains subject to its original terms. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
