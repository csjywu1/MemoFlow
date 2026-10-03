# Dataset setup

The repository does not redistribute dataset files.

## Argoverse 2 Motion Forecasting

Download the official Argoverse 2 motion-forecasting dataset and pass its root with `--av2-root`. The loader expects the official scenario layout:

```text
Argoverse2_Motion_Forecasting/
├── train/
│   └── <scenario_id>/scenario_<scenario_id>.parquet
└── val/
    └── <scenario_id>/scenario_<scenario_id>.parquet
```

Only focal-agent trajectory coordinates, validity flags, object types, and scenario identifiers are read by the default loader. Generated caches can be placed in `data/av2_joint_cache/` with `--av2-cache-root`.

## ETH/UCY trajectory-imputation files

Prepared ETH/UCY files are expected under:

```text
data/TrajImpute/pkl_type/<SCENE>-M/<DIFFICULTY>/
├── data_train.pkl
├── data_val.pkl
└── data_test.pkl
```

`<SCENE>` is one of `ETH`, `HOTEL`, `UNIV`, `ZARA1`, or `ZARA2`; `<DIFFICULTY>` is `Easy` or `Hard`.

Keep raw and processed datasets outside Git history. The repository `.gitignore` excludes these directories and generated tensor caches.
