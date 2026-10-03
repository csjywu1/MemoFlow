from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow.parquet as parquet
import torch
from torch.utils.data import Dataset

from src.data import (
    JointDataBundle,
    _canonicalize,
    _relative_rotation,
    build_engineered_keys,
)


# AV2 contains 50 observed and 60 future samples at 10 Hz. Selecting every
# fifth point gives the same 8-history/12-future tensor shape and approximately
# the same temporal resolution as the existing ETH/UCY protocol.
HISTORY_TIMESTEPS = tuple(range(14, 50, 5))
FUTURE_TIMESTEPS = tuple(range(54, 110, 5))
SELECTED_TIMESTEPS = HISTORY_TIMESTEPS + FUTURE_TIMESTEPS
OBJECT_TYPES = {
    "vehicle": 0,
    "pedestrian": 1,
    "motorcyclist": 2,
    "cyclist": 3,
    "bus": 4,
    "static": 5,
    "background": 6,
    "construction": 7,
    "riderless_bicycle": 8,
    "unknown": 9,
}
OBJECT_TYPE_NAMES = {value: key for key, value in OBJECT_TYPES.items()}


@dataclass
class RawAV2Split:
    positions: torch.Tensor
    native_valid: torch.Tensor
    object_type: torch.Tensor
    scenario_ids: list[str]


def _stable_u64(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _scenario_parquet(split_root: Path, scenario_id: str) -> Path:
    return split_root / scenario_id / f"scenario_{scenario_id}.parquet"


def _list_scenarios(split_root: Path) -> list[str]:
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    return [
        entry.name
        for entry in os.scandir(split_root)
        if entry.is_dir(follow_symlinks=False)
    ]


def _select_scenarios(
    split_root: Path,
    count: int,
    selection_seed: int,
) -> list[str]:
    scenario_ids = _list_scenarios(split_root)
    scenario_ids.sort(key=lambda item: (_stable_u64(selection_seed, item), item))
    if count <= 0:
        return scenario_ids
    if len(scenario_ids) < count:
        raise ValueError(
            f"requested {count} scenarios from {split_root}, only {len(scenario_ids)} exist"
        )
    return scenario_ids[:count]


def _read_focal_track(path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    table = parquet.read_table(
        path,
        columns=[
            "track_id",
            "object_type",
            "timestep",
            "position_x",
            "position_y",
            "focal_track_id",
        ],
    )
    values = table.to_pydict()
    if not values["focal_track_id"]:
        raise ValueError(f"empty scenario parquet: {path}")
    focal_track_id = values["focal_track_id"][0]
    selected_lookup = {step: index for index, step in enumerate(SELECTED_TIMESTEPS)}
    positions = torch.zeros(len(SELECTED_TIMESTEPS), 2, dtype=torch.float32)
    valid = torch.zeros(len(SELECTED_TIMESTEPS), dtype=torch.bool)
    object_type = "unknown"
    for track_id, kind, timestep, x, y in zip(
        values["track_id"],
        values["object_type"],
        values["timestep"],
        values["position_x"],
        values["position_y"],
    ):
        if track_id != focal_track_id or timestep not in selected_lookup:
            continue
        row = selected_lookup[timestep]
        positions[row] = torch.tensor([x, y], dtype=torch.float32)
        valid[row] = bool(np.isfinite(x) and np.isfinite(y))
        object_type = str(kind).lower()
    return positions, valid, OBJECT_TYPES.get(object_type, OBJECT_TYPES["unknown"])


def _fill_native_gaps(positions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Fill native gaps only for conditioning geometry; losses retain validity."""
    result = positions.clone()
    timeline = np.arange(positions.shape[1], dtype=np.float32)
    for row in range(positions.shape[0]):
        available = torch.where(valid[row])[0].cpu().numpy()
        if available.size == 0:
            continue
        for dim in range(2):
            values = positions[row, available, dim].cpu().numpy()
            result[row, :, dim] = torch.from_numpy(
                np.interp(timeline, available, values).astype(np.float32)
            )
    return result


def _load_or_build_raw_split(
    split_root: Path,
    scenario_ids: Sequence[str],
    cache_path: Path,
) -> RawAV2Split:
    expected_ids = list(scenario_ids)
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if (
            cached.get("requested_scenario_ids") == expected_ids
            and tuple(cached.get("selected_timesteps", ())) == SELECTED_TIMESTEPS
        ):
            return RawAV2Split(
                positions=cached["positions"].float(),
                native_valid=cached["native_valid"].bool(),
                object_type=cached["object_type"].long(),
                scenario_ids=expected_ids,
            )

    positions = []
    valid = []
    object_types = []
    kept_ids = []
    total = len(scenario_ids)
    for position_index, scenario_id in enumerate(scenario_ids, start=1):
        path = _scenario_parquet(split_root, scenario_id)
        try:
            focal_positions, focal_valid, object_type = _read_focal_track(path)
        except Exception as error:
            print(f"[AV2] skip unreadable scenario {scenario_id}: {error}", flush=True)
            continue
        # At least two native history points are required for a stable frame,
        # and every future evaluation point must have ground truth.
        if focal_valid[:8].sum() < 2 or not focal_valid[8:].all():
            continue
        positions.append(focal_positions)
        valid.append(focal_valid)
        object_types.append(object_type)
        kept_ids.append(scenario_id)
        if position_index % 500 == 0 or position_index == total:
            print(
                f"[AV2] {split_root.name}: read {position_index}/{total}, "
                f"kept {len(kept_ids)}",
                flush=True,
            )
    if not positions:
        raise RuntimeError(f"no usable focal tracks loaded from {split_root}")

    result = RawAV2Split(
        positions=torch.stack(positions),
        native_valid=torch.stack(valid),
        object_type=torch.tensor(object_types, dtype=torch.long),
        scenario_ids=kept_ids,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "positions": result.positions,
            "native_valid": result.native_valid,
            "object_type": result.object_type,
            "scenario_ids": result.scenario_ids,
            "requested_scenario_ids": expected_ids,
            "selected_timesteps": SELECTED_TIMESTEPS,
        },
        cache_path,
    )
    return result


def _artificial_mask(
    scenario_id: str,
    native_history_valid: torch.Tensor,
    mode: str,
    ratio: float,
    seed: int,
) -> torch.Tensor:
    available = torch.where(native_history_valid)[0].cpu().numpy()
    result = torch.zeros(8, dtype=torch.bool)
    if available.size <= 2 or ratio <= 0:
        return result
    missing_count = min(
        max(1, int(round(float(ratio) * available.size))),
        available.size - 2,
    )
    rng = np.random.default_rng(_stable_u64(seed, scenario_id, mode, ratio))
    selected_mode = _selected_mask_mode(scenario_id, mode, seed)
    if selected_mode == "random":
        chosen = rng.choice(available, size=missing_count, replace=False)
    elif selected_mode == "block":
        start = int(rng.integers(0, available.size - missing_count + 1))
        chosen = available[start : start + missing_count]
    elif selected_mode == "tail":
        chosen = available[-missing_count:]
    else:
        raise ValueError(f"unknown AV2 mask mode: {mode}")
    result[torch.as_tensor(chosen, dtype=torch.long)] = True
    return result


def _selected_mask_mode(scenario_id: str, mode: str, seed: int) -> str:
    if mode != "mixed":
        return mode
    return ("random", "block", "tail")[
        _stable_u64(seed, scenario_id, "mode") % 3
    ]


class AV2TrajectoryDataset(Dataset):
    def __init__(
        self,
        raw: RawAV2Split,
        mask_mode: str,
        mask_ratio: float,
        mask_seed: int,
        coordinate_scale: float | None = None,
    ) -> None:
        filled = _fill_native_gaps(raw.positions.float(), raw.native_valid.bool())
        artificial_missing = torch.stack(
            [
                _artificial_mask(
                    scenario_id=scenario_id,
                    native_history_valid=native_valid[:8],
                    mode=mask_mode,
                    ratio=mask_ratio,
                    seed=mask_seed,
                )
                for scenario_id, native_valid in zip(
                    raw.scenario_ids, raw.native_valid
                )
            ]
        )
        native_history = raw.native_valid[:, :8].bool()
        seen = native_history & ~artificial_missing
        history_with_nan = filled[:, :8].clone()
        history_with_nan[~seen] = torch.nan
        centers, rotations = _relative_rotation(history_with_nan, seen)
        target = _canonicalize(filled, centers, rotations)
        partial = _canonicalize(filled[:, :8], centers, rotations)
        partial = torch.where(seen[..., None], partial, torch.zeros_like(partial))

        if coordinate_scale is None:
            valid_values = target[raw.native_valid.bool()]
            coordinate_scale = float(valid_values.std().clamp_min(1e-3).item())
        self.scale = float(coordinate_scale)
        self.partial = partial / self.scale
        self.target = target / self.scale
        self.seen = seen
        self.impute_mask = artificial_missing & native_history
        self.future_valid = raw.native_valid[:, 8:].bool()
        self.unknown = torch.cat([self.impute_mask, self.future_valid], dim=1)
        self.keys = build_engineered_keys(self.partial, self.seen)
        self.object_type = raw.object_type.long()
        self.scenario_ids = raw.scenario_ids
        self.native_valid = raw.native_valid.bool()
        self.mask_mode = mask_mode
        self.mask_ratio = float(mask_ratio)
        self.mask_seed = int(mask_seed)
        self.selected_mask_modes = [
            _selected_mask_mode(scenario_id, mask_mode, mask_seed)
            for scenario_id in raw.scenario_ids
        ]

    def __len__(self) -> int:
        return self.target.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "partial": self.partial[index],
            "target": self.target[index],
            "seen": self.seen[index],
            "unknown": self.unknown[index],
            "impute_mask": self.impute_mask[index],
            "future_valid": self.future_valid[index],
            "key": self.keys[index],
            "index": torch.tensor(index, dtype=torch.long),
            "object_type": self.object_type[index],
        }


def _write_protocol(
    path: Path,
    *,
    dataset_root: Path,
    train: AV2TrajectoryDataset,
    val: AV2TrajectoryDataset,
    test: AV2TrajectoryDataset,
) -> None:
    def split_statistics(dataset: AV2TrajectoryDataset) -> dict[str, object]:
        scenario_digest = hashlib.sha256(
            "\n".join(dataset.scenario_ids).encode("utf-8")
        ).hexdigest()
        return {
            "count": len(dataset),
            "scenario_id_sha256": scenario_digest,
            "object_type_counts": dict(
                sorted(
                    Counter(
                        OBJECT_TYPE_NAMES.get(int(value), "unknown")
                        for value in dataset.object_type.tolist()
                    ).items()
                )
            ),
            "selected_mask_mode_counts": dict(
                sorted(Counter(dataset.selected_mask_modes).items())
            ),
            "artificial_missing_history_point_counts": dict(
                sorted(
                    Counter(
                        map(int, dataset.impute_mask.sum(dim=1).tolist())
                    ).items()
                )
            ),
            "native_valid_history_point_counts": dict(
                sorted(
                    Counter(
                        map(int, dataset.native_valid[:, :8].sum(dim=1).tolist())
                    ).items()
                )
            ),
            "native_valid_future_point_counts": dict(
                sorted(
                    Counter(
                        map(int, dataset.native_valid[:, 8:].sum(dim=1).tolist())
                    ).items()
                )
            ),
        }

    payload = {
        "dataset": "Argoverse 2 Motion Forecasting",
        "dataset_root": str(dataset_root),
        "input_modalities": ["focal-agent vector trajectory"],
        "excluded_modalities": ["camera", "image", "video", "lidar", "map_json"],
        "history_timesteps": list(HISTORY_TIMESTEPS),
        "future_timesteps": list(FUTURE_TIMESTEPS),
        "mask": {
            "mode": train.mask_mode,
            "ratio": train.mask_ratio,
            "seed": train.mask_seed,
            "online": True,
            "native_validity_separate": True,
        },
        "split_policy": {
            "train": "official train; memory values restricted to this split",
            "validation": "hash-selected first partition of official val",
            "test": "disjoint hash-selected second partition of official val",
            "official_test": "unused because public ground truth is unavailable",
        },
        "counts": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
        },
        "split_statistics": {
            "train": split_statistics(train),
            "validation": split_statistics(val),
            "test": split_statistics(test),
        },
        "coordinate_scale": train.scale,
        "overlap_checks": {
            "train_validation": len(set(train.scenario_ids) & set(val.scenario_ids)),
            "train_test": len(set(train.scenario_ids) & set(test.scenario_ids)),
            "validation_test": len(set(val.scenario_ids) & set(test.scenario_ids)),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_av2_bundle(
    repo_root: Path,
    av2_root: str | Path,
    av2_train_size: int = 8192,
    av2_val_size: int = 1024,
    av2_test_size: int = 1024,
    av2_test_offset: int | None = None,
    av2_selection_seed: int = 2026,
    mask_mode: str = "mixed",
    mask_ratio: float = 0.5,
    mask_seed: int = 42,
    av2_cache_root: str | Path | None = None,
    protocol_output: str | Path | None = None,
    **_: object,
) -> JointDataBundle:
    dataset_root = Path(av2_root).expanduser().resolve()
    train_root = dataset_root / "data" / "train"
    val_root = dataset_root / "data" / "val"
    cache_root = (
        Path(av2_cache_root)
        if av2_cache_root
        else repo_root / "data" / "av2_joint_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    test_offset = av2_val_size if av2_test_offset is None else int(av2_test_offset)
    if test_offset < av2_val_size:
        raise ValueError("av2_test_offset must not overlap the validation partition")

    cache_tag = f"s{av2_selection_seed}_n{av2_train_size}_v{av2_val_size}_t{av2_test_size}"
    if test_offset != av2_val_size:
        cache_tag += f"_o{test_offset}"
    train_cache = cache_root / f"train_{cache_tag}.pt"
    val_cache = cache_root / f"val_{cache_tag}.pt"
    test_cache = cache_root / f"test_{cache_tag}.pt"

    if train_cache.is_file() and val_cache.is_file() and test_cache.is_file():
        train_ids = torch.load(train_cache, map_location="cpu")["requested_scenario_ids"]
        val_ids = torch.load(val_cache, map_location="cpu")["requested_scenario_ids"]
        test_ids = torch.load(test_cache, map_location="cpu")["requested_scenario_ids"]
    else:
        train_ids = _select_scenarios(
            train_root, av2_train_size, av2_selection_seed
        )
        heldout_ids = _select_scenarios(
            val_root, test_offset + av2_test_size, av2_selection_seed + 1
        )
        val_ids = heldout_ids[:av2_val_size]
        test_ids = heldout_ids[test_offset : test_offset + av2_test_size]
        cache_tag = (
            f"s{av2_selection_seed}_n{len(train_ids)}_"
            f"v{len(val_ids)}_t{len(test_ids)}"
        )
        if test_offset != av2_val_size:
            cache_tag += f"_o{test_offset}"
        train_cache = cache_root / f"train_{cache_tag}.pt"
        val_cache = cache_root / f"val_{cache_tag}.pt"
        test_cache = cache_root / f"test_{cache_tag}.pt"

    raw_train = _load_or_build_raw_split(train_root, train_ids, train_cache)
    raw_val = _load_or_build_raw_split(val_root, val_ids, val_cache)
    raw_test = _load_or_build_raw_split(val_root, test_ids, test_cache)

    train = AV2TrajectoryDataset(
        raw_train, mask_mode, mask_ratio, mask_seed, coordinate_scale=None
    )
    val = AV2TrajectoryDataset(
        raw_val, mask_mode, mask_ratio, mask_seed, coordinate_scale=train.scale
    )
    test = AV2TrajectoryDataset(
        raw_test, mask_mode, mask_ratio, mask_seed, coordinate_scale=train.scale
    )
    if set(train.scenario_ids) & set(val.scenario_ids):
        raise RuntimeError("AV2 train/validation scenario leakage")
    if set(train.scenario_ids) & set(test.scenario_ids):
        raise RuntimeError("AV2 train/test scenario leakage")
    if set(val.scenario_ids) & set(test.scenario_ids):
        raise RuntimeError("AV2 validation/test scenario leakage")

    if protocol_output:
        _write_protocol(
            Path(protocol_output),
            dataset_root=dataset_root,
            train=train,
            val=val,
            test=test,
        )
    return JointDataBundle(
        train=train,
        val=val,
        test=test,
        coordinate_scale=train.scale,
    )
