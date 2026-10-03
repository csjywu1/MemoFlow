from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from memoflow.av2_data import (
    AV2TrajectoryDataset,
    RawAV2Split,
    _artificial_mask,
    load_av2_bundle,
)


def _write_scenario(root: Path, split: str, scenario_id: str, offset: float) -> None:
    directory = root / "data" / split / scenario_id
    directory.mkdir(parents=True)
    timesteps = list(range(110))
    table = pa.table(
        {
            "track_id": ["focal"] * 110,
            "object_type": ["vehicle"] * 110,
            "timestep": timesteps,
            "position_x": [offset + 0.4 * step for step in timesteps],
            "position_y": [offset + 0.1 * step for step in timesteps],
            "focal_track_id": ["focal"] * 110,
        }
    )
    pq.write_table(table, directory / f"scenario_{scenario_id}.parquet")


class AV2MaskTest(unittest.TestCase):
    def test_all_mask_modes_and_ratios_are_deterministic(self) -> None:
        valid = torch.ones(8, dtype=torch.bool)
        native_missing = valid.clone()
        native_missing[3] = False
        for mode in ("random", "block", "tail"):
            for ratio in (0.25, 0.5, 0.75):
                with self.subTest(mode=mode, ratio=ratio):
                    first = _artificial_mask(
                        "scenario-a", valid, mode, ratio, 42
                    )
                    second = _artificial_mask(
                        "scenario-a", valid, mode, ratio, 42
                    )
                    mask_with_native_gap = _artificial_mask(
                        "scenario-a", native_missing, mode, ratio, 42
                    )
                    self.assertTrue(torch.equal(first, second))
                    self.assertEqual(int(first.sum()), round(8 * ratio))
                    self.assertGreaterEqual(int((valid & ~first).sum()), 2)
                    self.assertFalse(bool(mask_with_native_gap[3]))

    def test_native_missing_is_not_an_imputation_target(self) -> None:
        positions = torch.arange(40, dtype=torch.float32).reshape(1, 20, 2)
        native_valid = torch.ones(1, 20, dtype=torch.bool)
        native_valid[0, 3] = False
        raw = RawAV2Split(
            positions=positions,
            native_valid=native_valid,
            object_type=torch.tensor([0]),
            scenario_ids=["scenario-b"],
        )
        dataset = AV2TrajectoryDataset(
            raw,
            mask_mode="tail",
            mask_ratio=0.5,
            mask_seed=42,
        )
        self.assertFalse(bool(dataset.impute_mask[0, 3]))
        self.assertFalse(bool(dataset.unknown[0, 3]))
        self.assertTrue(torch.equal(dataset.unknown[0, 8:], native_valid[0, 8:]))


class AV2BundleTest(unittest.TestCase):
    def test_bundle_has_disjoint_splits_and_no_map_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary) / "av2"
            repo_root = Path(temporary) / "repo"
            for index in range(8):
                _write_scenario(
                    dataset_root,
                    "train",
                    f"train-{index:02d}",
                    float(index),
                )
            for index in range(6):
                _write_scenario(
                    dataset_root,
                    "val",
                    f"val-{index:02d}",
                    float(index + 100),
                )
            protocol = repo_root / "results" / "protocol.json"
            bundle = load_av2_bundle(
                repo_root=repo_root,
                av2_root=dataset_root,
                av2_train_size=6,
                av2_val_size=2,
                av2_test_size=2,
                av2_selection_seed=7,
                mask_mode="block",
                mask_ratio=0.5,
                mask_seed=11,
                protocol_output=protocol,
            )
            self.assertEqual(len(bundle.train), 6)
            self.assertEqual(len(bundle.val), 2)
            self.assertEqual(len(bundle.test), 2)
            self.assertEqual(bundle.train.target.shape[1:], (20, 2))
            self.assertTrue((bundle.train.impute_mask.sum(dim=1) == 4).all())
            self.assertFalse(
                set(bundle.val.scenario_ids) & set(bundle.test.scenario_ids)
            )
            self.assertTrue(protocol.is_file())
            protocol_value = json.loads(protocol.read_text())
            train_stats = protocol_value["split_statistics"]["train"]
            self.assertEqual(train_stats["count"], 6)
            self.assertEqual(train_stats["object_type_counts"], {"vehicle": 6})
            self.assertEqual(
                train_stats["artificial_missing_history_point_counts"],
                {"4": 6},
            )
            self.assertEqual(
                len(train_stats["scenario_id_sha256"]),
                64,
            )


if __name__ == "__main__":
    unittest.main()
