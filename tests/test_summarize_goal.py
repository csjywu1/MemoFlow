from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.summarize import run_metadata


class RunMetadataTest(unittest.TestCase):
    def test_records_complete_traceability_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "full"
            artifact_dir = run_dir / "ETH_Easy_flow_memory_seed42"
            artifact_dir.mkdir(parents=True)
            config = {
                "dataset": "ETH",
                "difficulty": "Easy",
                "model": "flow",
                "use_memory": True,
                "seed": 42,
                "mask_mode": "mixed",
                "mask_ratio": 0.5,
                "mask_seed": 42,
                "batch_size": 8,
                "grad_accum_steps": 32,
            }
            summary = {
                "run": "ETH_Easy_flow_memory_seed42",
                "device": "cuda",
                "best_epoch": 7,
                "best_val_score": 0.4,
                "peak_gpu_memory_mib": 80.0,
                "peak_gpu_reserved_mib": 96.0,
            }
            (artifact_dir / "config.json").write_text(json.dumps(config))
            (artifact_dir / "summary.json").write_text(json.dumps(summary))
            (artifact_dir / "history.csv").write_text(
                "epoch,minADE,minFDE\n7,0.3,0.5\n"
            )
            (artifact_dir / "best.pt").write_bytes(b"checkpoint")

            metadata = run_metadata(run_dir)

            self.assertEqual(metadata["selection"]["best_epoch"], 7)
            self.assertEqual(metadata["selection"]["best_val_score"], 0.4)
            self.assertEqual(metadata["protocol"]["dataset"], "ETH")
            self.assertTrue(metadata["protocol"]["use_memory"])
            self.assertEqual(metadata["artifacts"]["checkpoint_size_bytes"], 10)
            self.assertTrue(
                metadata["artifacts"]["checkpoint"].endswith("best.pt")
            )
            self.assertEqual(metadata["full_config"], config)


if __name__ == "__main__":
    unittest.main()
