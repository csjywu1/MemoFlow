from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from memoflow.av2_data import _artificial_mask


RUNS = {
    "no_memory": {"model": "flow", "use_memory": False},
    "memory_cvae": {"model": "cvae", "use_memory": True},
    "full": {"model": "flow", "use_memory": True},
}
CORE_TEST_METRICS = {
    "test_minADE",
    "test_minFDE",
    "test_meanADE",
    "test_impute_minADE",
}
TYPE_METRICS = ("minADE", "minFDE", "meanADE", "impute_minADE", "count")


def exactly_one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern!r} below {directory}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--max-peak-mib", type=float, default=1536.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(condition),
                "evidence": evidence,
            }
        )

    goal_path = args.root / "goal_summary.json"
    protocol_path = args.root / "protocol_AV2_mixed_50_seed42.json"
    memory_audit_path = args.root / "memory_audit.json"
    goal = json.loads(goal_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    memory_audit = json.loads(memory_audit_path.read_text())

    check("goal summary passes", goal.get("passes") is True, goal.get("passes"))
    for split in ("validation", "test"):
        split_result = goal[split]
        gains = {
            "memory_gain_pct": split_result["memory_gain_pct"],
            "flow_gain_pct": split_result["flow_gain_pct"],
        }
        check(
            f"{split} gains meet threshold",
            split_result.get("passes") is True
            and all(value >= args.threshold for value in gains.values()),
            gains,
        )

    expected_counts = {"train": 8192, "validation": 1024, "test": 1024}
    check(
        "dataset cardinalities",
        protocol.get("counts") == expected_counts,
        protocol.get("counts"),
    )
    check(
        "trajectory-only input",
        protocol.get("input_modalities") == ["focal-agent vector trajectory"]
        and {"camera", "image", "video", "lidar", "map_json"}.issubset(
            set(protocol.get("excluded_modalities", []))
        ),
        {
            "input": protocol.get("input_modalities"),
            "excluded": protocol.get("excluded_modalities"),
        },
    )
    mask = protocol.get("mask", {})
    check(
        "formal mask configuration",
        mask.get("mode") == "mixed"
        and mask.get("ratio") == 0.5
        and mask.get("seed") == 42
        and mask.get("online") is True
        and mask.get("native_validity_separate") is True,
        mask,
    )
    overlaps = protocol.get("overlap_checks", {})
    check(
        "split leakage is zero",
        bool(overlaps) and all(value == 0 for value in overlaps.values()),
        overlaps,
    )
    for split, stats in protocol["split_statistics"].items():
        mode_counts = stats["selected_mask_mode_counts"]
        missing_counts = stats["artificial_missing_history_point_counts"]
        check(
            f"{split} mixed mask covers random/block/tail",
            set(mode_counts) == {"random", "block", "tail"}
            and sum(mode_counts.values()) == expected_counts[split],
            mode_counts,
        )
        check(
            f"{split} 50% mask means four missing history points",
            missing_counts == {"4": expected_counts[split]},
            missing_counts,
        )

    mask_sweep: dict[str, Any] = {}
    valid = torch.ones(8, dtype=torch.bool)
    native_missing = valid.clone()
    native_missing[3] = False
    for mode in ("random", "block", "tail"):
        for ratio in (0.25, 0.5, 0.75):
            key = f"{mode}_{int(ratio * 100)}"
            first = _artificial_mask("audit-scenario", valid, mode, ratio, 42)
            second = _artificial_mask("audit-scenario", valid, mode, ratio, 42)
            with_native_gap = _artificial_mask(
                "audit-scenario", native_missing, mode, ratio, 42
            )
            expected_missing = round(8 * ratio)
            mask_sweep[key] = {
                "deterministic": bool(torch.equal(first, second)),
                "missing_points": int(first.sum()),
                "expected_missing_points": expected_missing,
                "visible_points": int((valid & ~first).sum()),
                "native_gap_not_targeted": not bool(with_native_gap[3]),
            }
    check(
        "random/block/tail x 25/50/75 protocol sweep",
        all(
            item["deterministic"]
            and item["missing_points"] == item["expected_missing_points"]
            and item["visible_points"] >= 2
            and item["native_gap_not_targeted"]
            for item in mask_sweep.values()
        ),
        mask_sweep,
    )

    memory_overlaps = memory_audit.get("overlap_checks", {})
    check(
        "memory uses leak-free train split",
        bool(memory_overlaps)
        and all(value == 0 for value in memory_overlaps.values()),
        memory_overlaps,
    )

    run_evidence: dict[str, Any] = {}
    test_types = set(protocol["split_statistics"]["test"]["object_type_counts"])
    for name, expected in RUNS.items():
        directory = args.root / name
        config_path = exactly_one(directory, "*/config.json")
        history_path = exactly_one(directory, "*/history.csv")
        summary_path = exactly_one(directory, "*/summary.json")
        checkpoint_path = exactly_one(directory, "*/best.pt")
        config = json.loads(config_path.read_text())
        summary = json.loads(summary_path.read_text())
        with history_path.open(newline="") as handle:
            history = list(csv.DictReader(handle))

        common_ok = (
            config.get("dataset") == "AV2"
            and config.get("device") == "cuda"
            and config.get("amp") is True
            and config.get("batch_size") == 8
            and config.get("grad_accum_steps") == 16
            and config.get("gpu_memory_fraction") == 0.01
            and config.get("batch_sleep_ms") == 100.0
            and config.get("num_workers") == 0
            and config.get("av2_train_size") == 8192
            and config.get("av2_val_size") == 1024
            and config.get("av2_test_size") == 1024
            and config.get("mask_mode") == "mixed"
            and config.get("mask_ratio") == 0.5
            and config.get("mask_seed") == 42
        )
        identity_ok = (
            config.get("model") == expected["model"]
            and config.get("use_memory") is expected["use_memory"]
        )
        check(f"{name} resource and data configuration", common_ok, config)
        check(f"{name} ablation identity", identity_ok, expected)
        check(f"{name} has nonempty history", bool(history), len(history))
        check(
            f"{name} checkpoint exists",
            checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0,
            checkpoint_path.stat().st_size,
        )
        check(
            f"{name} core test metrics",
            CORE_TEST_METRICS.issubset(summary),
            sorted(set(summary) & CORE_TEST_METRICS),
        )
        missing_type_metrics = sorted(
            f"test_type_{object_type}_{metric}"
            for object_type in test_types
            for metric in TYPE_METRICS
            if f"test_type_{object_type}_{metric}" not in summary
        )
        check(
            f"{name} object-type test metrics",
            not missing_type_metrics,
            {"types": sorted(test_types), "missing": missing_type_metrics},
        )
        validation_fields = set(history[0]) if history else set()
        missing_val_type_metrics = sorted(
            f"type_{object_type}_{metric}"
            for object_type in test_types
            for metric in TYPE_METRICS
            if f"type_{object_type}_{metric}" not in validation_fields
        )
        check(
            f"{name} object-type validation metrics",
            not missing_val_type_metrics,
            {"types": sorted(test_types), "missing": missing_val_type_metrics},
        )
        peak_allocated = float(summary.get("peak_gpu_memory_mib", 0.0))
        peak_reserved = float(summary.get("peak_gpu_reserved_mib", 0.0))
        check(
            f"{name} CUDA peak is present and below cap",
            summary.get("device") == "cuda"
            and 0.0 < peak_allocated <= args.max_peak_mib
            and 0.0 < peak_reserved <= args.max_peak_mib,
            {
                "allocated_mib": peak_allocated,
                "reserved_mib": peak_reserved,
                "limit_mib": args.max_peak_mib,
            },
        )
        run_evidence[name] = {
            "config": str(config_path),
            "history": str(history_path),
            "summary": str(summary_path),
            "checkpoint": str(checkpoint_path),
            "epochs": len(history),
            "peak_gpu_memory_mib": peak_allocated,
            "peak_gpu_reserved_mib": peak_reserved,
        }

    passed = all(item["passed"] for item in checks)
    result = {
        "passed": passed,
        "threshold_pct": args.threshold,
        "max_peak_mib": args.max_peak_mib,
        "checks": checks,
        "runs": run_evidence,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    output = args.output or args.root / "goal_audit.json"
    output.write_text(rendered + "\n")
    print(rendered)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
