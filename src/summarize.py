from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict


def composite(ade: float, fde: float) -> float:
    return 0.5 * (ade + fde)


def best_validation(run_dir: Path) -> Dict[str, float]:
    history_path = next(run_dir.glob("*/history.csv"))
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    best = min(rows, key=lambda row: composite(float(row["minADE"]), float(row["minFDE"])))
    result = {
        "epoch": int(best["epoch"]),
        "minADE": float(best["minADE"]),
        "minFDE": float(best["minFDE"]),
        "score": composite(float(best["minADE"]), float(best["minFDE"])),
    }
    for key, value in best.items():
        if key in {"epoch", "train_loss", "minADE", "minFDE"}:
            continue
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def test_metrics(run_dir: Path) -> Dict[str, float]:
    summary_path = next(run_dir.glob("*/summary.json"))
    summary = json.loads(summary_path.read_text())
    result = {
        "minADE": float(summary["test_minADE"]),
        "minFDE": float(summary["test_minFDE"]),
        "score": composite(
            float(summary["test_minADE"]),
            float(summary["test_minFDE"]),
        ),
    }
    for key, value in summary.items():
        if not key.startswith("test_") or key in {"test_minADE", "test_minFDE"}:
            continue
        try:
            result[key.removeprefix("test_")] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def run_metadata(run_dir: Path) -> Dict[str, object]:
    summary_path = next(run_dir.glob("*/summary.json"))
    config_path = next(run_dir.glob("*/config.json"))
    history_path = next(run_dir.glob("*/history.csv"))
    checkpoint_path = next(run_dir.glob("*/best.pt"))
    summary = json.loads(summary_path.read_text())
    config = json.loads(config_path.read_text())
    return {
        "run": summary.get("run"),
        "device": summary.get("device"),
        "artifacts": {
            "config": str(config_path),
            "history": str(history_path),
            "summary": str(summary_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        },
        "selection": {
            "best_epoch": int(summary["best_epoch"]),
            "best_val_score": float(summary["best_val_score"]),
        },
        "protocol": {
            "dataset": config.get("dataset"),
            "difficulty": config.get("difficulty"),
            "model": config.get("model"),
            "use_memory": config.get("use_memory"),
            "seed": config.get("seed"),
            "mask_mode": config.get("mask_mode"),
            "mask_ratio": config.get("mask_ratio"),
            "mask_seed": config.get("mask_seed"),
        },
        "peak_gpu_memory_mib": float(summary.get("peak_gpu_memory_mib", 0.0)),
        "peak_gpu_reserved_mib": float(
            summary.get("peak_gpu_reserved_mib", 0.0)
        ),
        "resource_config": {
            key: config.get(key)
            for key in (
                "amp",
                "batch_size",
                "batch_sleep_ms",
                "grad_accum_steps",
                "gpu_memory_fraction",
                "hidden_dim",
                "layers",
                "heads",
                "num_workers",
            )
        },
        "full_config": config,
    }


def gain(baseline: float, full: float) -> float:
    return 100.0 * (baseline - full) / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    directories = {
        "full": args.root / "full",
        "memory_cvae": args.root / "memory_cvae",
        "no_memory": args.root / "no_memory",
    }
    result: Dict[str, object] = {
        "threshold_pct": args.threshold,
        "validation": {},
        "test": {},
        "runs": {
            name: run_metadata(path)
            for name, path in directories.items()
        },
    }
    for split, loader in (("validation", best_validation), ("test", test_metrics)):
        metrics = {name: loader(path) for name, path in directories.items()}
        metrics["memory_gain_pct"] = gain(
            metrics["no_memory"]["score"],
            metrics["full"]["score"],
        )
        metrics["flow_gain_pct"] = gain(
            metrics["memory_cvae"]["score"],
            metrics["full"]["score"],
        )
        metrics["passes"] = bool(
            metrics["memory_gain_pct"] >= args.threshold
            and metrics["flow_gain_pct"] >= args.threshold
        )
        result[split] = metrics
    result["passes"] = bool(
        result["validation"]["passes"] and result["test"]["passes"]
    )

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if not result["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
