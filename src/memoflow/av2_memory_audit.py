from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memoflow.data import load_bundle
from memoflow.model import TrajectoryMemory


def _empty_sums() -> Dict[str, float]:
    return {
        "top1_ADE": 0.0,
        "top1_FDE": 0.0,
        "top1_impute_ADE": 0.0,
        "best_of_k_ADE": 0.0,
        "best_of_k_FDE": 0.0,
        "best_of_k_impute_ADE": 0.0,
        "context_ADE": 0.0,
        "context_FDE": 0.0,
        "context_impute_ADE": 0.0,
        "top1_similarity": 0.0,
        "top1_object_type_match": 0.0,
    }


@torch.no_grad()
def audit_split(
    dataset,
    memory: TrajectoryMemory,
    train_object_type: torch.Tensor,
    batch_size: int,
    coordinate_scale: float,
) -> Dict[str, float]:
    sums = _empty_sums()
    rows = 0
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        keys = dataset.keys[start:stop]
        target = dataset.target[start:stop]
        impute_mask = dataset.impute_mask[start:stop].float()
        query_type = dataset.object_type[start:stop]
        result = memory.retrieve(keys)
        candidates = result.values
        top1 = candidates[:, 0]
        context = result.context

        future_error = torch.linalg.vector_norm(
            candidates[:, :, 8:] - target[:, None, 8:],
            dim=-1,
        )
        candidate_ade = future_error.mean(dim=-1)
        candidate_fde = future_error[:, :, -1]
        top1_history_error = torch.linalg.vector_norm(
            top1[:, :8] - target[:, :8], dim=-1
        )
        candidate_history_error = torch.linalg.vector_norm(
            candidates[:, :, :8] - target[:, None, :8], dim=-1
        )
        missing_count = impute_mask.sum(dim=-1).clamp_min(1)
        top1_impute = (
            top1_history_error * impute_mask
        ).sum(dim=-1) / missing_count
        candidate_impute = (
            candidate_history_error * impute_mask[:, None]
        ).sum(dim=-1) / missing_count[:, None]

        context_future_error = torch.linalg.vector_norm(
            context[:, 8:] - target[:, 8:], dim=-1
        )
        context_history_error = torch.linalg.vector_norm(
            context[:, :8] - target[:, :8], dim=-1
        )
        context_impute = (
            context_history_error * impute_mask
        ).sum(dim=-1) / missing_count
        top1_type = train_object_type[result.indices[:, 0]]

        sums["top1_ADE"] += float(candidate_ade[:, 0].sum())
        sums["top1_FDE"] += float(candidate_fde[:, 0].sum())
        sums["top1_impute_ADE"] += float(top1_impute.sum())
        sums["best_of_k_ADE"] += float(candidate_ade.min(dim=1).values.sum())
        sums["best_of_k_FDE"] += float(candidate_fde.min(dim=1).values.sum())
        sums["best_of_k_impute_ADE"] += float(
            candidate_impute.min(dim=1).values.sum()
        )
        sums["context_ADE"] += float(context_future_error.mean(dim=-1).sum())
        sums["context_FDE"] += float(context_future_error[:, -1].sum())
        sums["context_impute_ADE"] += float(context_impute.sum())
        sums["top1_similarity"] += float(
            torch.nn.functional.cosine_similarity(
                keys, memory.keys[result.indices[:, 0]], dim=-1
            ).sum()
        )
        sums["top1_object_type_match"] += float(
            (top1_type == query_type).float().sum()
        )
        rows += stop - start
    result = {
        key: value / max(rows, 1)
        for key, value in sums.items()
    }
    for key in list(result):
        if key.endswith(("ADE", "FDE")):
            result[key] *= coordinate_scale
    result["rows"] = rows
    result["top_k"] = memory.top_k
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--av2-root",
        default="data/AV2",
    )
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument(
        "--mask-mode",
        choices=["random", "block", "tail", "mixed"],
        default="mixed",
    )
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/memoflow_av2_low_vram_v1/memory_audit.json"
        ),
    )
    args = parser.parse_args()

    bundle = load_bundle(
        REPO_ROOT,
        "AV2",
        "Easy",
        av2_root=args.av2_root,
        av2_train_size=args.train_size,
        av2_val_size=args.val_size,
        av2_test_size=args.test_size,
        av2_selection_seed=args.selection_seed,
        mask_mode=args.mask_mode,
        mask_ratio=args.mask_ratio,
        mask_seed=args.mask_seed,
    )
    train_ids = set(bundle.train.scenario_ids)
    val_ids = set(bundle.val.scenario_ids)
    test_ids = set(bundle.test.scenario_ids)
    overlap = {
        "train_validation": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "validation_test": len(val_ids & test_ids),
    }
    if any(overlap.values()):
        raise RuntimeError(f"scenario leakage detected: {overlap}")

    memory = TrajectoryMemory(
        bundle.train.keys,
        bundle.train.target,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    report = {
        "dataset": "Argoverse 2 Motion Forecasting focal trajectories",
        "device": "cpu",
        "mask_mode": args.mask_mode,
        "mask_ratio": args.mask_ratio,
        "mask_seed": args.mask_seed,
        "selection_seed": args.selection_seed,
        "memory_rows": len(bundle.train),
        "overlap_checks": overlap,
        "validation": audit_split(
            bundle.val,
            memory,
            bundle.train.object_type,
            args.batch_size,
            bundle.coordinate_scale,
        ),
        "test": audit_split(
            bundle.test,
            memory,
            bundle.train.object_type,
            args.batch_size,
            bundle.coordinate_scale,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
