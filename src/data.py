from __future__ import annotations

import hashlib
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset


_RETRIEVAL_KEY_VARIANT = "legacy"


def set_retrieval_key_variant(variant: str) -> None:
    """Select the deterministic retrieval-key construction used by new datasets."""
    global _RETRIEVAL_KEY_VARIANT
    allowed = {
        "legacy",
        "suffix_motion25",
        "suffix_motion50",
        "suffix_nomask50",
        "suffix_block",
        "hybrid_motion25",
        "hybrid_motion50",
        "hybrid_nomask50",
        "hybrid_block",
    }
    if variant not in allowed:
        raise ValueError(f"unknown retrieval-key variant {variant!r}; expected one of {sorted(allowed)}")
    _RETRIEVAL_KEY_VARIANT = variant


@dataclass
class JointDataBundle:
    train: "JointTrajectoryDataset"
    val: "JointTrajectoryDataset"
    test: "JointTrajectoryDataset"
    coordinate_scale: float


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    """Load cached records written by NumPy 1.x/2.x interchangeably.

    The trajectory pickle files can contain NumPy internals whose module path
    changed from ``numpy.core`` to ``numpy._core``. Mapping the private path
    back to the public compatibility path keeps the data artifact unchanged
    while allowing every experiment environment to read the same split.
    """

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_missing_file(path: Path) -> Dict:
    with path.open("rb") as handle:
        return _NumpyCompatibleUnpickler(handle).load()


def _relative_rotation(
    obs_with_nan: torch.Tensor,
    seen: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return last-seen centers and row-vector rotations into heading frame."""
    count, _, _ = obs_with_nan.shape
    centers = torch.zeros(count, 2, dtype=torch.float32)
    rotations = torch.eye(2, dtype=torch.float32).repeat(count, 1, 1)
    safe_obs = torch.nan_to_num(obs_with_nan.float(), nan=0.0)
    for row in range(count):
        visible = torch.where(seen[row])[0]
        if visible.numel() == 0:
            continue
        centers[row] = safe_obs[row, visible[-1]]
        if visible.numel() < 2:
            continue
        displacement = safe_obs[row, visible[-1]] - safe_obs[row, visible[-2]]
        if torch.linalg.vector_norm(displacement) < 1e-6:
            continue
        angle = torch.atan2(displacement[1], displacement[0])
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        rotations[row] = torch.tensor(
            [[cosine, -sine], [sine, cosine]],
            dtype=torch.float32,
        )
    return centers, rotations


def _canonicalize(
    points: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    centered = points.float() - centers[:, None, :]
    return torch.einsum("ntd,ndk->ntk", centered, rotations)


def _interpolate_visible(
    partial: torch.Tensor,
    seen: torch.Tensor,
) -> torch.Tensor:
    """Linear interpolation using visible points only; endpoints use nearest value."""
    result = torch.zeros_like(partial)
    timeline = np.arange(partial.shape[1], dtype=np.float32)
    partial_np = partial.cpu().numpy()
    seen_np = seen.cpu().numpy()
    for row in range(partial.shape[0]):
        visible = np.flatnonzero(seen_np[row])
        if visible.size == 0:
            continue
        for dim in range(2):
            result[row, :, dim] = torch.from_numpy(
                np.interp(timeline, visible, partial_np[row, visible, dim]).astype(np.float32)
            )
    return result


def _motion_extrapolate_visible(
    partial: torch.Tensor,
    seen: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate internal gaps and extrapolate endpoints from visible velocities.

    Unlike ``np.interp``, this does not turn the motion at a missing suffix into
    an artificial zero velocity.  Segment velocities are divided by their time
    gap, so non-consecutive visible samples remain comparable.
    """
    result = _interpolate_visible(partial, seen)
    mean_velocity = torch.zeros(partial.shape[0], 2, dtype=partial.dtype)
    final_velocity = torch.zeros_like(mean_velocity)
    mean_acceleration = torch.zeros_like(mean_velocity)
    speed_stats = torch.zeros_like(mean_velocity)
    for row in range(partial.shape[0]):
        visible = torch.where(seen[row])[0]
        if visible.numel() < 2:
            continue
        points = partial[row, visible]
        gaps = (visible[1:] - visible[:-1]).to(partial.dtype).unsqueeze(-1)
        segment_velocity = (points[1:] - points[:-1]) / gaps.clamp_min(1.0)
        mean_velocity[row] = (points[-1] - points[0]) / float(visible[-1] - visible[0])
        tail_count = min(3, segment_velocity.shape[0])
        final_velocity[row] = segment_velocity[-tail_count:].median(dim=0).values
        if segment_velocity.shape[0] > 1:
            mean_acceleration[row] = (segment_velocity[1:] - segment_velocity[:-1]).mean(dim=0)
        speed = torch.linalg.vector_norm(segment_velocity, dim=-1)
        speed_stats[row, 0] = speed.mean()
        speed_stats[row, 1] = speed.std(unbiased=False)

        first = int(visible[0])
        last = int(visible[-1])
        if first > 0:
            head_velocity = segment_velocity[:tail_count].median(dim=0).values
            offsets = torch.arange(first, 0, -1, dtype=partial.dtype).unsqueeze(-1)
            result[row, :first] = points[0] - offsets * head_velocity
        if last + 1 < partial.shape[1]:
            offsets = torch.arange(
                1,
                partial.shape[1] - last,
                dtype=partial.dtype,
            ).unsqueeze(-1)
            result[row, last + 1 :] = points[-1] + offsets * final_velocity[row]
    return result, mean_velocity, final_velocity, mean_acceleration, speed_stats


def build_engineered_keys(
    partial_canonical: torch.Tensor,
    seen: torch.Tensor,
) -> torch.Tensor:
    """Physics- and missingness-aware retrieval key."""
    legacy_interpolated = _interpolate_visible(partial_canonical, seen)
    legacy_velocity = legacy_interpolated[:, 1:] - legacy_interpolated[:, :-1]
    legacy_acceleration = legacy_velocity[:, 1:] - legacy_velocity[:, :-1]
    legacy_speed = torch.linalg.vector_norm(legacy_velocity, dim=-1)
    legacy_features = torch.cat(
        [
            legacy_interpolated.reshape(legacy_interpolated.shape[0], -1),
            seen.float(),
            legacy_velocity.mean(dim=1),
            legacy_velocity[:, -1],
            legacy_acceleration.mean(dim=1),
            torch.stack(
                [legacy_speed.mean(dim=1), legacy_speed.std(dim=1)], dim=-1
            ),
        ],
        dim=-1,
    )
    legacy_key = torch.nn.functional.normalize(legacy_features, dim=-1, eps=1e-6)
    if _RETRIEVAL_KEY_VARIANT == "legacy":
        return legacy_key

    variant = _RETRIEVAL_KEY_VARIANT
    hybrid = variant.startswith("hybrid_")
    if hybrid:
        variant = "suffix_" + variant.removeprefix("hybrid_")

    (
        interpolated,
        mean_velocity,
        final_velocity,
        mean_acceleration,
        speed_stats,
    ) = _motion_extrapolate_visible(partial_canonical, seen)

    position = torch.nn.functional.normalize(
        interpolated.reshape(interpolated.shape[0], -1), dim=-1, eps=1e-6
    )
    mask = torch.nn.functional.normalize(seen.float(), dim=-1, eps=1e-6)
    if variant == "suffix_block":
        motion = [
            0.40 * torch.nn.functional.normalize(mean_velocity, dim=-1, eps=1e-6),
            0.80 * torch.nn.functional.normalize(final_velocity, dim=-1, eps=1e-6),
            0.20 * torch.nn.functional.normalize(mean_acceleration, dim=-1, eps=1e-6),
            0.40 * torch.nn.functional.normalize(speed_stats, dim=-1, eps=1e-6),
        ]
        mask_weight = 0.05
    else:
        scale = 25.0 if variant == "suffix_motion25" else 50.0
        motion = [
            0.45 * torch.tanh(scale * mean_velocity),
            0.90 * torch.tanh(scale * final_velocity),
            0.25 * torch.tanh(scale * mean_acceleration),
            0.45 * torch.tanh(scale * speed_stats),
        ]
        mask_weight = 0.0 if variant == "suffix_nomask50" else 0.05
    features = torch.cat([0.75 * position, mask_weight * mask, *motion], dim=-1)
    modified_key = torch.nn.functional.normalize(features, dim=-1, eps=1e-6)
    if not hybrid:
        return modified_key
    suffix4 = seen[:, :4].all(dim=1) & (~seen[:, 4:]).all(dim=1)
    return torch.where(suffix4[:, None], modified_key, legacy_key)


class JointTrajectoryDataset(Dataset):
    def __init__(
        self,
        missing_data: Dict,
        full_history: torch.Tensor,
        coordinate_scale: float,
    ) -> None:
        obs = torch.as_tensor(missing_data["obs_traj"], dtype=torch.float32)
        future = torch.as_tensor(missing_data["pred_traj"], dtype=torch.float32)
        missing = torch.as_tensor(missing_data["missing_mask"]).bool()
        seen = ~missing[..., 0]
        if full_history.shape != obs.shape:
            raise ValueError(f"full history shape {full_history.shape} != observation shape {obs.shape}")
        finite = torch.isfinite(obs).all(dim=-1)
        if not torch.equal(finite, seen):
            raise ValueError("missing_mask and NaN pattern disagree")

        centers, rotations = _relative_rotation(obs, seen)
        full = torch.cat([full_history.float(), future], dim=1)
        full_canonical = _canonicalize(full, centers, rotations)
        partial_filled = torch.nan_to_num(obs, nan=0.0)
        partial_canonical = _canonicalize(partial_filled, centers, rotations)
        partial_canonical = torch.where(
            seen[..., None],
            partial_canonical,
            torch.zeros_like(partial_canonical),
        )

        self.scale = float(coordinate_scale)
        self.partial = partial_canonical / self.scale
        self.target = full_canonical / self.scale
        self.seen = seen
        self.unknown = torch.ones(len(obs), 20, dtype=torch.bool)
        self.unknown[:, :8] = ~seen
        self.impute_mask = ~seen
        self.future_valid = torch.ones(len(obs), 12, dtype=torch.bool)
        self.keys = build_engineered_keys(self.partial, self.seen)
        self.centers = centers
        self.rotations = rotations

    def __len__(self) -> int:
        return self.partial.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "partial": self.partial[index],
            "target": self.target[index],
            "seen": self.seen[index],
            "unknown": self.unknown[index],
            "impute_mask": self.impute_mask[index],
            "future_valid": self.future_valid[index],
            "key": self.keys[index],
            "index": torch.tensor(index, dtype=torch.long),
        }


def _raw_full_history(
    repo_root: Path,
    dataset: str,
    split: str,
    expected_future: torch.Tensor,
    expected_observation: torch.Tensor,
    cache_path: Path | None = None,
) -> torch.Tensor:
    expected_quantized = (
        torch.round(expected_future.detach().cpu().float() * 10_000)
        .to(torch.int64)
        .numpy()
    )
    expected_signature = hashlib.sha256(expected_quantized.tobytes()).hexdigest()
    if cache_path is not None and cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if (
            cached.get("future_signature") == expected_signature
            and tuple(cached.get("shape", ())) == tuple(expected_observation.shape)
        ):
            return cached["full_history"].float()

    # Imported lazily so the goal module remains isolated from the original model.
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from dataloader_ours import TrajectoryDataset_ori

    raw = TrajectoryDataset_ori(str(repo_root / "data" / "datasets" / dataset.lower() / split))
    raw_history = raw.obs_traj.float()
    raw_future = raw.pred_traj.float()

    # The original TrajImpute generator iterated raw split files with
    # os.listdir(). Moving the project between filesystems can change that
    # directory order even though every trajectory is still present. Align by
    # trajectory content rather than relying on unstable filesystem order.
    def key(points: torch.Tensor) -> bytes:
        quantized = torch.round(points.detach().cpu().float() * 10_000).to(torch.int64)
        return quantized.numpy().tobytes()

    raw_groups: dict[bytes, list[int]] = defaultdict(list)
    expected_groups: dict[bytes, list[int]] = defaultdict(list)
    for index, future in enumerate(raw_future):
        raw_groups[key(future)].append(index)
    for index, future in enumerate(expected_future):
        expected_groups[key(future)].append(index)

    missing_keys = set(expected_groups) - set(raw_groups)
    if missing_keys:
        missing_rows = sum(len(expected_groups[item]) for item in missing_keys)
        raise ValueError(
            f"{missing_rows} missing-data rows have no matching raw future"
        )

    aligned = torch.empty_like(expected_observation, dtype=torch.float32)
    observation = expected_observation.float()
    for future_key, expected_rows in expected_groups.items():
        raw_rows = raw_groups[future_key]
        if len(expected_rows) % len(raw_rows):
            raise ValueError(
                "missing/raw trajectory multiplicities are not integer-aligned: "
                f"{len(expected_rows)} versus {len(raw_rows)}"
            )
        replicas = len(expected_rows) // len(raw_rows)
        expanded_raw_rows = raw_rows * replicas
        candidates = raw_history[expanded_raw_rows]
        costs = torch.empty(
            len(expected_rows),
            len(expanded_raw_rows),
            dtype=torch.float32,
        )
        # Rare trajectories can share an identical future. Solve a small
        # one-to-one assignment using their visible histories; test masks have
        # five replicas, so each raw trajectory receives exactly five slots.
        for matrix_row, expected_row in enumerate(expected_rows):
            seen = torch.isfinite(observation[expected_row]).all(dim=-1)
            if not seen.any():
                raise ValueError("trajectory has no visible historical coordinate")
            costs[matrix_row] = (
                candidates[:, seen] - observation[expected_row, seen][None]
            ).abs().amax(dim=(1, 2))
        assigned_rows, assigned_columns = linear_sum_assignment(costs.numpy())
        assigned_costs = costs[assigned_rows, assigned_columns]
        maximum = float(assigned_costs.max())
        if not torch.isfinite(assigned_costs).all() or maximum > 1e-4:
            raise ValueError(
                "raw/missing visible-history assignment failed: "
                f"maximum assigned difference {maximum}"
            )
        for matrix_row, matrix_column in zip(
            assigned_rows.tolist(), assigned_columns.tolist()
        ):
            aligned[expected_rows[matrix_row]] = candidates[matrix_column]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "future_signature": expected_signature,
                "shape": tuple(expected_observation.shape),
                "full_history": aligned,
                "alignment": "future-content groups plus visible-history assignment",
            },
            cache_path,
        )
    return aligned


def load_bundle(
    repo_root: Path,
    dataset: str,
    difficulty: str,
    **kwargs,
) -> JointDataBundle:
    if dataset.upper() in {"AV2", "ARGOVERSE2", "ARGOVERSE_2"}:
        from src.av2_data import load_av2_bundle

        return load_av2_bundle(repo_root=repo_root, **kwargs)

    root = (
        repo_root
        / "data"
        / "processed"
        / "TrajImpute"
        / "pkl_type"
        / f"{dataset.upper()}-M"
        / difficulty
    )
    loaded: Dict[str, Tuple[Dict, torch.Tensor]] = {}
    for split in ("train", "val", "test"):
        missing = _load_missing_file(root / f"data_{split}.pkl")
        future = torch.as_tensor(missing["pred_traj"], dtype=torch.float32)
        observation = torch.as_tensor(missing["obs_traj"], dtype=torch.float32)
        history = _raw_full_history(
            repo_root,
            dataset,
            split,
            future,
            observation,
            cache_path=(
                repo_root
                / "data"
                / "processed"
                / "TrajImpute"
                / "aligned_history_cache"
                / f"{dataset.upper()}_{difficulty}_{split}.pt"
            ),
        )
        loaded[split] = (missing, history)

    train_missing, train_history = loaded["train"]
    train_obs = torch.as_tensor(train_missing["obs_traj"], dtype=torch.float32)
    train_seen = ~torch.as_tensor(train_missing["missing_mask"]).bool()[..., 0]
    centers, rotations = _relative_rotation(train_obs, train_seen)
    train_full = torch.cat(
        [train_history, torch.as_tensor(train_missing["pred_traj"], dtype=torch.float32)],
        dim=1,
    )
    train_canonical = _canonicalize(train_full, centers, rotations)
    coordinate_scale = float(train_canonical.std().clamp_min(1e-3).item())

    train = JointTrajectoryDataset(*loaded["train"], coordinate_scale)
    val = JointTrajectoryDataset(*loaded["val"], coordinate_scale)
    test = JointTrajectoryDataset(*loaded["test"], coordinate_scale)
    return JointDataBundle(train=train, val=val, test=test, coordinate_scale=coordinate_scale)
