from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time as time_module
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import JointDataBundle, load_bundle, set_retrieval_key_variant
from models.memoflow import (
    JointCVAE,
    JointFlowModel,
    RetrievalResult,
    TrajectoryMemory,
    VariationalAnchorFlow,
    adapt_retrieved_anchors,
    clamp_observed,
    masked_mse,
    source_state,
)

OBJECT_TYPE_NAMES = {
    0: "vehicle",
    1: "pedestrian",
    2: "motorcyclist",
    3: "cyclist",
    4: "bus",
    5: "static",
    6: "background",
    7: "construction",
    8: "riderless_bicycle",
    9: "unknown",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def attach_scene_retrieval_keys(bundle: JointDataBundle, args) -> None:
    """Fuse audited HD-map descriptors into the retrieval key.

    The trajectory model and loss remain unchanged; only the memory query
    geometry is scene-conditioned.  The same train-fitted scene descriptor
    definition is used for train/validation/test, and the cache is aligned by
    scenario ID rather than filesystem order.
    """
    if not getattr(args, "scene_aware_retrieval", False):
        return
    cache_root = Path(args.scene_cache_root)
    weight = float(args.scene_key_weight)
    for split in ("train", "val", "test"):
        dataset = getattr(bundle, split)
        cache = torch.load(cache_root / f"{split}.pt", map_location="cpu", weights_only=False)
        mapping = {str(sid): i for i, sid in enumerate(cache["scenario_ids"])}
        missing = [sid for sid in dataset.scenario_ids if str(sid) not in mapping]
        if missing:
            raise RuntimeError(f"scene cache missing {len(missing)} {split} scenario IDs")
        order = torch.tensor([mapping[str(sid)] for sid in dataset.scenario_ids], dtype=torch.long)
        scene = cache["scene_features"][order].float()
        if dataset.keys.shape[-1] != scene.shape[-1]:
            padded = torch.zeros(scene.shape[0], dataset.keys.shape[-1], dtype=scene.dtype)
            padded[:, : min(scene.shape[-1], padded.shape[-1])] = scene[:, : padded.shape[-1]]
            scene = padded
        dataset.keys = F.normalize(dataset.keys.float() + weight * scene, dim=-1)
        dataset.scene_cache_name = str(cache_root / f"{split}.pt")
    args.scene_cache_audit = str(cache_root)


def make_memory(
    bundle: JointDataBundle,
    args,
    device: torch.device,
    train_indices: Optional[torch.Tensor] = None,
) -> Optional[TrajectoryMemory]:
    if not args.use_memory:
        return None
    keys = bundle.train.keys if train_indices is None else bundle.train.keys[train_indices]
    values = bundle.train.target if train_indices is None else bundle.train.target[train_indices]
    return TrajectoryMemory(
        keys=keys,
        values=values,
        top_k=args.memory_k,
        temperature=args.memory_temperature,
        source_indices=(
            torch.arange(len(bundle.train), dtype=torch.long)
            if train_indices is None
            else train_indices
        ),
    ).to(device)


def memory_for_batch(
    memory: Optional[TrajectoryMemory],
    batch: Dict[str, torch.Tensor],
    use_memory: bool,
    training: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    target = batch["target"]
    if not use_memory:
        return torch.zeros_like(target), None
    if memory is None:
        raise RuntimeError("memory-enabled run has no train memory")
    query_indices = batch["index"] if training else None
    result = memory.retrieve(batch["key"], query_indices=query_indices)
    anchors = memory.sample_anchors(result, samples=1).squeeze(1)
    return result.context, anchors


def refresh_retrieval_index(model, memory, args) -> None:
    if memory is None:
        return
    if getattr(args, "learned_retrieval_keys", False):
        if not hasattr(model, "retrieval_key_encoder"):
            raise RuntimeError("learned retrieval keys require variational_flow")
        model.retrieval_key_encoder.eval()
        memory.set_key_encoder(model.retrieval_key_encoder)
    else:
        memory.set_key_encoder(None)


def adapt_retrieval(
    retrieval: RetrievalResult,
    batch: Dict[str, torch.Tensor],
    args,
) -> RetrievalResult:
    adapted = adapt_retrieved_anchors(
        retrieval.values,
        batch["partial"],
        batch["seen"],
        smooth_weight=args.anchor_smooth_weight,
        extrapolation_weight=args.anchor_extrapolation_weight,
    )
    return RetrievalResult(
        values=adapted,
        weights=retrieval.weights,
        indices=retrieval.indices,
        context=torch.zeros_like(retrieval.context),
    )


def rerank_retrieval(
    model: VariationalAnchorFlow,
    retrieval: RetrievalResult,
    batch: Dict[str, torch.Tensor],
    args,
) -> RetrievalResult:
    """Rerank memory candidates using train-supervised predicted future risk."""
    if (
        args.reranker_strength <= 0
        and args.selector_strength <= 0
        and args.refinement_reranker_strength <= 0
    ):
        return retrieval
    combined = torch.log(retrieval.weights.clamp_min(1e-8))
    if args.reranker_strength > 0:
        predicted_risk = model.score_references(
            batch["partial"], batch["seen"], retrieval.values
        )
        temperature = max(float(args.reranker_temperature), 1e-6)
        combined = (
            combined - args.reranker_strength * predicted_risk / temperature
        )
    if args.selector_strength > 0:
        selector_logits = reference_selector_logits(
            model, batch, retrieval.values, args
        )
        combined = combined + args.selector_strength * torch.log_softmax(
            selector_logits / max(args.selector_temperature, 1e-6), dim=-1
        )
    if args.refinement_reranker_strength > 0:
        batch_size, candidates = retrieval.values.shape[:2]
        partial = batch["partial"][:, None].expand(
            -1, candidates, -1, -1
        ).reshape(batch_size * candidates, 8, 2)
        seen = batch["seen"][:, None].expand(
            -1, candidates, -1
        ).reshape(batch_size * candidates, 8)
        reference = retrieval.values.reshape(batch_size * candidates, 20, 2)
        refined = model.predict_mean_trajectory(
            partial, seen, reference
        ).reshape(batch_size, candidates, 20, 2)
        correction = torch.linalg.vector_norm(
            refined[:, :, 8:] - retrieval.values[:, :, 8:], dim=-1
        )
        correction_risk = (
            correction.mean(dim=-1)
            + args.refinement_reranker_fde_weight * correction[:, :, -1]
        )
        combined = (
            combined
            - args.refinement_reranker_strength * correction_risk
        )
    order = combined.argsort(dim=-1, descending=True)
    rows = torch.arange(order.shape[0], device=order.device)[:, None]
    values = retrieval.values[rows, order]
    indices = retrieval.indices[rows, order]
    weights = torch.softmax(combined[rows, order], dim=-1)
    context = (weights[..., None, None] * values).sum(dim=1)
    return RetrievalResult(
        values=values,
        weights=weights,
        indices=indices,
        context=context,
    )


def reference_selector_logits(model, batch, references, args):
    if getattr(args, "set_aware_selector", False):
        return model.select_set_reference_logits(
            batch["partial"], batch["seen"], references,
            object_type=batch.get("object_type"),
        )
    return model.select_reference_logits(
        batch["partial"], batch["seen"], references
    )


def select_conditional_mode_subset(model, batch, center, candidates, args):
    """Select coverage branches from a larger pool without target access.

    The selector sees only the incomplete query and predicted trajectories.
    Training targets supervise its ordering, but this helper is also the exact
    target-free path used by validation and test inference.
    """
    selected = int(args.conditional_selected_modes)
    if selected <= 0:
        return candidates
    pool = candidates[:, 1:]
    if selected > pool.shape[1]:
        raise ValueError("conditional selected modes exceed candidate pool")
    logits = model.select_set_reference_logits(
        batch["partial"], batch["seen"], pool,
        object_type=batch.get("object_type"),
    )
    index = logits.topk(selected, dim=1, largest=True, sorted=True).indices
    rows = torch.arange(pool.shape[0], device=pool.device)[:, None]
    chosen = pool[rows, index]
    center_copies = args.num_samples - selected
    return torch.cat([
        center[:, None].expand(-1, center_copies, -1, -1), chosen
    ], dim=1)


def mantra_proposals(
    model: VariationalAnchorFlow,
    memory: TrajectoryMemory,
    batch: Dict[str, torch.Tensor],
    samples: int,
    exclude_query: bool = False,
    args=None,
) -> torch.Tensor:
    """Generate motion-aware MANTRA proposals on the matched fold memory.

    Position matching remains the default for checkpoint compatibility.  The
    optional motion terms use only visible history and therefore cannot leak
    missing history or future targets.  A larger relevance pool can also be
    reduced with endpoint farthest-point sampling, retaining the nearest
    proposal while adding a validation-controlled number of distinct modes.
    """
    mask = batch["seen"][:, :, None].to(batch["partial"].dtype)
    position_error = (
        memory.values[None, :, :8] - batch["partial"][:, None]
    ).square().sum(dim=-1)
    recency_power = float(getattr(args, "mantra_recency_power", 0.0))
    recency = torch.linspace(
        1.0 / 8.0, 1.0, 8,
        device=position_error.device,
        dtype=position_error.dtype,
    ).pow(recency_power)
    position_mask = mask[..., 0] * recency[None]
    distance = (
        position_error * position_mask[:, None]
    ).sum(dim=-1) / position_mask.sum(dim=-1)[:, None].clamp_min(1e-6)

    velocity_weight = float(getattr(args, "mantra_velocity_weight", 0.0))
    if velocity_weight:
        query_velocity = batch["partial"][:, 1:] - batch["partial"][:, :-1]
        memory_velocity = memory.values[:, 1:8] - memory.values[:, :7]
        velocity_mask = (
            batch["seen"][:, 1:] & batch["seen"][:, :-1]
        ).to(position_error.dtype)
        velocity_error = (
            memory_velocity[None] - query_velocity[:, None]
        ).square().sum(dim=-1)
        velocity_distance = (
            velocity_error * velocity_mask[:, None]
        ).sum(dim=-1) / velocity_mask.sum(dim=-1)[:, None].clamp_min(1.0)
        has_velocity = velocity_mask.sum(dim=-1, keepdim=True) > 0
        distance = distance + velocity_weight * torch.where(
            has_velocity, velocity_distance, torch.zeros_like(velocity_distance)
        )

    acceleration_weight = float(
        getattr(args, "mantra_acceleration_weight", 0.0)
    )
    if acceleration_weight:
        query_velocity = batch["partial"][:, 1:] - batch["partial"][:, :-1]
        query_acceleration = query_velocity[:, 1:] - query_velocity[:, :-1]
        memory_velocity = memory.values[:, 1:8] - memory.values[:, :7]
        memory_acceleration = memory_velocity[:, 1:] - memory_velocity[:, :-1]
        acceleration_mask = (
            batch["seen"][:, 2:]
            & batch["seen"][:, 1:-1]
            & batch["seen"][:, :-2]
        ).to(position_error.dtype)
        acceleration_error = (
            memory_acceleration[None] - query_acceleration[:, None]
        ).square().sum(dim=-1)
        acceleration_distance = (
            acceleration_error * acceleration_mask[:, None]
        ).sum(dim=-1) / acceleration_mask.sum(dim=-1)[:, None].clamp_min(1.0)
        has_acceleration = acceleration_mask.sum(dim=-1, keepdim=True) > 0
        distance = distance + acceleration_weight * torch.where(
            has_acceleration,
            acceleration_distance,
            torch.zeros_like(acceleration_distance),
        )
    if exclude_query:
        own_row = batch["index"][:, None] == memory.source_indices[None, :]
        distance = distance.masked_fill(own_row, torch.inf)
    pool_multiplier = max(
        1, int(getattr(args, "mantra_candidate_pool_multiplier", 1))
    )
    pool_count = min(samples * pool_multiplier, memory.values.shape[0])
    pool_indices = distance.topk(pool_count, largest=False).indices
    diversity_samples = min(
        samples, max(0, int(getattr(args, "mantra_diversity_samples", 0)))
    )
    if diversity_samples > 1 and pool_count > samples:
        rows = torch.arange(distance.shape[0], device=distance.device)
        pool_values = memory.values[pool_indices]
        endpoints = pool_values[:, :, -1]
        chosen = [torch.zeros(distance.shape[0], dtype=torch.long, device=distance.device)]
        minimum_distance = torch.linalg.vector_norm(
            endpoints - endpoints[:, :1], dim=-1
        )
        relevance_strength = float(
            getattr(args, "mantra_diversity_relevance", 0.25)
        )
        rank_relevance = torch.linspace(
            1.0, 0.0, pool_count,
            device=distance.device,
            dtype=distance.dtype,
        )[None].expand(distance.shape[0], -1)
        for _ in range(1, diversity_samples):
            score = minimum_distance + relevance_strength * rank_relevance
            for previous in chosen:
                score[rows, previous] = -torch.inf
            next_column = score.argmax(dim=-1)
            chosen.append(next_column)
            next_endpoint = endpoints[rows, next_column]
            minimum_distance = torch.minimum(
                minimum_distance,
                torch.linalg.vector_norm(
                    endpoints - next_endpoint[:, None], dim=-1
                ),
            )
        for column in range(pool_count):
            if len(chosen) >= samples:
                break
            candidate = torch.full_like(chosen[0], column)
            already_used = torch.zeros_like(candidate, dtype=torch.bool)
            for previous in chosen:
                already_used |= previous == candidate
            if already_used.all():
                continue
            # Per-row fallback to the first unused relevance-ranked element.
            if already_used.any():
                for alternative in range(pool_count):
                    alt = torch.full_like(candidate, alternative)
                    used = torch.zeros_like(already_used)
                    for previous in chosen:
                        used |= previous == alt
                    candidate = torch.where(already_used & ~used, alt, candidate)
                    already_used = torch.zeros_like(already_used)
                    for previous in chosen:
                        already_used |= previous == candidate
                    if not already_used.any():
                        break
            chosen.append(candidate)
        selected_columns = torch.stack(chosen[:samples], dim=-1)
        indices = pool_indices[rows[:, None], selected_columns]
    else:
        indices = pool_indices[:, :samples]
    count = indices.shape[1]
    proposals = memory.values[indices]
    if count < samples:
        repeats = (samples + count - 1) // count
        proposals = proposals.repeat(1, repeats, 1, 1)[:, :samples]
    feature = torch.cat(
        [batch["partial"], batch["seen"][:, :, None].to(batch["partial"].dtype)],
        dim=-1,
    ).reshape(batch["partial"].shape[0], -1)
    residual = model.mantra_refine(feature).reshape(-1, 1, 20, 2)
    proposals = proposals + 0.15 * residual
    target = batch["target"][:, None].expand(-1, samples, -1, -1).reshape(
        batch["target"].shape[0] * samples, 20, 2
    )
    seen = batch["seen"][:, None].expand(-1, samples, -1).reshape(
        batch["seen"].shape[0] * samples, 8
    )
    return clamp_observed(
        proposals.reshape(-1, 20, 2), target, seen
    ).reshape(batch["target"].shape[0], samples, 20, 2)


def sample_training_reference(
    retrieval: RetrievalResult,
    batch: Dict[str, torch.Tensor],
    args,
) -> torch.Tensor:
    """Sample from a target-aware/uniform mixture for robust anchor transport."""
    future_error = torch.linalg.vector_norm(
        retrieval.values[:, :, 8:] - batch["target"][:, None, 8:], dim=-1
    )
    future_cost = (
        (1.0 - args.ot_fde_weight) * future_error.mean(dim=-1)
        + args.ot_fde_weight * future_error[:, :, -1]
    )
    ot_weights = torch.softmax(-future_cost / args.ot_temperature, dim=-1)
    uniform = torch.full_like(ot_weights, 1.0 / ot_weights.shape[1])
    exposure = float(args.uniform_anchor_exposure)
    sampling_weights = (1.0 - exposure) * ot_weights + exposure * uniform
    chosen = torch.multinomial(sampling_weights, num_samples=1)
    rows = torch.arange(batch["target"].shape[0], device=batch["target"].device)[:, None]
    return retrieval.values[rows, chosen].squeeze(1)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


def _optimizer_step(
    model,
    optimizer,
    scaler,
    args,
) -> None:
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_retrieval_key_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Learn retrieval geometry from train futures without inference leakage."""
    model.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train retrieval keys", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        if batch["key"].shape[0] < 2:
            continue
        with _autocast(device, args.amp):
            embedding = model.encode_retrieval_key(batch["key"])
            similarity = embedding @ embedding.T
            diagonal = torch.eye(
                embedding.shape[0], device=device, dtype=torch.bool
            )
            prediction_logits = (
                similarity / args.retrieval_key_temperature
            ).masked_fill(diagonal, -torch.inf)

            future = batch["target"][:, 8:]
            pairwise = torch.linalg.vector_norm(
                future[:, None] - future[None, :], dim=-1
            )
            target_cost = pairwise.mean(dim=-1) + (
                args.retrieval_key_fde_weight * pairwise[:, :, -1]
            )
            target_cost = target_cost.masked_fill(diagonal, torch.inf)
            positive_count = min(
                args.retrieval_key_positive_k, embedding.shape[0] - 1
            )
            positive_cost, positive_index = torch.topk(
                target_cost, k=positive_count, largest=False, dim=-1
            )
            positive_weight = torch.softmax(
                -positive_cost / args.retrieval_target_temperature, dim=-1
            ).to(similarity.dtype)
            positive_log_probability = torch.log_softmax(
                prediction_logits, dim=-1
            ).gather(1, positive_index)
            row_loss = -(
                positive_weight * positive_log_probability
            ).sum(dim=-1)
            suffix4 = batch["seen"][:, :4].all(dim=1) & (
                ~batch["seen"][:, 4:]
            ).all(dim=1)
            query_weight = torch.where(
                suffix4,
                torch.full_like(row_loss, args.retrieval_key_suffix_weight),
                torch.ones_like(row_loss),
            )
            loss = (row_loss * query_weight).sum() / query_weight.sum().clamp_min(1)
        scaler.scale(loss / args.grad_accum_steps).backward()
        if (
            batch_index % args.grad_accum_steps == 0
            or batch_index == len(loader)
        ):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def train_flow_epoch(model, memory, loader, optimizer, scaler, args, device) -> float:
    model.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train flow", leave=False),
        start=1,
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            if args.use_memory:
                if memory is None:
                    raise RuntimeError("memory-enabled run has no train memory")
                retrieval = memory.retrieve(
                    batch["key"], query_indices=batch["index"]
                )
                context = retrieval.context
                future_error = torch.linalg.vector_norm(
                    retrieval.values[:, :, 8:] - batch["target"][:, None, 8:],
                    dim=-1,
                )
                future_cost = (
                    (1.0 - args.ot_fde_weight) * future_error.mean(dim=-1)
                    + args.ot_fde_weight * future_error[:, :, -1]
                )
                coupling_weights = torch.softmax(
                    -future_cost / args.ot_temperature,
                    dim=-1,
                )
                chosen = torch.multinomial(coupling_weights, num_samples=1)
                rows = torch.arange(
                    batch["target"].shape[0], device=device
                )[:, None]
                anchors = retrieval.values[rows, chosen].squeeze(1)
                memory_source = source_state(
                    batch["target"], batch["seen"], anchors, args.source_noise
                )
                gaussian_source = source_state(
                    batch["target"], batch["seen"], None, args.source_noise
                )
                use_anchor = (
                    torch.rand(batch["target"].shape[0], 1, 1, device=device)
                    < args.memory_source_ratio
                )
                source = torch.where(use_anchor, memory_source, gaussian_source)
            else:
                context = torch.zeros_like(batch["target"])
                source = source_state(
                    batch["target"], batch["seen"], None, args.source_noise
                )
            time = torch.rand(batch["target"].shape[0], device=device)
            state = (
                (1 - time[:, None, None]) * source
                + time[:, None, None] * batch["target"]
            )
            state = clamp_observed(state, batch["target"], batch["seen"])
            target_velocity = batch["target"] - source
            prediction = model(
                state, time, batch["partial"], batch["seen"], context
            )
            step_weights = torch.ones(20, device=device)
            step_weights[-1] = args.endpoint_loss_weight
            loss = masked_mse(
                prediction,
                target_velocity,
                batch["unknown"],
                step_weights=step_weights,
            )
        scaler.scale(loss / args.grad_accum_steps).backward()
        if (
            batch_index % args.grad_accum_steps == 0
            or batch_index == len(loader)
        ):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
        if args.batch_sleep_ms > 0:
            time_module.sleep(args.batch_sleep_ms / 1000.0)
    return total / max(count, 1)


def train_anchor_flow_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    if memory is None:
        raise RuntimeError("anchor flow requires memory")
    model.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train anchor flow", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            retrieval = memory.retrieve(
                batch["key"], query_indices=batch["index"]
            )
            retrieval = adapt_retrieval(retrieval, batch, args)
            reference = sample_training_reference(retrieval, batch, args)
            source = source_state(
                batch["target"], batch["seen"], reference, args.source_noise
            )
            time = torch.rand(batch["target"].shape[0], device=device)
            state = (
                (1 - time[:, None, None]) * source
                + time[:, None, None] * batch["target"]
            )
            state = clamp_observed(state, batch["target"], batch["seen"])
            prediction = model(
                state,
                time,
                batch["partial"],
                batch["seen"],
                reference,
            )
            step_weights = torch.ones(20, device=device)
            step_weights[-1] = args.endpoint_loss_weight
            loss = masked_mse(
                prediction,
                batch["target"] - source,
                batch["unknown"],
                step_weights=step_weights,
            )
        scaler.scale(loss / args.grad_accum_steps).backward()
        if (
            batch_index % args.grad_accum_steps == 0
            or batch_index == len(loader)
        ):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
        if args.batch_sleep_ms > 0:
            time_module.sleep(args.batch_sleep_ms / 1000.0)
    return total / max(count, 1)


def train_variational_flow_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    if memory is None:
        raise RuntimeError("variational flow requires memory")
    model.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train variational flow", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            retrieval = memory.retrieve(
                batch["key"], query_indices=batch["index"]
            )
            retrieval = adapt_retrieval(retrieval, batch, args)
            mean_reference = (
                retrieval.weights[..., None, None] * retrieval.values
            ).sum(dim=1)
            mean_prediction = model.predict_mean_trajectory(
                batch["partial"], batch["seen"], mean_reference
            )
            mean_prediction = clamp_observed(
                mean_prediction, batch["target"], batch["seen"]
            )
            mean_error = torch.linalg.vector_norm(
                mean_prediction - batch["target"], dim=-1
            )
            mean_weights = torch.ones_like(mean_error)
            mean_weights[:, :8] = args.mean_head_history_weight
            mean_weights[:, -1] = args.mean_head_endpoint_weight
            mean_mask = batch["unknown"].to(mean_error.dtype)
            mean_head_loss = (
                mean_error * mean_weights * mean_mask
            ).sum() / (mean_weights * mean_mask).sum().clamp_min(1)
            reranker_loss = torch.zeros((), device=device)
            if args.reranker_loss_weight > 0:
                predicted_risk = model.score_references(
                    batch["partial"], batch["seen"], retrieval.values
                )
                candidate_error = torch.linalg.vector_norm(
                    retrieval.values[:, :, 8:]
                    - batch["target"][:, None, 8:],
                    dim=-1,
                )
                target_risk = (
                    candidate_error.mean(dim=-1)
                    + args.reranker_fde_weight * candidate_error[:, :, -1]
                )
                reranker_loss = F.smooth_l1_loss(
                    predicted_risk, target_risk
                )
            reference = sample_training_reference(retrieval, batch, args)
            (
                residual_q,
                latent_q,
                residual_p,
                latent_p,
                residual_mean,
                latent_mean,
                kl,
            ) = (
                model.training_residual_pair(
                batch["partial"],
                batch["seen"],
                reference,
                batch["target"],
                )
            )
            if args.residual_scope == "history":
                scope = torch.zeros_like(residual_q)
                scope[:, :8] = 1
                residual_q = residual_q * scope
                residual_p = residual_p * scope
                residual_mean = residual_mean * scope
            elif args.residual_scope == "future":
                scope = torch.zeros_like(residual_q)
                scope[:, 8:] = 1
                residual_q = residual_q * scope
                residual_p = residual_p * scope
                residual_mean = residual_mean * scope
            if args.flow_latent_source == "prior":
                residual, latent = residual_p, latent_p
            elif args.flow_latent_source == "mean":
                residual, latent = residual_mean, latent_mean
            elif args.flow_latent_source == "posterior":
                residual, latent = residual_q, latent_q
            else:
                use_prior = (
                    torch.rand(batch["target"].shape[0], 1, 1, device=device)
                    < args.prior_flow_probability
                )
                residual = torch.where(use_prior, residual_p, residual_q)
                latent = torch.where(
                    use_prior[:, :, 0], latent_p, latent_q
                )
            source = source_state(
                batch["target"],
                batch["seen"],
                reference + args.residual_scale * residual,
                args.source_noise,
            )
            time = torch.rand(batch["target"].shape[0], device=device)
            state = (
                (1 - time[:, None, None]) * source
                + time[:, None, None] * batch["target"]
            )
            state = clamp_observed(state, batch["target"], batch["seen"])
            target_velocity = batch["target"] - source
            prediction = model(
                state,
                time,
                batch["partial"],
                batch["seen"],
                reference,
                latent,
            )
            step_weights = torch.ones(20, device=device)
            step_weights[-1] = args.endpoint_loss_weight
            flow_loss = masked_mse(
                prediction,
                target_velocity,
                batch["unknown"],
                step_weights=step_weights,
            )
            residual_mask = batch["unknown"].clone()
            if args.residual_scope == "history":
                residual_mask[:, 8:] = False
            elif args.residual_scope == "future":
                residual_mask[:, :8] = False
            residual_loss = masked_mse(
                residual_q,
                batch["target"] - reference,
                residual_mask,
            )
            prior_residual_loss = masked_mse(
                residual_mean,
                batch["target"] - reference,
                residual_mask,
            )
            rollout_loss = torch.zeros((), device=device, dtype=flow_loss.dtype)
            if args.rollout_loss_weight > 0:
                rollout = source
                rollout_step = (
                    args.training_flow_strength / args.training_rollout_steps
                )
                for rollout_index in range(args.training_rollout_steps):
                    rollout_time = torch.full(
                        (rollout.shape[0],),
                        rollout_index / args.training_rollout_steps,
                        device=device,
                        dtype=rollout.dtype,
                    )
                    rollout = rollout + rollout_step * model(
                        rollout,
                        rollout_time,
                        batch["partial"],
                        batch["seen"],
                        reference,
                        latent,
                    )
                    rollout = clamp_observed(
                        rollout, batch["target"], batch["seen"]
                    )
                rollout_loss = masked_mse(
                    rollout,
                    batch["target"],
                    batch["unknown"],
                )
            risk_rollout_loss = torch.zeros(
                (), device=device, dtype=flow_loss.dtype
            )
            if args.risk_rollout_loss_weight > 0:
                # The risk branch represents the conditional centre of the
                # retrieved set.  It is trained with the prior mean because
                # this branch should minimize expected displacement instead
                # of adding another rare mode.
                risk_reference = (
                    retrieval.weights[..., None, None] * retrieval.values
                ).sum(dim=1)
                (
                    _,
                    _,
                    _,
                    _,
                    risk_residual,
                    risk_latent,
                    _,
                ) = model.training_residual_pair(
                    batch["partial"],
                    batch["seen"],
                    risk_reference,
                    batch["target"],
                )
                if args.residual_scope == "history":
                    risk_residual[:, 8:] = 0
                elif args.residual_scope == "future":
                    risk_residual[:, :8] = 0
                risk_source = source_state(
                    batch["target"],
                    batch["seen"],
                    risk_reference + args.residual_scale * risk_residual,
                    args.source_noise,
                )
                risk_rollout = risk_source
                risk_step = (
                    args.training_flow_strength / args.training_rollout_steps
                )
                for rollout_index in range(args.training_rollout_steps):
                    rollout_time = torch.full(
                        (risk_rollout.shape[0],),
                        rollout_index / args.training_rollout_steps,
                        device=device,
                        dtype=risk_rollout.dtype,
                    )
                    risk_rollout = risk_rollout + risk_step * model(
                        risk_rollout,
                        rollout_time,
                        batch["partial"],
                        batch["seen"],
                        risk_reference,
                        risk_latent,
                    )
                    risk_rollout = clamp_observed(
                        risk_rollout, batch["target"], batch["seen"]
                    )
                risk_mask = batch["unknown"].clone()
                risk_mask[:, :8] = False
                risk_step_weights = torch.ones(20, device=device)
                risk_step_weights[-1] = args.risk_endpoint_loss_weight
                risk_rollout_loss = masked_mse(
                    risk_rollout,
                    batch["target"],
                    risk_mask,
                    step_weights=risk_step_weights,
                )
            loss = (
                flow_loss
                + args.residual_loss_weight * residual_loss
                + args.prior_residual_loss_weight * prior_residual_loss
                + args.kl_weight * kl.mean()
                + args.rollout_loss_weight * rollout_loss
                + args.risk_rollout_loss_weight * risk_rollout_loss
                + args.reranker_loss_weight * reranker_loss
                + args.mean_head_loss_weight * mean_head_loss
            )
        scaler.scale(loss / args.grad_accum_steps).backward()
        if (
            batch_index % args.grad_accum_steps == 0
            or batch_index == len(loader)
        ):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
        if args.batch_sleep_ms > 0:
            time_module.sleep(args.batch_sleep_ms / 1000.0)
    return total / max(count, 1)


def train_mean_head_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Fit only the conditional-mean branch while transport stays frozen."""
    if memory is None:
        raise RuntimeError("mean head requires memory")
    model.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train mean head", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            if args.external_center_candidate_training_source:
                if not hasattr(model, "external_center"):
                    raise RuntimeError(
                        "external candidate source requires loaded checkpoint"
                    )
                proposal_values = mantra_proposals(
                    model, memory, batch, args.num_samples,
                    exclude_query=True, args=args,
                )
                center_values = external_center_trajectories(
                    model.external_center, batch, args.external_center_kind,
                    args.num_samples, False,
                )
                center_count = center_values.shape[1]
                center_partial = batch["partial"][:, None].expand(
                    -1, center_count, -1, -1
                ).reshape(batch["partial"].shape[0] * center_count, 8, 2)
                center_seen = batch["seen"][:, None].expand(
                    -1, center_count, -1
                ).reshape(batch["seen"].shape[0] * center_count, 8)
                center_object_type = (
                    batch["object_type"][:, None].expand(
                        -1, center_count
                    ).reshape(-1)
                    if "object_type" in batch else None
                )
                refined_center = model.predict_motion_mean_trajectory(
                    center_partial, center_seen,
                    center_values.reshape(-1, 20, 2),
                    object_type=center_object_type,
                ).reshape_as(center_values)
                center_values = center_values + args.external_center_refiner_strength * (
                    refined_center - center_values
                )
                # Match inference: preserve repaired history and centre only
                # the future candidate set around the retrieved modes.
                center_values[:, :, :8] = proposal_values[:, :, :8]
                center_future = center_values[:, :, 8:]
                blend = float(args.external_center_proposal_blend)
                if blend:
                    center_future = center_future + blend * (
                        proposal_values[:, :, 8:] - center_future
                    )
                if args.external_center_endpoint_warp_power > 0:
                    endpoint_delta = (
                        proposal_values[:, :, -1] - center_future[:, :, -1]
                    )
                    ramp = torch.linspace(
                        1.0 / 12.0, 1.0, 12,
                        device=center_future.device, dtype=center_future.dtype,
                    ).pow(args.external_center_endpoint_warp_power)
                    center_future = center_future + (
                        ramp[None, None, :, None] * endpoint_delta[:, :, None]
                    )
                proposal_values = torch.cat(
                    [center_values[:, :, :8], center_future], dim=2
                )
                proposal_weights = torch.full(
                    proposal_values.shape[:2], 1.0 / proposal_values.shape[1],
                    device=proposal_values.device, dtype=proposal_values.dtype,
                )
                retrieval = RetrievalResult(
                    values=proposal_values, weights=proposal_weights,
                    indices=torch.full(
                        proposal_weights.shape, -1,
                        device=proposal_values.device, dtype=torch.long,
                    ),
                    context=proposal_values.mean(dim=1),
                )
            elif args.external_center_training_source:
                if not hasattr(model, "external_center"):
                    raise RuntimeError(
                        "external center training source requires loaded checkpoint"
                    )
                proposal_values = external_center_trajectories(
                    model.external_center,
                    batch,
                    args.external_center_kind,
                    1,
                    False,
                )
                proposal_weights = torch.ones(
                    proposal_values.shape[:2],
                    device=proposal_values.device,
                    dtype=proposal_values.dtype,
                )
                retrieval = RetrievalResult(
                    values=proposal_values,
                    weights=proposal_weights,
                    indices=torch.full(
                        proposal_weights.shape,
                        -1,
                        device=proposal_values.device,
                        dtype=torch.long,
                    ),
                    context=proposal_values[:, 0],
                )
            elif args.mantra_training_source:
                proposal_values = mantra_proposals(
                    model,
                    memory,
                    batch,
                    min(args.num_samples, memory.top_k),
                    exclude_query=True,
                    args=args,
                )
                candidates = proposal_values.shape[1]
                proposal_weights = torch.full(
                    (proposal_values.shape[0], candidates),
                    1.0 / candidates,
                    device=proposal_values.device,
                    dtype=proposal_values.dtype,
                )
                retrieval = RetrievalResult(
                    values=proposal_values,
                    weights=proposal_weights,
                    indices=torch.full(
                        proposal_weights.shape,
                        -1,
                        device=proposal_values.device,
                        dtype=torch.long,
                    ),
                    context=proposal_values.mean(dim=1),
                )
            else:
                retrieval = memory.retrieve(
                    batch["key"], query_indices=batch["index"]
                )
                retrieval = adapt_retrieval(retrieval, batch, args)
            proposal_retrieval = retrieval
            mean_values = (
                retrieval.context[:, None]
                if args.mean_head_center_only
                else retrieval.values
            )
            batch_size = mean_values.shape[0]
            candidates = (
                1
                if (
                    args.memory_set_mean_head
                    or args.set_aware_center_head
                    or args.database_center_head
                )
                else mean_values.shape[1]
            )
            mean_training_target = batch["target"]
            if (
                args.mean_head_endpoint_warp_training
                and not args.mean_head_direct_candidate_risk
            ):
                mean_training_target = batch["target"].clone()
                mean_endpoint = proposal_retrieval.values[:, :, -1].mean(dim=1)
                endpoint_bias = mean_endpoint - batch["target"][:, -1]
                compensation_ramp = torch.linspace(
                    1.0 / 12.0,
                    1.0,
                    12,
                    device=mean_training_target.device,
                    dtype=mean_training_target.dtype,
                ).pow(args.endpoint_warp_power)
                # Keep the endpoint target itself unchanged because endpoint
                # warping makes the generated final point candidate-specific.
                mean_training_target[:, 8:-1] = (
                    mean_training_target[:, 8:-1]
                    - compensation_ramp[:-1][None, :, None]
                    * endpoint_bias[:, None]
                )
            target = mean_training_target[:, None].expand(
                -1, candidates, -1, -1
            ).reshape(batch_size * candidates, 20, 2)
            if (
                args.memory_set_mean_head
                or args.set_aware_center_head
                or args.database_center_head
            ):
                partial = batch["partial"]
                seen = batch["seen"]
                center_prediction = (
                    model.predict_database_center(
                        partial,
                        seen,
                        proposal_retrieval.values,
                        object_type=batch.get("object_type"),
                    )
                    if args.database_center_head
                    else model.predict_set_aware_center(
                        partial,
                        seen,
                        proposal_retrieval.values,
                        object_type=batch.get("object_type"),
                    )
                    if args.set_aware_center_head
                    else model.predict_memory_set_center(
                        partial, seen, proposal_retrieval.values
                    )
                )
                if args.mean_head_direct_candidate_risk:
                    candidate_values = proposal_retrieval.values[
                        :, : args.num_samples
                    ]
                    candidates = candidate_values.shape[1]
                    prediction = center_prediction[:, None].expand(
                        -1, candidates, -1, -1
                    ).clone()
                    endpoint_delta = (
                        candidate_values[:, :, -1]
                        - center_prediction[:, None, -1]
                    )
                    primary_ramp = torch.linspace(
                        1.0 / 12.0,
                        1.0,
                        12,
                        device=prediction.device,
                        dtype=prediction.dtype,
                    ).pow(args.endpoint_warp_power)
                    prediction[:, :, 8:] = (
                        prediction[:, :, 8:]
                        + primary_ramp[None, None, :, None]
                        * endpoint_delta[:, :, None]
                    )
                    coverage_samples = min(
                        candidates, int(args.endpoint_warp_coverage_samples)
                    )
                    if coverage_samples:
                        coverage_ramp = torch.linspace(
                            1.0 / 12.0,
                            1.0,
                            12,
                            device=prediction.device,
                            dtype=prediction.dtype,
                        ).pow(args.endpoint_warp_coverage_power)
                        prediction[:, :coverage_samples, 8:] = (
                            center_prediction[:, None, 8:]
                            + coverage_ramp[None, None, :, None]
                            * endpoint_delta[:, :coverage_samples, None]
                        )
                    target = batch["target"][:, None].expand(
                        -1, candidates, -1, -1
                    ).reshape(batch_size * candidates, 20, 2)
                    partial = batch["partial"][:, None].expand(
                        -1, candidates, -1, -1
                    ).reshape(batch_size * candidates, 8, 2)
                    seen = batch["seen"][:, None].expand(
                        -1, candidates, -1
                    ).reshape(batch_size * candidates, 8)
                    prediction = prediction.reshape(
                        batch_size * candidates, 20, 2
                    )
                else:
                    prediction = center_prediction
            else:
                reference = mean_values.reshape(
                    batch_size * candidates, 20, 2
                )
                partial = batch["partial"][:, None].expand(
                    -1, candidates, -1, -1
                ).reshape(batch_size * candidates, 8, 2)
                seen = batch["seen"][:, None].expand(
                    -1, candidates, -1
                ).reshape(batch_size * candidates, 8)
                if args.motion_mean_head:
                    object_type = batch.get("object_type")
                    if object_type is not None:
                        object_type = object_type[:, None].expand(
                            -1, candidates
                        ).reshape(-1)
                    prediction = model.predict_motion_mean_trajectory(
                        partial, seen, reference, object_type=object_type
                    )
                else:
                    prediction = model.predict_mean_trajectory(
                        partial, seen, reference
                    )
            prediction = clamp_observed(
                prediction, target, seen
            )
            delta = prediction - target
            if args.mean_head_loss_type == "mse":
                error = delta.pow(2).sum(dim=-1)
            elif args.mean_head_loss_type == "smooth_l1":
                error = F.smooth_l1_loss(
                    prediction, target, reduction="none"
                ).sum(dim=-1)
            else:
                error = torch.linalg.vector_norm(delta, dim=-1)
            weights = torch.ones_like(error)
            weights[:, :8] = args.mean_head_history_weight
            weights[:, -1] = args.mean_head_endpoint_weight
            mask = batch["unknown"][:, None].expand(
                -1, candidates, -1
            ).reshape(batch_size * candidates, 20).to(error.dtype)
            per_example_mean_loss = (
                error * weights * mask
            ).sum(dim=-1) / (
                weights * mask
            ).sum(dim=-1).clamp_min(1)
            mean_target_loss = per_example_mean_loss.mean()
            if args.mean_head_velocity_weight > 0:
                prediction_velocity = prediction[:, 8:] - prediction[:, 7:-1]
                target_velocity = target[:, 8:] - target[:, 7:-1]
                velocity_loss = (
                    prediction_velocity - target_velocity
                ).pow(2).sum(dim=-1).mean()
                mean_target_loss = (
                    mean_target_loss
                    + args.mean_head_velocity_weight * velocity_loss
                )
            if args.mean_head_acceleration_weight > 0:
                prediction_velocity = prediction[:, 8:] - prediction[:, 7:-1]
                target_velocity = target[:, 8:] - target[:, 7:-1]
                prediction_acceleration = (
                    prediction_velocity[:, 1:] - prediction_velocity[:, :-1]
                )
                target_acceleration = (
                    target_velocity[:, 1:] - target_velocity[:, :-1]
                )
                acceleration_loss = (
                    prediction_acceleration - target_acceleration
                ).pow(2).sum(dim=-1).mean()
                mean_target_loss = (
                    mean_target_loss
                    + args.mean_head_acceleration_weight * acceleration_loss
                )
            if args.mean_head_direct_candidate_risk:
                scene_candidate_loss = per_example_mean_loss.reshape(
                    batch_size, candidates
                )
                mean_target_loss = (
                    mean_target_loss
                    + args.mean_head_oracle_weight
                    * scene_candidate_loss.amin(dim=1).mean()
                )
            if args.mean_head_cvar_weight > 0:
                cvar_source = (
                    scene_candidate_loss.mean(dim=1)
                    if args.mean_head_direct_candidate_risk
                    else per_example_mean_loss
                )
                cvar_count = max(
                    1,
                    int(round(
                        args.mean_head_cvar_fraction
                        * cvar_source.shape[0]
                    )),
                )
                cvar_loss = cvar_source.topk(
                    cvar_count, largest=True
                ).values.mean()
                mean_target_loss = (
                    mean_target_loss
                    + args.mean_head_cvar_weight * cvar_loss
                )
            loss = args.mean_head_loss_weight * mean_target_loss
            if args.mode_head_loss_weight > 0 or args.selector_loss_weight > 0:
                selector_candidates = proposal_retrieval.values
                if args.selector_refined_cost:
                    selector_batch, selector_count = selector_candidates.shape[:2]
                    selector_partial = batch["partial"][:, None].expand(
                        -1, selector_count, -1, -1
                    ).reshape(selector_batch * selector_count, 8, 2)
                    selector_seen = batch["seen"][:, None].expand(
                        -1, selector_count, -1
                    ).reshape(selector_batch * selector_count, 8)
                    selector_candidates = model.refine_candidate_trajectory(
                        selector_partial,
                        selector_seen,
                        selector_candidates.reshape(
                            selector_batch * selector_count, 20, 2
                        ),
                    ).reshape(selector_batch, selector_count, 20, 2)
                candidate_error = torch.linalg.vector_norm(
                    selector_candidates[:, :, 8:]
                    - batch["target"][:, None, 8:],
                    dim=-1,
                )
                candidate_cost = (
                    candidate_error.mean(dim=-1)
                    + args.mode_head_fde_weight * candidate_error[:, :, -1]
                )
            if args.mode_head_loss_weight > 0:
                mode_prediction = model.predict_mode_trajectory(
                    partial, seen, reference
                )
                mode_prediction = clamp_observed(
                    mode_prediction, target, seen
                )
                mode_error = torch.linalg.vector_norm(
                    mode_prediction - target, dim=-1
                ).reshape(batch_size, candidates, 20)
                oracle = torch.softmax(
                    -candidate_cost / max(args.mode_head_temperature, 1e-6),
                    dim=-1,
                )
                mode_weights = weights.reshape(batch_size, candidates, 20)
                mode_mask = mask.reshape(batch_size, candidates, 20)
                per_candidate = (
                    mode_error * mode_weights * mode_mask
                ).sum(dim=-1) / (
                    mode_weights * mode_mask
                ).sum(dim=-1).clamp_min(1)
                mode_target_loss = (oracle * per_candidate).sum(dim=-1).mean()
                preservation = torch.linalg.vector_norm(
                    mode_prediction - reference, dim=-1
                ).reshape(batch_size, candidates, 20)
                preservation = (
                    preservation * mode_weights * mode_mask
                ).sum() / (mode_weights * mode_mask).sum().clamp_min(1)
                loss = (
                    loss
                    + args.mode_head_loss_weight * mode_target_loss
                    + args.mode_head_preservation_weight * preservation
                )
            if args.selector_loss_weight > 0:
                selector_logits = reference_selector_logits(
                    model, batch, proposal_retrieval.values, args
                )
                selector_target = candidate_cost.argmin(dim=-1)
                selector_loss = F.cross_entropy(
                    selector_logits / max(args.selector_temperature, 1e-6),
                    selector_target,
                )
                loss = loss + args.selector_loss_weight * selector_loss
            if args.candidate_refiner_loss_weight > 0:
                candidate_reference = proposal_retrieval.values
                if args.set_aware_candidate_refiner:
                    refiner_prediction = model.refine_candidate_set(
                        batch["partial"], batch["seen"], candidate_reference,
                        object_type=batch.get("object_type"),
                    ).reshape(batch_size * candidates, 20, 2)
                else:
                    refiner_prediction = model.refine_candidate_trajectory(
                        partial,
                        seen,
                        candidate_reference.reshape(
                            batch_size * candidates, 20, 2
                        ),
                    )
                refiner_prediction = clamp_observed(
                    refiner_prediction, target, seen
                ).reshape(batch_size, candidates, 20, 2)
                refiner_error = torch.linalg.vector_norm(
                    refiner_prediction
                    - batch["target"][:, None],
                    dim=-1,
                )
                refiner_mask = batch["unknown"][:, None].expand(
                    -1, candidates, -1
                ).to(refiner_error.dtype)
                refiner_step_weights = torch.ones_like(refiner_error)
                refiner_step_weights[:, :, :8] = (
                    args.candidate_refiner_history_weight
                )
                refiner_step_weights[:, :, -1] = (
                    args.candidate_refiner_endpoint_weight
                )
                refiner_denominator = (
                    refiner_step_weights * refiner_mask
                ).sum().clamp_min(1)
                per_candidate_target = (
                    refiner_error * refiner_step_weights * refiner_mask
                ).sum(dim=-1) / (
                    refiner_step_weights * refiner_mask
                ).sum(dim=-1).clamp_min(1)
                original_future_error = torch.linalg.vector_norm(
                    candidate_reference[:, :, 8:]
                    - batch["target"][:, None, 8:],
                    dim=-1,
                ).mean(dim=-1)
                target_candidates = max(
                    1,
                    min(
                        candidates,
                        int(round(
                            args.candidate_refiner_target_fraction * candidates
                        )),
                    ),
                )
                target_indices = original_future_error.topk(
                    target_candidates, dim=-1, largest=False
                ).indices
                refiner_target_loss = per_candidate_target.gather(
                    1, target_indices
                ).mean()
                refiner_oracle_loss = per_candidate_target.min(dim=1).values.mean()
                refiner_scene_mean = per_candidate_target.mean(dim=1)
                refiner_cvar_count = max(
                    1,
                    int(round(
                        args.candidate_refiner_cvar_fraction
                        * refiner_scene_mean.shape[0]
                    )),
                )
                refiner_cvar_loss = refiner_scene_mean.topk(
                    refiner_cvar_count, largest=True
                ).values.mean()

                preservation_error = torch.linalg.vector_norm(
                    refiner_prediction - candidate_reference,
                    dim=-1,
                )
                preservation_loss = (
                    preservation_error
                    * refiner_step_weights
                    * refiner_mask
                ).sum() / refiner_denominator

                # Keep the retrieved hypotheses separated at intermediate
                # future waypoints. Endpoints are restored at inference, so
                # this term protects path diversity without weakening FDE.
                refined_path = refiner_prediction[:, :, 8:-1].reshape(
                    batch_size, candidates, -1
                )
                original_path = candidate_reference[:, :, 8:-1].reshape(
                    batch_size, candidates, -1
                )
                refined_distance = torch.cdist(refined_path, refined_path)
                original_distance = torch.cdist(original_path, original_path)
                off_diagonal = ~torch.eye(
                    candidates,
                    device=refined_distance.device,
                    dtype=torch.bool,
                )[None]
                diversity_loss = F.relu(
                    args.candidate_refiner_diversity_ratio
                    * original_distance.detach()
                    - refined_distance
                )[off_diagonal.expand_as(refined_distance)].mean()
                loss = (
                    loss
                    + args.candidate_refiner_loss_weight
                    * refiner_target_loss
                    + args.candidate_refiner_oracle_weight
                    * refiner_oracle_loss
                    + args.candidate_refiner_cvar_weight
                    * refiner_cvar_loss
                    + args.candidate_refiner_preservation_weight
                    * preservation_loss
                    + args.candidate_refiner_diversity_weight
                    * diversity_loss
                )
        scaler.scale(loss / args.grad_accum_steps).backward()
        if (
            batch_index % args.grad_accum_steps == 0
            or batch_index == len(loader)
        ):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def metric_target_hinge_loss(
    candidates: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    ade: torch.Tensor,
    fde: torch.Tensor,
    row_mean: torch.Tensor,
    args,
) -> torch.Tensor:
    """Differentiable penalties for only the still-violated table metrics."""
    scale = float(args.runtime_coordinate_scale)
    row_min_ade = ade.min(dim=1).values
    row_min_fde = fde.min(dim=1).values
    trajectory_error = torch.linalg.vector_norm(
        candidates - batch["target"][:, None], dim=-1
    )
    impute_valid = batch.get(
        "impute_mask", ~batch["seen"]
    ).to(trajectory_error.dtype)
    impute_ade = (
        trajectory_error[:, :, :8] * impute_valid[:, None]
    ).sum(dim=-1) / impute_valid.sum(dim=-1)[:, None].clamp_min(1)
    joint_valid = batch.get(
        "unknown", torch.ones_like(batch["target"][..., 0])
    ).to(trajectory_error.dtype)
    joint_ade = (
        trajectory_error * joint_valid[:, None]
    ).sum(dim=-1) / joint_valid.sum(dim=-1)[:, None].clamp_min(1)
    tail_count = max(1, int(round(0.1 * row_mean.shape[0])))
    soft_miss_rate = torch.sigmoid(
        (row_min_fde * scale - 2.0) / args.risk_target_mr_temperature
    ).mean() * 100.0
    metrics = (
        row_min_ade.mean() * scale,
        row_min_fde.mean() * scale,
        row_mean.mean() * scale,
        soft_miss_rate,
        row_mean.topk(tail_count, largest=True).values.mean() * scale,
        impute_ade.min(dim=1).values.mean() * scale,
        joint_ade.min(dim=1).values.mean() * scale,
    )
    targets = (
        args.risk_target_minade,
        args.risk_target_minfde,
        args.risk_target_meanade,
        args.risk_target_mr,
        args.risk_target_p90,
        args.risk_target_impute,
        args.risk_target_jointade,
    )
    return sum(
        F.relu(metric / target - 1.0).square()
        for metric, target in zip(metrics, targets)
    )


def train_conditional_mode_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Fit a multimodal set around the frozen conditional low-risk centre."""
    if not hasattr(model, "external_center"):
        raise RuntimeError("conditional mode training requires external center")
    if args.conditional_ranker_only:
        model.eval()
        model.set_candidate_ranker.train()
    elif args.conditional_mixer_only:
        model.eval()
        model.conditional_mode_mixer.train()
    else:
        model.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train conditional modes", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            center = external_center_trajectories(
                model.external_center, batch, args.external_center_kind, 1, False
            )[:, 0]
            refined = model.predict_motion_mean_trajectory(
                batch["partial"], batch["seen"], center,
                object_type=batch.get("object_type"),
            )
            center = center + args.external_center_refiner_strength * (
                refined - center
            )
            center_retrieval = None
            if args.risk_center_memory_residual_strength > 0:
                center_retrieval = memory.retrieve(
                    batch["key"], query_indices=batch["index"]
                )
                center_retrieval = adapt_retrieval(
                    center_retrieval, batch, args
                )
                center_residual = model.memory_set_center(
                    batch["partial"], batch["seen"], center_retrieval.values
                )
                center = center + args.risk_center_memory_residual_strength * (
                    center_residual
                    * model.free_mask(batch["seen"], center_residual.dtype)
                )
            if args.risk_center_enhanced_residual_strength > 0:
                if args.risk_center_memory_residual_strength <= 0:
                    center_retrieval = memory.retrieve(
                        batch["key"], query_indices=batch["index"]
                    )
                    center_retrieval = adapt_retrieval(
                        center_retrieval, batch, args
                    )
                enhanced_residual = model.predict_enhanced_memory_center(
                    batch["partial"],
                    batch["seen"],
                    center_retrieval.values,
                    object_type=batch.get("object_type"),
                )
                center = center + args.risk_center_enhanced_residual_strength * (
                    enhanced_residual
                    * model.free_mask(batch["seen"], enhanced_residual.dtype)
                )
            if args.memory_conditional_mode and center_retrieval is None:
                center_retrieval = adapt_retrieval(
                    memory.retrieve(
                        batch["key"], query_indices=batch["index"]
                    ),
                    batch,
                    args,
                )
            pool_center_copies = (
                1 if (
                    args.conditional_selected_modes > 0
                    or args.conditional_mixer_modes > 0
                )
                else args.conditional_center_copies
            )
            candidates = model.predict_conditional_modes(
                batch["partial"], batch["seen"], center, args.num_samples,
                object_type=batch.get("object_type"),
                center_copies=pool_center_copies,
                teacher_modes=args.conditional_teacher_modes,
                split_decoder=args.conditional_split_decoder,
                coverage_modes=args.conditional_coverage_modes,
                memory_values=(
                    center_retrieval.values
                    if center_retrieval is not None else None
                ),
                memory_weights=(
                    center_retrieval.weights
                    if center_retrieval is not None else None
                ),
                memory_conditional=args.memory_conditional_mode,
                memory_retrieval_init_strength=(
                    args.memory_conditional_retrieval_init_strength
                ),
            )
            if args.risk_center_temporal_power > 0:
                temporal_gate = torch.linspace(
                    1.0 / 12.0,
                    1.0,
                    12,
                    device=candidates.device,
                    dtype=candidates.dtype,
                ).pow(args.risk_center_temporal_power)
                candidates = candidates.clone()
                candidates[:, :, 8:] = (
                    center[:, None, 8:]
                    + temporal_gate[None, None, :, None]
                    * (candidates[:, :, 8:] - center[:, None, 8:])
                )
            mixer_pool = None
            mixed = None
            if args.conditional_mixer_modes > 0:
                mixer_pool = candidates[:, 1:]
                mixed = model.mix_conditional_modes(
                    batch["partial"],
                    batch["seen"],
                    mixer_pool,
                    args.conditional_mixer_modes,
                    temperature=args.conditional_mixer_temperature,
                    object_type=batch.get("object_type"),
                )
                candidates = torch.cat([
                    center[:, None].expand(
                        -1, args.conditional_center_copies, -1, -1
                    ),
                    mixed,
                ], dim=1)
            target = batch["target"][:, None]
            rank_loss = torch.zeros((), device=device)
            if args.conditional_selected_modes > 0:
                rank_pool = candidates[:, 1:]
                rank_future_error = torch.linalg.vector_norm(
                    rank_pool[:, :, 8:] - target[:, :, 8:], dim=-1
                )
                rank_future_valid = batch.get(
                    "future_valid",
                    torch.ones_like(
                        batch["target"][:, 8:, 0], dtype=torch.bool
                    ),
                ).to(rank_future_error.dtype)
                rank_ade = (
                    rank_future_error * rank_future_valid[:, None]
                ).sum(dim=-1) / rank_future_valid.sum(
                    dim=-1
                )[:, None].clamp_min(1)
                rank_reverse = torch.flip(
                    rank_future_valid.bool(), dims=[1]
                ).float().argmax(dim=1)
                rank_final = rank_future_valid.shape[1] - 1 - rank_reverse
                rank_fde = rank_future_error.gather(
                    2,
                    rank_final[:, None, None].expand(
                        -1, rank_pool.shape[1], 1
                    ),
                ).squeeze(-1)
                rank_cost = (
                    rank_ade + args.mode_set_rank_fde_weight * rank_fde
                )
                target_k = min(
                    int(args.conditional_selected_modes), rank_pool.shape[1]
                )
                target_index = rank_cost.topk(
                    target_k, dim=1, largest=False
                ).indices
                target_probability = torch.zeros_like(rank_cost).scatter_(
                    1, target_index, 1.0 / target_k
                )
                rank_logits = model.select_set_reference_logits(
                    batch["partial"], batch["seen"], rank_pool,
                    object_type=batch.get("object_type"),
                )
                rank_loss = -(
                    target_probability
                    * torch.log_softmax(
                        rank_logits / args.mode_set_rank_target_temperature,
                        dim=-1,
                    )
                ).sum(dim=-1).mean()
                candidates = select_conditional_mode_subset(
                    model, batch, center, candidates, args
                )
            future_error = torch.linalg.vector_norm(
                candidates[:, :, 8:] - target[:, :, 8:], dim=-1
            )
            future_valid = batch.get(
                "future_valid",
                torch.ones_like(batch["target"][:, 8:, 0], dtype=torch.bool),
            ).to(future_error.dtype)
            mixer_distill_loss = torch.zeros((), device=device)
            if (
                mixer_pool is not None
                and mixed is not None
                and args.conditional_mixer_distill_weight > 0
            ):
                pool_error = torch.linalg.vector_norm(
                    mixer_pool[:, :, 8:] - target[:, :, 8:], dim=-1
                )
                pool_ade = (
                    pool_error * future_valid[:, None]
                ).sum(dim=-1) / future_valid.sum(
                    dim=-1
                )[:, None].clamp_min(1)
                pool_reverse = torch.flip(
                    future_valid.bool(), dims=[1]
                ).float().argmax(dim=1)
                pool_final = future_valid.shape[1] - 1 - pool_reverse
                pool_fde = pool_error.gather(
                    2,
                    pool_final[:, None, None].expand(
                        -1, mixer_pool.shape[1], 1
                    ),
                ).squeeze(-1)
                teacher_indices = [pool_ade.argmin(dim=1)]
                if args.conditional_mixer_modes > 1:
                    teacher_indices.append(pool_fde.argmin(dim=1))
                blend_weights = (0.1, 0.5, 1.0, 2.0)
                for head_index in range(2, args.conditional_mixer_modes):
                    weight = blend_weights[min(
                        head_index - 2, len(blend_weights) - 1
                    )]
                    teacher_indices.append(
                        (pool_ade + weight * pool_fde).argmin(dim=1)
                    )
                teacher_index = torch.stack(teacher_indices, dim=1)
                rows = torch.arange(
                    mixer_pool.shape[0], device=mixer_pool.device
                )[:, None]
                teacher = mixer_pool[rows, teacher_index].detach()
                distill_error = torch.linalg.vector_norm(
                    mixed[:, :, 8:] - teacher[:, :, 8:], dim=-1
                )
                mixer_distill_loss = (
                    distill_error * future_valid[:, None]
                ).sum(dim=-1).div(
                    future_valid.sum(dim=-1)[:, None].clamp_min(1)
                ).mean()
                teacher_endpoint = teacher.gather(
                    2,
                    pool_final[:, None, None, None].expand(
                        -1, teacher.shape[1], 1, 2
                    ),
                ).squeeze(2)
                mixed_endpoint = mixed.gather(
                    2,
                    pool_final[:, None, None, None].expand(
                        -1, mixed.shape[1], 1, 2
                    ),
                ).squeeze(2)
                mixer_distill_loss = (
                    mixer_distill_loss
                    + args.conditional_mixer_distill_endpoint_weight
                    * torch.linalg.vector_norm(
                        mixed_endpoint - teacher_endpoint, dim=-1
                    ).mean()
                )
            ade = (
                future_error * future_valid[:, None]
            ).sum(dim=-1) / future_valid.sum(dim=-1)[:, None].clamp_min(1)
            reverse_index = torch.flip(
                future_valid.bool(), dims=[1]
            ).float().argmax(dim=1)
            final_index = future_valid.shape[1] - 1 - reverse_index
            fde = future_error.gather(
                2, final_index[:, None, None].expand(-1, args.num_samples, 1)
            ).squeeze(-1)
            row_mean = ade.mean(dim=1)
            if args.conditional_coverage_modes:
                coverage_ade = ade[:, -args.conditional_coverage_modes:]
                coverage_fde = fde[:, -args.conditional_coverage_modes:]
                risk_ade = ade[:, :-args.conditional_coverage_modes]
                risk_fde = fde[:, :-args.conditional_coverage_modes]
            else:
                coverage_ade = ade[:, args.conditional_center_copies:]
                coverage_fde = fde[:, args.conditional_center_copies:]
                risk_ade = ade
                risk_fde = fde
            risk_row_mean = risk_ade.mean(dim=1)
            early_mean_loss = torch.zeros((), device=device)
            if args.mode_set_early_mean_weight > 0:
                early_steps = min(
                    int(args.mode_set_early_mean_steps),
                    future_error.shape[-1],
                )
                early_valid = future_valid[:, :early_steps]
                early_ade = (
                    future_error[:, :, :early_steps]
                    * early_valid[:, None]
                ).sum(dim=-1) / early_valid.sum(
                    dim=-1
                )[:, None].clamp_min(1)
                early_mean_loss = early_ade.mean()
            coverage_modes = coverage_ade.shape[1]
            suffix_pattern = torch.tensor(
                [True, True, True, True, False, False, False, False],
                device=device,
            )
            suffix_rows = (batch["seen"] == suffix_pattern[None]).all(dim=1)
            row_weight = torch.ones_like(row_mean)
            row_weight = row_weight + (
                args.mode_set_suffix_weight - 1.0
            ) * suffix_rows.to(row_weight.dtype)

            def weighted_row_mean(value: torch.Tensor) -> torch.Tensor:
                while value.dim() > 1:
                    value = value.mean(dim=-1)
                return (value * row_weight).sum() / row_weight.sum().clamp_min(1)

            coverage_mean_loss = weighted_row_mean(coverage_ade)
            coverage_oracle_loss = weighted_row_mean(
                coverage_ade.topk(
                    min(args.mode_set_oracle_topk, coverage_modes),
                    dim=1, largest=False,
                ).values
            )
            coverage_fde_loss = weighted_row_mean(
                coverage_fde.topk(
                    min(args.mode_set_fde_topk, coverage_modes),
                    dim=1, largest=False,
                ).values
            )
            risk_slot_loss = torch.zeros((), device=device)
            if args.mode_set_risk_slot_indices:
                valid_risk_slots = [
                    index for index in args.mode_set_risk_slot_indices
                    if 0 <= index < coverage_modes
                ]
                if valid_risk_slots:
                    risk_slot_loss = weighted_row_mean(
                        coverage_ade[:, valid_risk_slots]
                    )
            target_endpoint = batch["target"].gather(
                1, final_index[:, None, None].expand(-1, 1, 2)
            ).squeeze(1)
            center_endpoint = center.gather(
                1, final_index[:, None, None].expand(-1, 1, 2)
            ).squeeze(1)
            endpoint_delta = target_endpoint - center_endpoint
            endpoint_angle = torch.atan2(
                endpoint_delta[:, 1], endpoint_delta[:, 0]
            )
            assigned_mode = torch.floor(
                (endpoint_angle + math.pi) / (2 * math.pi) * coverage_modes
            ).long().clamp(0, coverage_modes - 1)
            assigned_ade = coverage_ade.gather(
                1, assigned_mode[:, None]
            ).squeeze(1)
            assigned_fde = coverage_fde.gather(
                1, assigned_mode[:, None]
            ).squeeze(1)
            loss = (
                args.mode_set_mean_weight * risk_row_mean.mean()
                + args.mode_set_early_mean_weight * early_mean_loss
                + args.mode_set_coverage_mean_weight * coverage_mean_loss
                + args.mode_set_oracle_weight
                * coverage_oracle_loss
                + args.mode_set_fde_weight
                * coverage_fde_loss
                + args.mode_set_risk_slot_mean_weight * risk_slot_loss
                + args.mode_set_mean_fde_weight * risk_fde.mean()
                + args.mode_set_assignment_weight * assigned_ade.mean()
                + args.mode_set_assignment_fde_weight * assigned_fde.mean()
                + args.mode_set_rank_weight * rank_loss
                + args.conditional_mixer_distill_weight * mixer_distill_loss
            )
            if args.risk_target_hinge_weight > 0:
                loss = loss + args.risk_target_hinge_weight * (
                    metric_target_hinge_loss(
                        candidates, batch, ade, fde, row_mean, args
                    )
                )
            if args.conditional_teacher_modes and args.mode_set_rank_weight > 0:
                teacher_pool = model.predict_frozen_coverage_pool(
                    batch["partial"], batch["seen"], center,
                    object_type=batch.get("object_type"),
                )
                teacher_error = torch.linalg.vector_norm(
                    teacher_pool[:, :, 8:] - target[:, :, 8:], dim=-1
                )
                teacher_ade = (
                    teacher_error * future_valid[:, None]
                ).sum(dim=-1) / future_valid.sum(dim=-1)[:, None].clamp_min(1)
                teacher_fde = teacher_error.gather(
                    2,
                    final_index[:, None, None].expand(
                        -1, teacher_pool.shape[1], 1
                    ),
                ).squeeze(-1)
                teacher_cost = (
                    teacher_ade
                    + args.mode_set_rank_fde_weight * teacher_fde
                )
                teacher_logits = model.select_set_reference_logits(
                    batch["partial"], batch["seen"], teacher_pool,
                    object_type=batch.get("object_type"),
                )
                teacher_distribution = torch.softmax(
                    -teacher_cost / args.mode_set_rank_target_temperature,
                    dim=-1,
                )
                rank_loss = -(
                    teacher_distribution
                    * torch.log_softmax(teacher_logits, dim=-1)
                ).sum(dim=-1).mean()
                loss = loss + args.mode_set_rank_weight * rank_loss
            if args.mode_set_cvar_weight > 0:
                cvar_count = max(1, int(round(
                    args.mode_set_cvar_fraction * risk_row_mean.shape[0]
                )))
                loss = loss + args.mode_set_cvar_weight * risk_row_mean.topk(
                    cvar_count, largest=True
                ).values.mean()
            if args.mode_set_diversity_weight > 0 and args.num_samples > 1:
                endpoints = candidates[:, -coverage_modes:, -1]
                distance = torch.cdist(endpoints, endpoints)
                off_diagonal = ~torch.eye(
                    coverage_modes, device=distance.device, dtype=torch.bool
                )[None]
                diversity = F.relu(
                    args.mode_set_diversity_margin - distance
                )[off_diagonal.expand_as(distance)].mean()
                loss = loss + args.mode_set_diversity_weight * diversity
        scaler.scale(loss / args.grad_accum_steps).backward()
        if batch_index % args.grad_accum_steps == 0 or batch_index == len(loader):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def train_dual_pool_selector_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Fit target-free inference scores for repair and forecast responsibilities."""
    if not hasattr(model, "external_center"):
        raise RuntimeError("dual-pool selector training requires external center")
    model.eval()
    model.dual_pool_selector.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train dual-pool selector", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            with torch.no_grad():
                center = external_center_trajectories(
                    model.external_center,
                    batch,
                    args.external_center_kind,
                    1,
                    False,
                )[:, 0]
                refined = model.predict_motion_mean_trajectory(
                    batch["partial"],
                    batch["seen"],
                    center,
                    object_type=batch.get("object_type"),
                )
                center = center + args.risk_center_refiner_strength * (
                    refined - center
                )
                retrieval = adapt_retrieval(
                    memory.retrieve(batch["key"], query_indices=batch["index"]),
                    batch,
                    args,
                )
                if args.risk_center_memory_residual_strength > 0:
                    residual = model.memory_set_center(
                        batch["partial"], batch["seen"], retrieval.values
                    )
                    center = center + args.risk_center_memory_residual_strength * (
                        residual * model.free_mask(batch["seen"], residual.dtype)
                    )
                if args.risk_center_enhanced_residual_strength > 0:
                    residual = model.predict_enhanced_memory_center(
                        batch["partial"],
                        batch["seen"],
                        retrieval.values,
                        object_type=batch.get("object_type"),
                    )
                    center = center + args.risk_center_enhanced_residual_strength * (
                        residual * model.free_mask(batch["seen"], residual.dtype)
                    )
                pool = model.predict_dual_pool(
                    batch["partial"],
                    batch["seen"],
                    center,
                    args.num_samples,
                    object_type=batch.get("object_type"),
                )
            logits = model.dual_pool_selector(
                batch["partial"],
                batch["seen"],
                pool,
                retrieval.values,
                retrieval.weights,
            )
            target = batch["target"][:, None]
            future_error = torch.linalg.vector_norm(
                pool[:, :, 8:] - target[:, :, 8:], dim=-1
            )
            future_valid = batch.get(
                "future_valid",
                torch.ones_like(batch["target"][:, 8:, 0], dtype=torch.bool),
            ).to(future_error.dtype)
            ade = (future_error * future_valid[:, None]).sum(dim=-1) / (
                future_valid.sum(dim=-1)[:, None].clamp_min(1)
            )
            reverse = torch.flip(future_valid.bool(), dims=[1]).float().argmax(dim=1)
            final_index = future_valid.shape[1] - 1 - reverse
            fde = future_error.gather(
                2, final_index[:, None, None].expand(-1, pool.shape[1], 1)
            ).squeeze(-1)
            missing = batch.get("impute_mask", ~batch["seen"]).to(future_error.dtype)
            history_error = torch.linalg.vector_norm(
                pool[:, :, :8] - target[:, :, :8], dim=-1
            )
            missing_count = missing.sum(dim=-1)
            imputation = (history_error * missing[:, None]).sum(dim=-1) / (
                missing_count[:, None].clamp_min(1)
            )
            joint = (
                (history_error * missing[:, None]).sum(dim=-1)
                + (future_error * future_valid[:, None]).sum(dim=-1)
            ) / (
                missing_count + future_valid.sum(dim=-1)
            )[:, None].clamp_min(1)
            costs = tuple(
                value.detach() for value in (imputation, joint, ade, fde, ade)
            )

            def target_distribution(cost):
                normalized = (cost - cost.mean(dim=1, keepdim=True)) / (
                    cost.std(dim=1, keepdim=True).clamp_min(1e-4)
                )
                return torch.softmax(
                    -normalized / args.dual_pool_target_temperature, dim=-1
                )

            loss = torch.zeros((), device=device)
            if args.dual_pool_composite_selector_only:
                modes_per_expert = args.num_samples
                coverage_slice = slice(0, modes_per_expert)
                mean_slice = slice(3 * modes_per_expert, 4 * modes_per_expert)
                coverage_cost = (
                    costs[2][:, coverage_slice] / args.risk_target_minade
                    + args.dual_pool_forecast_fde_weight
                    * costs[3][:, coverage_slice] / args.risk_target_minfde
                    + args.dual_pool_forecast_joint_weight
                    * costs[1][:, coverage_slice] / args.risk_target_jointade
                    + args.dual_pool_repair_joint_weight
                    * costs[0][:, coverage_slice] / args.risk_target_impute
                )
                normalized_coverage_cost = (
                    coverage_cost
                    - coverage_cost.mean(dim=1, keepdim=True)
                ) / coverage_cost.std(dim=1, keepdim=True).clamp_min(1e-4)
                coverage_probability = torch.softmax(
                    -normalized_coverage_cost
                    / args.dual_pool_target_temperature,
                    dim=1,
                )
                role_only_loss = F.cross_entropy(
                    logits[2][:, coverage_slice],
                    coverage_cost.argmin(dim=1),
                ) - (
                    coverage_probability
                    * torch.log_softmax(
                        logits[2][:, coverage_slice], dim=1
                    )
                ).sum(dim=1).mean() + F.cross_entropy(
                    logits[4][:, mean_slice],
                    costs[4][:, mean_slice].argmin(dim=1),
                )
                loss = role_only_loss
            elif args.dual_pool_minade_selector_only:
                modes_per_expert = args.num_samples
                coverage_slice = slice(0, modes_per_expert)
                mean_slice = slice(3 * modes_per_expert, 4 * modes_per_expert)
                role_only_loss = F.cross_entropy(
                    logits[2][:, coverage_slice],
                    costs[2][:, coverage_slice].argmin(dim=1),
                ) + F.cross_entropy(
                    logits[4][:, mean_slice],
                    costs[4][:, mean_slice].argmin(dim=1),
                )
                loss = role_only_loss
            elif args.dual_pool_coverage_specialist_only:
                # Each head is trained only on the pool it is allowed to use
                # at inference.  This prevents the global ranking objective
                # from teaching a coverage head to prefer a mean-expert mode.
                modes_per_expert = args.num_samples
                coverage_slice = slice(0, modes_per_expert)
                mean_slice = slice(3 * modes_per_expert, 4 * modes_per_expert)
                role_losses = [
                    F.cross_entropy(
                        logits[0], costs[0].argmin(dim=1)
                    )
                ]
                for specialist_logits, specialist_cost in zip(
                    logits[1:4], costs[1:4]
                ):
                    role_losses.append(
                        F.cross_entropy(
                            specialist_logits[:, coverage_slice],
                            specialist_cost[:, coverage_slice].argmin(dim=1),
                        )
                    )
                role_losses.append(
                    F.cross_entropy(
                        logits[4][:, mean_slice],
                        costs[4][:, mean_slice].argmin(dim=1),
                    )
                )
                role_only_loss = sum(role_losses)
                if args.dual_pool_coverage_specialist_soft_weight > 0:
                    soft_losses = []
                    for specialist_logits, specialist_cost in zip(
                        logits[1:4], costs[1:4]
                    ):
                        restricted = specialist_cost[:, coverage_slice]
                        target_probability = target_distribution(restricted)
                        soft_losses.append(
                            -(
                                target_probability
                                * torch.log_softmax(
                                    specialist_logits[:, coverage_slice], dim=-1
                                )
                            ).sum(dim=-1).mean()
                        )
                    restricted = costs[4][:, mean_slice]
                    target_probability = target_distribution(restricted)
                    soft_losses.append(
                        -(
                            target_probability
                            * torch.log_softmax(
                                logits[4][:, mean_slice], dim=-1
                            )
                        ).sum(dim=-1).mean()
                    )
                    role_only_loss = role_only_loss + (
                        args.dual_pool_coverage_specialist_soft_weight
                        * sum(soft_losses)
                    )
                if args.dual_pool_coverage_regression_weight > 0:
                    def restricted_regression(logits_value, cost_value, mode_slice):
                        restricted_cost = cost_value[:, mode_slice]
                        normalized_cost = (
                            restricted_cost
                            - restricted_cost.mean(dim=1, keepdim=True)
                        ) / restricted_cost.std(
                            dim=1, keepdim=True
                        ).clamp_min(1e-4)
                        restricted_logits = logits_value[:, mode_slice]
                        normalized_logits = (
                            restricted_logits
                            - restricted_logits.mean(dim=1, keepdim=True)
                        ) / restricted_logits.std(
                            dim=1, keepdim=True
                        ).clamp_min(1e-4)
                        return F.smooth_l1_loss(
                            normalized_logits, -normalized_cost
                        )

                    regression_losses = [
                        restricted_regression(
                            specialist_logits,
                            specialist_cost,
                            coverage_slice,
                        )
                        for specialist_logits, specialist_cost in zip(
                            logits[1:4], costs[1:4]
                        )
                    ]
                    regression_losses.append(
                        restricted_regression(
                            logits[4], costs[4], mean_slice
                        )
                    )
                    role_only_loss = role_only_loss + (
                        args.dual_pool_coverage_regression_weight
                        * sum(regression_losses)
                    )
                loss = role_only_loss
            else:
            # Four sharp specialist heads: imputation, JointADE, minADE, minFDE.
                for specialist_logits, specialist_cost in zip(
                    logits[:4], costs[:4]
                ):
                    target_probability = target_distribution(specialist_cost)
                    loss = loss - (
                        target_probability
                        * torch.log_softmax(specialist_logits, dim=-1)
                    ).sum(dim=-1).mean()
            # Explicit multi-label supervision keeps the mean-risk head's
            # complete top-eight set low, rather than learning one winner only.
            topk_values = (1, 1, 1, 1, args.num_samples - 4)
            for current_logits, current_cost, target_k in zip(
                logits, costs, topk_values
            ):
                target_index = current_cost.topk(
                    min(target_k, current_cost.shape[1]),
                    dim=1,
                    largest=False,
                ).indices
                target_label = torch.zeros_like(current_logits).scatter_(
                    1, target_index, 1.0
                )
                loss = loss + args.dual_pool_topk_weight * (
                    F.binary_cross_entropy_with_logits(
                        current_logits, target_label
                    )
                )
            if args.dual_pool_pairwise_weight > 0:
                def pairwise_loss(logits, cost):
                    normalized = (cost - cost.mean(dim=1, keepdim=True)) / (
                        cost.std(dim=1, keepdim=True).clamp_min(1e-4)
                    )
                    target_order = torch.sigmoid(
                        (
                            normalized[:, None, :]
                            - normalized[:, :, None]
                        ) / args.dual_pool_pairwise_temperature
                    )
                    predicted_order = (
                        logits[:, :, None] - logits[:, None, :]
                    )
                    return F.binary_cross_entropy_with_logits(
                        predicted_order, target_order
                    )

                loss = loss + args.dual_pool_pairwise_weight * sum(
                    pairwise_loss(current_logits, current_cost)
                    for current_logits, current_cost in zip(logits, costs)
                )
            if args.dual_pool_regression_weight > 0:
                def regression_loss(logits, cost):
                    normalized_cost = (
                        cost - cost.mean(dim=1, keepdim=True)
                    ) / cost.std(dim=1, keepdim=True).clamp_min(1e-4)
                    normalized_logits = (
                        logits - logits.mean(dim=1, keepdim=True)
                    ) / logits.std(dim=1, keepdim=True).clamp_min(1e-4)
                    return F.smooth_l1_loss(
                        normalized_logits, -normalized_cost
                    )

                # Direct regression is intentionally limited to the low-mean
                # head; applying it to specialists collapses coverage.
                loss = loss + args.dual_pool_regression_weight * regression_loss(
                    logits[-1], costs[-1]
                )
            if (not args.dual_pool_coverage_specialist_only) and args.dual_pool_coverage_specialist_weight > 0:
                modes_per_expert = args.num_samples
                coverage_slice = slice(0, modes_per_expert)
                mean_slice = slice(3 * modes_per_expert, 4 * modes_per_expert)
                specialist_losses = []
                for specialist, cost in zip(
                    logits[1:4], costs[1:4]
                ):
                    specialist_cost = cost[:, coverage_slice]
                    specialist_losses.append(
                        F.cross_entropy(
                            specialist[:, coverage_slice],
                            specialist_cost.argmin(dim=1),
                        )
                    )
                specialist_losses.append(
                    F.cross_entropy(
                        logits[4][:, mean_slice],
                        costs[4][:, mean_slice].argmin(dim=1),
                    )
                )
                loss = loss + args.dual_pool_coverage_specialist_weight * sum(
                    specialist_losses
                )
            if (
                args.dual_pool_coverage_specialist_only
                or args.dual_pool_minade_selector_only
                or args.dual_pool_composite_selector_only
            ):
                loss = role_only_loss
            if args.dual_pool_adaptive_gate:
                gate_logits = model.dual_pool_selector.forward_mixture_gate(
                    batch["partial"],
                    batch["seen"],
                    retrieval.values,
                    retrieval.weights,
                )
                def set_score(index):
                    # The gate target is computed from train labels only.  It
                    # uses the same normalized risk units as the audit and
                    # does not inspect the held-out test split.
                    # Choose the allocation by its worst normalized audit
                    # risk.  A sum can hide one failed constraint behind four
                    # easy ones and systematically over-allocate coverage;
                    # minimax supervision instead learns the smallest
                    # query-specific coverage budget that balances all roles.
                    return torch.stack(
                        [
                            costs[0][:, index].min(dim=1).values
                            / args.risk_target_impute,
                            costs[2][:, index].min(dim=1).values
                            / args.risk_target_minade,
                            costs[3][:, index].min(dim=1).values
                            / args.risk_target_minfde,
                            costs[1][:, index].min(dim=1).values
                            / args.risk_target_jointade,
                            args.dual_pool_adaptive_gate_mean_weight
                            * costs[4][:, index].mean(dim=1)
                            / args.risk_target_meanade,
                        ],
                        dim=1,
                    ).amax(dim=1)

                if args.dual_pool_adaptive_multiclass_gate:
                    count_values = (0, 1, 3, 6, 9, 12)
                    count_scores = []
                    for count_value in count_values:
                        index = torch.cat(
                            [
                                torch.arange(count_value, device=device),
                                torch.arange(
                                    3 * args.num_samples,
                                    4 * args.num_samples,
                                    device=device,
                                )[: args.num_samples - count_value],
                            ]
                        )
                        count_scores.append(set_score(index))
                    gate_target = torch.stack(count_scores, dim=1).argmin(dim=1)
                    loss = loss + args.dual_pool_adaptive_gate_loss_weight * (
                        F.cross_entropy(gate_logits, gate_target)
                    )
                else:
                    low_index = torch.cat(
                        [
                            torch.arange(
                                args.dual_pool_adaptive_min_count,
                                device=device,
                            ),
                            torch.arange(
                                3 * args.num_samples,
                                4 * args.num_samples,
                                device=device,
                            )[
                                : args.num_samples
                                - args.dual_pool_adaptive_min_count
                            ],
                        ]
                    )
                    high_index = torch.cat(
                        [
                            torch.arange(
                                args.dual_pool_adaptive_max_count,
                                device=device,
                            ),
                            torch.arange(
                                3 * args.num_samples,
                                4 * args.num_samples,
                                device=device,
                            )[
                                : args.num_samples
                                - args.dual_pool_adaptive_max_count
                            ],
                        ]
                    )
                    low_gain_cost = (
                        costs[2][:, low_index].min(dim=1).values
                        / args.risk_target_minade
                        + 0.3 * costs[3][:, low_index].min(dim=1).values
                        / args.risk_target_minfde
                        + 0.2 * costs[1][:, low_index].min(dim=1).values
                        / args.risk_target_jointade
                    )
                    high_gain_cost = (
                        costs[2][:, high_index].min(dim=1).values
                        / args.risk_target_minade
                        + 0.3 * costs[3][:, high_index].min(dim=1).values
                        / args.risk_target_minfde
                        + 0.2 * costs[1][:, high_index].min(dim=1).values
                        / args.risk_target_jointade
                    )
                    gain = (low_gain_cost - high_gain_cost).detach()
                    gain_threshold = torch.quantile(
                        gain,
                        args.dual_pool_adaptive_gate_gain_quantile,
                    )
                    gate_target = (gain >= gain_threshold).to(
                        gate_logits.dtype
                    )
                    loss = loss + args.dual_pool_adaptive_gate_loss_weight * (
                        F.binary_cross_entropy_with_logits(
                            gate_logits, gate_target
                        )
                    )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.dual_pool_selector.parameters(), args.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach()) * batch["target"].shape[0]
        count += batch["target"].shape[0]
    return total / max(count, 1)


def train_mode_alignment_gate_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Distill the train-only optimal mode projection into a target-free gate."""
    if not hasattr(model, "external_center"):
        raise RuntimeError("mode-alignment training requires external center")
    model.eval()
    model.mode_alignment_gate.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for raw_batch in tqdm(
        loader, desc="train mode-alignment gate", leave=False
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            with torch.no_grad():
                center = external_center_trajectories(
                    model.external_center,
                    batch,
                    args.external_center_kind,
                    1,
                    False,
                )[:, 0]
                refined = model.predict_motion_mean_trajectory(
                    batch["partial"],
                    batch["seen"],
                    center,
                    object_type=batch.get("object_type"),
                )
                center = center + args.risk_center_refiner_strength * (
                    refined - center
                )
                retrieval = adapt_retrieval(
                    memory.retrieve(batch["key"], query_indices=batch["index"]),
                    batch,
                    args,
                )
                if args.risk_center_memory_residual_strength > 0:
                    residual = model.memory_set_center(
                        batch["partial"], batch["seen"], retrieval.values
                    )
                    center = center + args.risk_center_memory_residual_strength * (
                        residual * model.free_mask(batch["seen"], residual.dtype)
                    )
                if args.risk_center_enhanced_residual_strength > 0:
                    residual = model.predict_enhanced_memory_center(
                        batch["partial"],
                        batch["seen"],
                        retrieval.values,
                        object_type=batch.get("object_type"),
                    )
                    center = center + args.risk_center_enhanced_residual_strength * (
                        residual * model.free_mask(batch["seen"], residual.dtype)
                    )
                pool = model.predict_dual_pool(
                    batch["partial"],
                    batch["seen"],
                    center,
                    args.num_samples,
                    object_type=batch.get("object_type"),
                )
                start = args.mode_alignment_expert_index * args.num_samples
                coverage = pool[:, : args.num_samples]
                low_risk = pool[:, start : start + args.num_samples]
            aligned, alpha, aligned_mean = model.align_coverage_modes(
                batch["partial"],
                batch["seen"],
                coverage,
                low_risk,
                retrieval.values,
                retrieval.weights,
            )
            future_valid = batch.get(
                "future_valid",
                torch.ones_like(batch["target"][:, 8:, 0], dtype=torch.bool),
            ).to(aligned.dtype)
            direction = coverage[:, :, 8:] - aligned_mean[:, :, 8:]
            target_delta = (
                batch["target"][:, None, 8:] - aligned_mean[:, :, 8:]
            )
            valid = future_valid[:, None, :, None]
            projection_target = (
                (direction * target_delta * valid).sum(dim=(-1, -2))
                / direction.square().mul(valid).sum(
                    dim=(-1, -2)
                ).clamp_min(1e-8)
            ).clamp(0, 1).detach()
            projection_label = projection_target
            if args.mode_alignment_projection_binary_target:
                projection_label = (projection_target >= .5).to(alpha.dtype)
            if args.mode_alignment_projection_loss == "bce":
                probability = alpha.clamp(min=1e-5, max=1 - 1e-5)
                projection_loss = -(
                    args.mode_alignment_projection_positive_weight
                    * projection_label * probability.log()
                    + (1 - projection_label) * (1 - probability).log()
                ).mean()
            else:
                projection_loss = F.smooth_l1_loss(alpha, projection_label)
            future_error = torch.linalg.vector_norm(
                aligned[:, :, 8:] - batch["target"][:, None, 8:], dim=-1
            )
            ade = (future_error * future_valid[:, None]).sum(dim=-1) / (
                future_valid.sum(dim=-1)[:, None].clamp_min(1)
            )
            reverse = torch.flip(future_valid.bool(), dims=[1]).float().argmax(dim=1)
            final_index = future_valid.shape[1] - 1 - reverse
            fde = future_error.gather(
                2, final_index[:, None, None].expand(-1, aligned.shape[1], 1)
            ).squeeze(-1)
            coverage_error = torch.linalg.vector_norm(
                coverage[:, :, 8:] - batch["target"][:, None, 8:], dim=-1
            )
            coverage_ade = (
                coverage_error * future_valid[:, None]
            ).sum(dim=-1) / future_valid.sum(dim=-1)[:, None].clamp_min(1)
            coverage_fde = coverage_error.gather(
                2,
                final_index[:, None, None].expand(-1, coverage.shape[1], 1),
            ).squeeze(-1)
            ranking_loss = torch.zeros((), device=device)
            if args.mode_alignment_ranking_weight > 0:
                rank_probability = model.mode_alignment_gate.last_rank_scores.clamp(
                    min=1e-5, max=1.0 - 1e-5
                )
                rank_logits = torch.logit(rank_probability)
                # Explicitly retain both the best-displacement and best-endpoint
                # candidates. The remaining preserved slots follow a joint cost.
                ranking_loss = F.cross_entropy(
                    rank_logits, coverage_ade.argmin(dim=1)
                ) + F.cross_entropy(rank_logits, coverage_fde.argmin(dim=1))
                preserve_count = max(
                    1, min(args.mode_alignment_preserve_count, coverage.shape[1])
                )
                joint_cost = coverage_ade + (
                    args.mode_alignment_ranking_fde_weight * coverage_fde
                )
                target_index = joint_cost.topk(
                    preserve_count, dim=1, largest=False
                ).indices
                target_label = torch.zeros_like(rank_logits).scatter_(
                    1, target_index, 1.0
                )
                positive_weight = float(
                    max(1, coverage.shape[1] - preserve_count)
                ) / float(preserve_count)
                ranking_loss = ranking_loss + F.binary_cross_entropy_with_logits(
                    rank_logits,
                    target_label,
                    pos_weight=torch.as_tensor(
                        positive_weight, device=device, dtype=rank_logits.dtype
                    ),
                )
            row_mean = ade.mean(dim=1)
            cvar_count = max(1, int(row_mean.shape[0] * .2))
            loss = (
                args.mode_alignment_projection_weight * projection_loss
                + args.mode_alignment_mean_weight * row_mean.mean()
                + args.mode_alignment_oracle_weight
                * ade.min(dim=1).values.mean()
                + args.mode_alignment_fde_weight
                * fde.min(dim=1).values.mean()
                + args.mode_alignment_cvar_weight
                * row_mean.topk(cvar_count).values.mean()
                + args.mode_alignment_ranking_weight * ranking_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.mode_alignment_gate.parameters(), args.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach()) * batch["target"].shape[0]
        count += batch["target"].shape[0]
    return total / max(count, 1)


def train_coverage_compression_mixer_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Compress coverage into ADE/FDE specialists beside frozen low-risk modes."""
    if not hasattr(model, "external_center"):
        raise RuntimeError("coverage compression requires external center")
    model.eval()
    model.conditional_mode_mixer.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for raw_batch in tqdm(
        loader, desc="train coverage compression", leave=False
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            with torch.no_grad():
                center = external_center_trajectories(
                    model.external_center,
                    batch,
                    args.external_center_kind,
                    1,
                    False,
                )[:, 0]
                refined = model.predict_motion_mean_trajectory(
                    batch["partial"], batch["seen"], center,
                    object_type=batch.get("object_type"),
                )
                center = center + args.risk_center_refiner_strength * (
                    refined - center
                )
                retrieval = adapt_retrieval(
                    memory.retrieve(batch["key"], query_indices=batch["index"]),
                    batch,
                    args,
                )
                if args.risk_center_memory_residual_strength > 0:
                    residual = model.memory_set_center(
                        batch["partial"], batch["seen"], retrieval.values
                    )
                    center = center + args.risk_center_memory_residual_strength * (
                        residual * model.free_mask(batch["seen"], residual.dtype)
                    )
                if args.risk_center_enhanced_residual_strength > 0:
                    residual = model.predict_enhanced_memory_center(
                        batch["partial"], batch["seen"], retrieval.values,
                        object_type=batch.get("object_type"),
                    )
                    center = center + args.risk_center_enhanced_residual_strength * (
                        residual * model.free_mask(batch["seen"], residual.dtype)
                    )
                pool = model.predict_dual_pool(
                    batch["partial"], batch["seen"], center,
                    args.num_samples, object_type=batch.get("object_type"),
                )
                coverage = pool[:, : args.num_samples]
            compressed, selection_weights = model.conditional_mode_mixer(
                batch["partial"],
                batch["seen"],
                coverage,
                args.coverage_compression_modes,
                temperature=args.coverage_compression_temperature,
                object_type=batch.get("object_type"),
                return_weights=True,
            )
            target = batch["target"][:, None]
            future_valid = batch.get(
                "future_valid",
                torch.ones_like(batch["target"][:, 8:, 0], dtype=torch.bool),
            ).to(compressed.dtype)

            def candidate_errors(samples):
                distance = torch.linalg.vector_norm(
                    samples[:, :, 8:] - target[:, :, 8:], dim=-1
                )
                ade = (distance * future_valid[:, None]).sum(dim=-1) / (
                    future_valid.sum(dim=-1)[:, None].clamp_min(1)
                )
                reverse = torch.flip(
                    future_valid.bool(), dims=[1]
                ).float().argmax(dim=1)
                final_index = future_valid.shape[1] - 1 - reverse
                fde = distance.gather(
                    2,
                    final_index[:, None, None].expand(-1, samples.shape[1], 1),
                ).squeeze(-1)
                return ade, fde, distance

            coverage_ade, coverage_fde, _ = candidate_errors(coverage)
            compressed_ade, compressed_fde, compressed_distance = (
                candidate_errors(compressed)
            )
            rows = torch.arange(coverage.shape[0], device=device)
            teacher_index = coverage_ade.topk(
                args.coverage_compression_modes,
                dim=1,
                largest=False,
                sorted=True,
            ).indices
            teacher = coverage[rows[:, None], teacher_index]
            selector_loss = -torch.log(
                selection_weights.gather(
                    2, teacher_index[:, :, None]
                ).squeeze(-1).clamp_min(1e-8)
            ).mean()
            distill = torch.linalg.vector_norm(
                compressed[:, :, 8:] - teacher[:, :, 8:], dim=-1
            )
            distill = (distill * future_valid[:, None]).sum(dim=-1) / (
                future_valid.sum(dim=-1)[:, None].clamp_min(1)
            )
            endpoint_distance = torch.cdist(
                compressed[:, :, -1], compressed[:, :, -1]
            )
            off_diagonal = ~torch.eye(
                compressed.shape[1], device=device, dtype=torch.bool
            )[None]
            diversity = F.relu(
                args.coverage_compression_diversity_margin
                - endpoint_distance
            )[off_diagonal.expand_as(endpoint_distance)].mean()
            loss = (
                args.coverage_compression_ade_weight
                * compressed_ade.min(dim=1).values.mean()
                + args.coverage_compression_fde_weight
                * compressed_fde.min(dim=1).values.mean()
                + args.coverage_compression_aux_weight
                * (compressed_ade.mean() + 0.25 * compressed_fde.mean())
                + args.coverage_compression_distill_weight * distill.mean()
                + args.coverage_compression_selector_weight * selector_loss
                + args.coverage_compression_diversity_weight * diversity
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.conditional_mode_mixer.parameters(), args.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach()) * batch["target"].shape[0]
        count += batch["target"].shape[0]
    return total / max(count, 1)


def train_direct_coverage_experts_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Train a small retrieval-conditioned expert set beside low-risk modes."""
    if not hasattr(model, "external_center"):
        raise RuntimeError("direct coverage experts require external center")
    model.eval()
    model.memory_conditional_mode_decoder.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for raw_batch in tqdm(
        loader, desc="train direct coverage experts", leave=False
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            with torch.no_grad():
                base_center = external_center_trajectories(
                    model.external_center,
                    batch,
                    args.external_center_kind,
                    1,
                    False,
                )[:, 0]
                refined = model.predict_motion_mean_trajectory(
                    batch["partial"], batch["seen"], base_center,
                    object_type=batch.get("object_type"),
                )
                center = base_center + args.risk_center_refiner_strength * (
                    refined - base_center
                )
                retrieval = adapt_retrieval(
                    memory.retrieve(batch["key"], query_indices=batch["index"]),
                    batch,
                    args,
                )
                if args.risk_center_memory_residual_strength > 0:
                    residual = model.memory_set_center(
                        batch["partial"], batch["seen"], retrieval.values
                    )
                    center = center + args.risk_center_memory_residual_strength * (
                        residual * model.free_mask(batch["seen"], residual.dtype)
                    )
                if args.risk_center_enhanced_residual_strength > 0:
                    residual = model.predict_enhanced_memory_center(
                        batch["partial"], batch["seen"], retrieval.values,
                        object_type=batch.get("object_type"),
                    )
                    center = center + args.risk_center_enhanced_residual_strength * (
                        residual * model.free_mask(batch["seen"], residual.dtype)
                    )
            residual, gate = model.memory_conditional_mode_decoder(
                batch["partial"],
                batch["seen"],
                center,
                retrieval.values,
                args.direct_coverage_modes,
                memory_weights=retrieval.weights,
                object_type=batch.get("object_type"),
                retrieval_init_strength=args.direct_coverage_retrieval_strength,
            )
            experts = center[:, None] + residual * gate * model.free_mask(
                batch["seen"], residual.dtype
            )[:, None]
            target = batch["target"][:, None]
            future_valid = batch.get(
                "future_valid",
                torch.ones_like(batch["target"][:, 8:, 0], dtype=torch.bool),
            ).to(experts.dtype)
            distance = torch.linalg.vector_norm(
                experts[:, :, 8:] - target[:, :, 8:], dim=-1
            )
            ade = (distance * future_valid[:, None]).sum(dim=-1) / (
                future_valid.sum(dim=-1)[:, None].clamp_min(1)
            )
            reverse = torch.flip(
                future_valid.bool(), dims=[1]
            ).float().argmax(dim=1)
            final_index = future_valid.shape[1] - 1 - reverse
            fde = distance.gather(
                2,
                final_index[:, None, None].expand(-1, experts.shape[1], 1),
            ).squeeze(-1)
            endpoints = experts[:, :, -1]
            endpoint_distance = torch.cdist(endpoints, endpoints)
            off_diagonal = ~torch.eye(
                experts.shape[1], device=device, dtype=torch.bool
            )[None]
            diversity = F.relu(
                args.direct_coverage_diversity_margin - endpoint_distance
            )[off_diagonal.expand_as(endpoint_distance)].mean()
            loss = (
                args.direct_coverage_oracle_weight
                * ade.min(dim=1).values.mean()
                + args.direct_coverage_fde_weight
                * fde.min(dim=1).values.mean()
                + args.direct_coverage_mean_weight * ade.mean()
                + args.direct_coverage_diversity_weight * diversity
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.memory_conditional_mode_decoder.parameters(), args.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach()) * batch["target"].shape[0]
        count += batch["target"].shape[0]
    return total / max(count, 1)


def train_risk_center_set_refiner_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Fit nonlinear low-risk/coverage corrections around a frozen centre."""
    if memory is None or not hasattr(model, "external_center"):
        raise RuntimeError("risk-set training requires memory and a centre model")
    model.eval()
    model.set_candidate_refiner.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train risk set refiner", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            with torch.no_grad():
                retrieval = memory.retrieve(
                    batch["key"], query_indices=batch["index"]
                )
                retrieval = adapt_retrieval(retrieval, batch, args)
                base_center = external_center_trajectories(
                    model.external_center,
                    batch,
                    args.external_center_kind,
                    1,
                    False,
                )[:, 0]
                refined_center = model.predict_motion_mean_trajectory(
                    batch["partial"],
                    batch["seen"],
                    base_center,
                    object_type=batch.get("object_type"),
                )
                center = base_center + args.risk_center_refiner_strength * (
                    refined_center - base_center
                )
                if args.risk_center_memory_residual_strength > 0:
                    memory_residual = model.memory_set_center(
                        batch["partial"], batch["seen"], retrieval.values
                    )
                    center = center + args.risk_center_memory_residual_strength * (
                        memory_residual
                        * model.free_mask(batch["seen"], memory_residual.dtype)
                    )
                if args.risk_center_enhanced_residual_strength > 0:
                    enhanced_residual = model.predict_enhanced_memory_center(
                        batch["partial"],
                        batch["seen"],
                        retrieval.values,
                        object_type=batch.get("object_type"),
                    )
                    center = center + args.risk_center_enhanced_residual_strength * (
                        enhanced_residual
                        * model.free_mask(
                            batch["seen"], enhanced_residual.dtype
                        )
                    )
                coverage = model.predict_conditional_modes(
                    batch["partial"],
                    batch["seen"],
                    center,
                    args.num_samples,
                    object_type=batch.get("object_type"),
                    center_copies=args.conditional_center_copies,
                    teacher_modes=0,
                )
                candidates = center[:, None] + args.risk_center_blend_strength * (
                    coverage - center[:, None]
                )
            predictions = model.refine_candidate_set(
                batch["partial"],
                batch["seen"],
                candidates,
                object_type=batch.get("object_type"),
            )
            future_error = torch.linalg.vector_norm(
                predictions[:, :, 8:] - batch["target"][:, None, 8:], dim=-1
            )
            future_valid = batch.get(
                "future_valid",
                torch.ones_like(
                    batch["target"][:, 8:, 0], dtype=torch.bool
                ),
            ).to(future_error.dtype)
            ade = (
                future_error * future_valid[:, None]
            ).sum(dim=-1) / future_valid.sum(dim=-1)[:, None].clamp_min(1)
            reverse_index = torch.flip(
                future_valid.bool(), dims=[1]
            ).float().argmax(dim=1)
            final_index = future_valid.shape[1] - 1 - reverse_index
            fde = future_error.gather(
                2,
                final_index[:, None, None].expand(-1, args.num_samples, 1),
            ).squeeze(-1)
            row_mean = ade.mean(dim=1)
            loss = (
                args.mode_set_mean_weight * row_mean.mean()
                + args.mode_set_oracle_weight
                * ade.topk(
                    min(args.mode_set_oracle_topk, args.num_samples),
                    dim=1,
                    largest=False,
                ).values.mean()
                + args.mode_set_fde_weight
                * fde.topk(
                    min(args.mode_set_fde_topk, args.num_samples),
                    dim=1,
                    largest=False,
                ).values.mean()
                + args.mode_set_mean_fde_weight * fde.mean()
            )
            if args.mode_set_cvar_weight > 0:
                cvar_count = max(
                    1,
                    int(round(args.mode_set_cvar_fraction * row_mean.shape[0])),
                )
                loss = loss + args.mode_set_cvar_weight * row_mean.topk(
                    cvar_count, largest=True
                ).values.mean()
        scaler.scale(loss / args.grad_accum_steps).backward()
        if batch_index % args.grad_accum_steps == 0 or batch_index == len(loader):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def train_risk_center_memory_residual_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Fit a bounded retrieval residual around the frozen low-risk centre."""
    if memory is None or not hasattr(model, "external_center"):
        raise RuntimeError("risk-centre residual training requires memory")
    model.eval()
    model.memory_set_center.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train centre retrieval residual", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            with torch.no_grad():
                retrieval = memory.retrieve(
                    batch["key"], query_indices=batch["index"]
                )
                retrieval = adapt_retrieval(retrieval, batch, args)
                base_center = external_center_trajectories(
                    model.external_center,
                    batch,
                    args.external_center_kind,
                    1,
                    False,
                )[:, 0]
                refined_center = model.predict_motion_mean_trajectory(
                    batch["partial"],
                    batch["seen"],
                    base_center,
                    object_type=batch.get("object_type"),
                )
                center = base_center + args.risk_center_refiner_strength * (
                    refined_center - base_center
                )
            residual = model.memory_set_center(
                batch["partial"], batch["seen"], retrieval.values
            )
            prediction = center + args.risk_center_memory_residual_strength * (
                residual * model.free_mask(batch["seen"], residual.dtype)
            )
            prediction = clamp_observed(
                prediction, batch["target"], batch["seen"]
            )
            future_error = torch.linalg.vector_norm(
                prediction[:, 8:] - batch["target"][:, 8:], dim=-1
            )
            future_valid = batch.get(
                "future_valid",
                torch.ones_like(
                    batch["target"][:, 8:, 0], dtype=torch.bool
                ),
            ).to(future_error.dtype)
            step_weight = torch.ones_like(future_error)
            step_weight[:, -1] = args.mean_head_endpoint_weight
            row_loss = (
                future_error * future_valid * step_weight
            ).sum(dim=-1) / (
                future_valid * step_weight
            ).sum(dim=-1).clamp_min(1)
            loss = row_loss.mean()
            if args.mean_head_cvar_weight > 0:
                cvar_count = max(
                    1,
                    int(round(args.mean_head_cvar_fraction * row_loss.shape[0])),
                )
                loss = loss + args.mean_head_cvar_weight * row_loss.topk(
                    cvar_count, largest=True
                ).values.mean()
        scaler.scale(loss / args.grad_accum_steps).backward()
        if batch_index % args.grad_accum_steps == 0 or batch_index == len(loader):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def train_enhanced_memory_center_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Fit a high-capacity residual on top of the frozen retrieval centre."""
    if memory is None or not hasattr(model, "external_center"):
        raise RuntimeError("enhanced centre training requires memory")
    model.eval()
    model.enhanced_memory_center.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train enhanced memory centre", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            with torch.no_grad():
                retrieval = memory.retrieve(
                    batch["key"], query_indices=batch["index"]
                )
                retrieval = adapt_retrieval(retrieval, batch, args)
                base_center = external_center_trajectories(
                    model.external_center,
                    batch,
                    args.external_center_kind,
                    1,
                    False,
                )[:, 0]
                refined_center = model.predict_motion_mean_trajectory(
                    batch["partial"],
                    batch["seen"],
                    base_center,
                    object_type=batch.get("object_type"),
                )
                center = base_center + args.risk_center_refiner_strength * (
                    refined_center - base_center
                )
                if args.risk_center_memory_residual_strength > 0:
                    center = center + args.risk_center_memory_residual_strength * (
                        model.memory_set_center(
                            batch["partial"], batch["seen"], retrieval.values
                        )
                        * model.free_mask(batch["seen"], center.dtype)
                    )
            residual = model.predict_enhanced_memory_center(
                batch["partial"],
                batch["seen"],
                retrieval.values,
                object_type=batch.get("object_type"),
            )
            prediction = center + args.risk_center_enhanced_residual_strength * (
                residual * model.free_mask(batch["seen"], residual.dtype)
            )
            prediction = clamp_observed(
                prediction, batch["target"], batch["seen"]
            )
            future_error = torch.linalg.vector_norm(
                prediction[:, 8:] - batch["target"][:, 8:], dim=-1
            )
            future_valid = batch.get(
                "future_valid",
                torch.ones_like(
                    batch["target"][:, 8:, 0], dtype=torch.bool
                ),
            ).to(future_error.dtype)
            step_weight = torch.ones_like(future_error)
            step_weight[:, -1] = args.mean_head_endpoint_weight
            row_loss = (
                future_error * future_valid * step_weight
            ).sum(dim=-1) / (
                future_valid * step_weight
            ).sum(dim=-1).clamp_min(1)
            loss = row_loss.mean()
            if args.mean_head_cvar_weight > 0:
                cvar_count = max(
                    1,
                    int(round(args.mean_head_cvar_fraction * row_loss.shape[0])),
                )
                loss = loss + args.mean_head_cvar_weight * row_loss.topk(
                    cvar_count, largest=True
                ).values.mean()
        scaler.scale(loss / args.grad_accum_steps).backward()
        if batch_index % args.grad_accum_steps == 0 or batch_index == len(loader):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def train_adaptive_risk_gate_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Fit a query-only gate between low-risk and multimodal predictions."""
    if memory is None or not hasattr(model, "external_center"):
        raise RuntimeError("adaptive risk-gate training requires memory and center")
    model.eval()
    if args.candidate_aware_risk_gate or args.dual_risk_gate:
        model.set_candidate_ranker.train()
    if args.memory_aware_risk_gate:
        model.retrieval_projection_gate.train()
    if (
        not args.candidate_aware_risk_gate
        and not args.memory_aware_risk_gate
    ) or args.dual_risk_gate:
        model.coverage_risk_gate.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train adaptive risk gate", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            with torch.no_grad():
                retrieval = memory.retrieve(
                    batch["key"], query_indices=batch["index"]
                )
                retrieval = adapt_retrieval(retrieval, batch, args)
                base_center = external_center_trajectories(
                    model.external_center,
                    batch,
                    args.external_center_kind,
                    1,
                    False,
                )[:, 0]
                refined_center = model.predict_motion_mean_trajectory(
                    batch["partial"],
                    batch["seen"],
                    base_center,
                    object_type=batch.get("object_type"),
                )
                center = base_center + args.risk_center_refiner_strength * (
                    refined_center - base_center
                )
                if args.risk_center_memory_residual_strength > 0:
                    memory_residual = model.memory_set_center(
                        batch["partial"], batch["seen"], retrieval.values
                    )
                    center = center + args.risk_center_memory_residual_strength * (
                        memory_residual
                        * model.free_mask(batch["seen"], memory_residual.dtype)
                    )
                if args.risk_center_enhanced_residual_strength > 0:
                    enhanced_residual = model.predict_enhanced_memory_center(
                        batch["partial"],
                        batch["seen"],
                        retrieval.values,
                        object_type=batch.get("object_type"),
                    )
                    center = center + args.risk_center_enhanced_residual_strength * (
                        enhanced_residual
                        * model.free_mask(
                            batch["seen"], enhanced_residual.dtype
                        )
                    )
                coverage = model.predict_conditional_modes(
                    batch["partial"],
                    batch["seen"],
                    center,
                    args.num_samples,
                    object_type=batch.get("object_type"),
                    center_copies=args.conditional_center_copies,
                    teacher_modes=0,
                    split_decoder=args.conditional_split_decoder,
                    coverage_modes=args.conditional_coverage_modes,
                )
            gate_rank_scores = None
            if args.dual_risk_gate:
                raw_gate = model.predict_coverage_risk_gate(
                    batch["partial"], batch["seen"], args.num_samples
                )
                gate_rank_scores = model.select_set_reference_logits(
                    batch["partial"],
                    batch["seen"],
                    coverage,
                    object_type=batch.get("object_type"),
                )
            elif args.memory_aware_risk_gate:
                raw_gate = model.predict_retrieval_projection_gate(
                    batch["partial"],
                    batch["seen"],
                    center,
                    coverage,
                    retrieval.values,
                    retrieval.weights,
                    object_type=batch.get("object_type"),
                )
            elif args.candidate_aware_risk_gate:
                raw_gate = torch.sigmoid(model.select_set_reference_logits(
                    batch["partial"],
                    batch["seen"],
                    coverage,
                    object_type=batch.get("object_type"),
                ))
            else:
                raw_gate = model.predict_coverage_risk_gate(
                    batch["partial"], batch["seen"], args.num_samples
                )
            soft_gate_for_supervision = raw_gate
            projection_loss = torch.zeros((), device=device)
            if args.adaptive_risk_gate_projection_weight > 0:
                coverage_start = min(
                    max(int(args.conditional_center_copies), 0),
                    args.num_samples - 1,
                )
                with torch.no_grad():
                    branch_residual = (
                        coverage[:, coverage_start:, 8:]
                        - center[:, None, 8:]
                    )
                    target_residual = (
                        batch["target"][:, None, 8:]
                        - center[:, None, 8:]
                    )
                    projection_valid = batch.get(
                        "future_valid",
                        torch.ones_like(
                            batch["target"][:, 8:, 0], dtype=torch.bool
                        ),
                    ).to(branch_residual.dtype)[:, None, :, None]
                    numerator = (
                        branch_residual * target_residual * projection_valid
                    ).sum(dim=(-1, -2))
                    denominator = (
                        branch_residual.square() * projection_valid
                    ).sum(dim=(-1, -2)).clamp_min(1e-6)
                    optimal_gate = (numerator / denominator).clamp(0.0, 1.0)
                    if args.adaptive_risk_gate_min < 1.0:
                        optimal_gate = (
                            (optimal_gate - args.adaptive_risk_gate_min)
                            / (1.0 - args.adaptive_risk_gate_min)
                        ).clamp(0.0, 1.0)
                projection_loss = F.smooth_l1_loss(
                    raw_gate[:, coverage_start:], optimal_gate
                )
            if args.adaptive_risk_gate_topk > 0:
                active = min(
                    args.adaptive_risk_gate_topk,
                    args.num_samples - args.conditional_center_copies,
                )
                coverage_gate = raw_gate[:, args.conditional_center_copies:]
                hard_gate = torch.zeros_like(coverage_gate).scatter_(
                    1,
                    coverage_gate.topk(active, dim=1).indices,
                    1.0,
                )
                straight_through = (
                    hard_gate + coverage_gate - coverage_gate.detach()
                )
                raw_gate = raw_gate.clone()
                raw_gate[:, args.conditional_center_copies:] = straight_through
            if args.adaptive_risk_gate_preserve_topk > 0:
                coverage_start = min(
                    max(int(args.conditional_center_copies), 0),
                    args.num_samples - 1,
                )
                preserve_count = min(
                    args.adaptive_risk_gate_preserve_topk,
                    args.num_samples - coverage_start,
                )
                preserve_scores = (
                    gate_rank_scores[:, coverage_start:]
                    if gate_rank_scores is not None
                    else raw_gate[:, coverage_start:]
                )
                preserve_index = preserve_scores.topk(
                    preserve_count, dim=1
                ).indices
                preserve_mask = torch.zeros_like(preserve_scores).scatter_(
                    1, preserve_index, 1.0
                )
                preserve_gate = (
                    preserve_mask
                    + (1.0 - preserve_mask) * preserve_scores
                )
                raw_gate = raw_gate.clone()
                raw_gate[:, coverage_start:] = preserve_gate
            gate = args.adaptive_risk_gate_min + (
                1.0 - args.adaptive_risk_gate_min
            ) * raw_gate
            candidates = center[:, None] + gate[:, :, None, None] * (
                coverage - center[:, None]
            )
            future_error = torch.linalg.vector_norm(
                candidates[:, :, 8:] - batch["target"][:, None, 8:], dim=-1
            )
            future_valid = batch.get(
                "future_valid",
                torch.ones_like(
                    batch["target"][:, 8:, 0], dtype=torch.bool
                ),
            ).to(future_error.dtype)
            ade = (
                future_error * future_valid[:, None]
            ).sum(dim=-1) / future_valid.sum(dim=-1)[:, None].clamp_min(1)
            reverse_index = torch.flip(
                future_valid.bool(), dims=[1]
            ).float().argmax(dim=1)
            final_index = future_valid.shape[1] - 1 - reverse_index
            fde = future_error.gather(
                2,
                final_index[:, None, None].expand(-1, args.num_samples, 1),
            ).squeeze(-1)
            row_mean = ade.mean(dim=1)
            loss = (
                args.mode_set_oracle_weight * ade.min(dim=1).values.mean()
                + args.mode_set_fde_weight * fde.min(dim=1).values.mean()
                + args.mode_set_mean_weight * row_mean.mean()
                + args.mode_set_mean_fde_weight * fde.mean()
                + args.adaptive_risk_gate_projection_weight * projection_loss
            )
            if args.risk_target_hinge_weight > 0:
                scale = float(args.runtime_coordinate_scale)
                row_min_ade = ade.min(dim=1).values
                row_min_fde = fde.min(dim=1).values
                gated_candidates = coverage.clone()
                gated_candidates[:, :, 8:] = candidates[:, :, 8:]
                joint_error = torch.linalg.vector_norm(
                    gated_candidates - batch["target"][:, None], dim=-1
                )
                joint_valid = batch.get(
                    "unknown", torch.ones_like(batch["target"][..., 0])
                ).to(joint_error.dtype)
                joint_ade = (
                    joint_error * joint_valid[:, None]
                ).sum(dim=-1) / joint_valid.sum(
                    dim=-1
                )[:, None].clamp_min(1)
                tail_count = max(
                    1, int(round(0.1 * row_mean.shape[0]))
                )
                soft_miss_rate = torch.sigmoid(
                    (row_min_fde * scale - 2.0)
                    / args.risk_target_mr_temperature
                ).mean() * 100.0
                constrained_metrics = (
                    row_min_ade.mean() * scale,
                    row_min_fde.mean() * scale,
                    row_mean.mean() * scale,
                    soft_miss_rate,
                    row_mean.topk(
                        tail_count, largest=True
                    ).values.mean() * scale,
                    joint_ade.min(dim=1).values.mean() * scale,
                )
                constrained_targets = (
                    args.risk_target_minade,
                    args.risk_target_minfde,
                    args.risk_target_meanade,
                    args.risk_target_mr,
                    args.risk_target_p90,
                    args.risk_target_jointade,
                )
                target_hinge = sum(
                    F.relu(metric / target - 1.0).square()
                    for metric, target in zip(
                        constrained_metrics, constrained_targets
                    )
                )
                loss = loss + args.risk_target_hinge_weight * target_hinge
            if args.adaptive_risk_gate_winner_weight > 0:
                coverage_start = min(
                    max(int(args.conditional_center_copies), 0),
                    args.num_samples - 1,
                )
                if (
                    args.candidate_aware_risk_gate
                    or args.memory_aware_risk_gate
                    or args.dual_risk_gate
                ):
                    raw_branch_error = torch.linalg.vector_norm(
                        coverage[:, coverage_start:, 8:]
                        - batch["target"][:, None, 8:],
                        dim=-1,
                    )
                    branch_ade = (
                        raw_branch_error * future_valid[:, None]
                    ).sum(dim=-1) / future_valid.sum(
                        dim=-1
                    )[:, None].clamp_min(1)
                    branch_fde = raw_branch_error.gather(
                        2,
                        final_index[:, None, None].expand(
                            -1, raw_branch_error.shape[1], 1
                        ),
                    ).squeeze(-1)
                else:
                    branch_ade = ade[:, coverage_start:]
                    branch_fde = fde[:, coverage_start:]
                winner_cost = (
                    branch_ade
                    + args.adaptive_risk_gate_winner_fde_weight * branch_fde
                )
                winner = winner_cost.argmin(dim=1)
                winner_target = torch.zeros_like(branch_ade).scatter_(
                    1, winner[:, None], 1.0
                )
                if gate_rank_scores is not None:
                    winner_loss = F.cross_entropy(
                        gate_rank_scores[:, coverage_start:], winner
                    )
                else:
                    branch_gate = soft_gate_for_supervision[
                        :, coverage_start:
                    ].clamp(
                        min=1e-5, max=1.0 - 1e-5
                    )
                    positive_weight = float(
                        args.adaptive_risk_gate_winner_positive_weight
                    )
                    winner_loss = -(
                        positive_weight
                        * winner_target
                        * branch_gate.log()
                        + (1.0 - winner_target)
                        * (1.0 - branch_gate).log()
                    ).mean()
                loss = (
                    loss
                    + args.adaptive_risk_gate_winner_weight * winner_loss
                )
            if args.mode_set_cvar_weight > 0:
                cvar_count = max(
                    1,
                    int(round(args.mode_set_cvar_fraction * row_mean.shape[0])),
                )
                loss = loss + args.mode_set_cvar_weight * row_mean.topk(
                    cvar_count, largest=True
                ).values.mean()
        scaler.scale(loss / args.grad_accum_steps).backward()
        if batch_index % args.grad_accum_steps == 0 or batch_index == len(loader):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def train_memory_candidate_ranker_epoch(
    model, memory, loader, optimizer, scaler, args, device
) -> float:
    """Fit a target-free-at-inference ranker using train trajectories only."""
    if memory is None or not hasattr(model, "external_center"):
        raise RuntimeError("memory-ranker training requires memory and a centre model")
    model.eval()
    model.memory_candidate_ranker.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train memory ranker", leave=False), start=1
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            with torch.no_grad():
                retrieval = memory.retrieve(
                    batch["key"], query_indices=batch["index"]
                )
                retrieval = adapt_retrieval(retrieval, batch, args)
                base_center = external_center_trajectories(
                    model.external_center,
                    batch,
                    args.external_center_kind,
                    1,
                    False,
                )[:, 0]
                refined_center = model.predict_motion_mean_trajectory(
                    batch["partial"],
                    batch["seen"],
                    base_center,
                    object_type=batch.get("object_type"),
                )
                center = base_center + args.risk_center_refiner_strength * (
                    refined_center - base_center
                )
                if args.memory_ranker_candidate_source == "retrieval":
                    candidates = retrieval.values
                elif args.memory_ranker_candidate_source == "mantra":
                    candidates = mantra_proposals(
                        model, memory, batch, args.num_samples, args=args
                    )
                elif args.memory_ranker_candidate_source == "dual_pool":
                    candidates = model.predict_dual_pool(
                        batch["partial"],
                        batch["seen"],
                        center,
                        args.num_samples,
                        object_type=batch.get("object_type"),
                    )[:, : args.num_samples]
                else:
                    candidates = model.predict_conditional_modes(
                        batch["partial"],
                        batch["seen"],
                        center,
                        args.num_samples,
                        object_type=batch.get("object_type"),
                        center_copies=args.conditional_center_copies,
                        teacher_modes=0,
                    )
                candidate_count = candidates.shape[1]
                future_error = torch.linalg.vector_norm(
                    candidates[:, :, 8:]
                    - batch["target"][:, None, 8:],
                    dim=-1,
                )
                future_valid = batch.get(
                    "future_valid",
                    torch.ones_like(
                        batch["target"][:, 8:, 0], dtype=torch.bool
                    ),
                ).to(future_error.dtype)
                ade = (
                    future_error * future_valid[:, None]
                ).sum(dim=-1) / future_valid.sum(dim=-1)[:, None].clamp_min(1)
                reverse_index = torch.flip(
                    future_valid.bool(), dims=[1]
                ).float().argmax(dim=1)
                final_index = future_valid.shape[1] - 1 - reverse_index
                fde = future_error.gather(
                    2,
                    final_index[:, None, None].expand(
                        -1, candidate_count, 1
                    ),
                ).squeeze(-1)
                target_cost = ade + args.memory_ranker_fde_weight * fde
                target_probability = torch.softmax(
                    -target_cost / args.memory_ranker_target_temperature,
                    dim=1,
                )
            predicted_cost = model.score_candidates_with_memory(
                batch["partial"],
                batch["seen"],
                candidates,
                retrieval.values,
                retrieval.weights,
                object_type=batch.get("object_type"),
            )
            log_probability = torch.log_softmax(
                -predicted_cost / args.memory_ranker_temperature,
                dim=1,
            )
            listwise = -(target_probability * log_probability).sum(dim=1).mean()
            normalized_target = target_cost / target_cost.mean(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            # Ranking is identifiable only up to a per-query offset.  Centre
            # both sides before regression; the previous uncentred target had
            # mean one while the prediction was forced to mean zero, creating
            # an impossible auxiliary objective that weakened top-1 routing.
            normalized_target = normalized_target - normalized_target.mean(
                dim=1, keepdim=True
            )
            normalized_prediction = predicted_cost - predicted_cost.mean(
                dim=1, keepdim=True
            )
            regression = F.smooth_l1_loss(
                normalized_prediction, normalized_target
            )
            best_index = target_cost.argmin(dim=1)
            hard_classification = F.cross_entropy(
                -predicted_cost / args.memory_ranker_temperature,
                best_index,
            )
            best_prediction = predicted_cost.gather(
                1, best_index[:, None]
            )
            pairwise_margin = F.relu(
                0.1 + best_prediction - predicted_cost
            )
            pairwise_mask = torch.ones_like(pairwise_margin)
            pairwise_mask.scatter_(1, best_index[:, None], 0)
            pairwise = (
                pairwise_margin * pairwise_mask
            ).sum() / pairwise_mask.sum().clamp_min(1)
            loss = (
                listwise
                + args.memory_ranker_regression_weight * regression
                + hard_classification
                + 0.5 * pairwise
            )
        scaler.scale(loss / args.grad_accum_steps).backward()
        if batch_index % args.grad_accum_steps == 0 or batch_index == len(loader):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def train_cvae_epoch(model, memory, loader, optimizer, scaler, args, device) -> float:
    model.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(
        tqdm(loader, desc="train cvae", leave=False),
        start=1,
    ):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        with _autocast(device, args.amp):
            context, _ = memory_for_batch(
                memory, batch, args.use_memory, training=True
            )
            prediction, kl = model(
                batch["partial"], batch["seen"], context, batch["target"]
            )
            prediction = clamp_observed(
                prediction, batch["target"], batch["seen"]
            )
            reconstruction = masked_mse(
                prediction, batch["target"], batch["unknown"]
            )
            loss = reconstruction + args.kl_weight * kl.mean()
        scaler.scale(loss / args.grad_accum_steps).backward()
        if (
            batch_index % args.grad_accum_steps == 0
            or batch_index == len(loader)
        ):
            _optimizer_step(model, optimizer, scaler, args)
        total += float(loss.item())
        count += 1
        if args.batch_sleep_ms > 0:
            time_module.sleep(args.batch_sleep_ms / 1000.0)
    return total / max(count, 1)


@torch.no_grad()
def sample_flow(model, memory, batch, args) -> torch.Tensor:
    batch_size = batch["target"].shape[0]
    if args.use_memory:
        result = memory.retrieve(batch["key"], query_indices=None)
        memory_samples = min(
            result.weights.shape[1],
            max(1, round(args.num_samples * args.memory_source_ratio)),
        )
        anchors = memory.sample_anchors(
            result,
            samples=memory_samples,
            cover_topk=True,
            diversity_mode=args.diversity_mode,
            relevance_strength=args.diversity_relevance,
            coverage_samples=args.coverage_samples,
        )
        context = result.context
    else:
        anchors = None
        context = torch.zeros_like(batch["target"])

    sources = []
    strengths = []
    for sample_index in range(args.num_samples):
        uses_anchor = anchors is not None and sample_index < anchors.shape[1]
        anchor = anchors[:, sample_index] if uses_anchor else None
        evaluation_noise = (
            args.source_noise
            if args.eval_source_noise < 0
            else args.eval_source_noise
        )
        sources.append(
            source_state(
                batch["target"],
                batch["seen"],
                anchor,
                evaluation_noise,
            )
        )
        source_strength = args.flow_strength
        if uses_anchor and args.memory_flow_strength >= 0:
            source_strength = args.memory_flow_strength
        strengths.append(source_strength)

    source_tensor = torch.stack(sources, dim=1)
    result_chunks = []
    chunk_size = max(1, min(args.sample_chunk_size, args.num_samples))
    for chunk_start in range(0, args.num_samples, chunk_size):
        chunk_end = min(args.num_samples, chunk_start + chunk_size)
        current_size = chunk_end - chunk_start
        state = source_tensor[:, chunk_start:chunk_end].reshape(
            batch_size * current_size, 20, 2
        )
        target = batch["target"][:, None].expand(
            -1, current_size, -1, -1
        ).reshape(
            batch_size * current_size, 20, 2
        )
        partial = batch["partial"][:, None].expand(
            -1, current_size, -1, -1
        ).reshape(batch_size * current_size, 8, 2)
        seen = batch["seen"][:, None].expand(
            -1, current_size, -1
        ).reshape(batch_size * current_size, 8)
        memory_context = context[:, None].expand(
            -1, current_size, -1, -1
        ).reshape(batch_size * current_size, 20, 2)
        chunk_strengths = torch.tensor(
            strengths[chunk_start:chunk_end],
            device=state.device,
            dtype=state.dtype,
        )
        step_size = (
            chunk_strengths[None]
            .expand(batch_size, -1)
            .reshape(-1)
            / args.flow_steps
        )
        for step in range(args.flow_steps):
            time = step * step_size
            velocity = model(
                state,
                time,
                partial,
                seen,
                memory_context,
            )
            state = state + step_size[:, None, None] * velocity
            state = clamp_observed(state, target, seen)
        reference_blend = float(getattr(args, "reference_blend", 0.0))
        if reference_blend:
            free = torch.ones(
                state.shape[0], state.shape[1], 1,
                device=state.device, dtype=state.dtype,
            )
            free[:, :8] = (~seen).to(state.dtype)[:, :, None]
            state = state + reference_blend * free * (reference - state)
            state = clamp_observed(state, target, seen)
        result_chunks.append(
            state.reshape(batch_size, current_size, 20, 2)
        )
    return torch.cat(result_chunks, dim=1)


@torch.no_grad()
def sample_anchor_flow(model, memory, batch, args) -> torch.Tensor:
    if memory is None:
        raise RuntimeError("anchor flow requires memory")
    batch_size = batch["target"].shape[0]
    retrieval = memory.retrieve(batch["key"], query_indices=None)
    retrieval = adapt_retrieval(retrieval, batch, args)
    references = memory.sample_anchors(
        retrieval,
        samples=args.num_samples,
        cover_topk=True,
        diversity_mode=args.diversity_mode,
        relevance_strength=args.diversity_relevance,
        coverage_samples=args.coverage_samples,
    )
    target_flat = batch["target"][:, None].expand(
        -1, args.num_samples, -1, -1
    ).reshape(batch_size * args.num_samples, 20, 2)
    seen_flat = batch["seen"][:, None].expand(
        -1, args.num_samples, -1
    ).reshape(batch_size * args.num_samples, 8)
    evaluation_noise = (
        args.source_noise if args.eval_source_noise < 0 else args.eval_source_noise
    )
    source_tensor = source_state(
        target_flat,
        seen_flat,
        references.reshape(batch_size * args.num_samples, 20, 2),
        evaluation_noise,
    ).reshape(batch_size, args.num_samples, 20, 2)

    result_chunks = []
    chunk_size = max(1, min(args.sample_chunk_size, args.num_samples))
    for chunk_start in range(0, args.num_samples, chunk_size):
        chunk_end = min(args.num_samples, chunk_start + chunk_size)
        current_size = chunk_end - chunk_start
        state = source_tensor[:, chunk_start:chunk_end].reshape(
            batch_size * current_size, 20, 2
        )
        target = batch["target"][:, None].expand(
            -1, current_size, -1, -1
        ).reshape(batch_size * current_size, 20, 2)
        partial = batch["partial"][:, None].expand(
            -1, current_size, -1, -1
        ).reshape(batch_size * current_size, 8, 2)
        seen = batch["seen"][:, None].expand(
            -1, current_size, -1
        ).reshape(batch_size * current_size, 8)
        reference = references[:, chunk_start:chunk_end].reshape(
            batch_size * current_size, 20, 2
        )
        strength = (
            args.memory_flow_strength
            if args.memory_flow_strength >= 0
            else args.flow_strength
        )
        step_size = strength / args.flow_steps
        for step in range(args.flow_steps):
            time = torch.full(
                (state.shape[0],),
                step / args.flow_steps,
                device=state.device,
                dtype=state.dtype,
            )
            velocity = model(state, time, partial, seen, reference)
            state = state + step_size * velocity
            state = clamp_observed(state, target, seen)
        result_chunks.append(
            state.reshape(batch_size, current_size, 20, 2)
        )
    return torch.cat(result_chunks, dim=1)


@torch.no_grad()
def sample_variational_flow(model, memory, batch, args) -> torch.Tensor:
    if memory is None:
        raise RuntimeError("variational flow requires memory")
    batch_size = batch["target"].shape[0]
    retrieval = memory.retrieve(batch["key"], query_indices=None)
    retrieval = adapt_retrieval(retrieval, batch, args)
    retrieval = rerank_retrieval(model, retrieval, batch, args)
    references = memory.sample_anchors(
        retrieval,
        samples=args.num_samples,
        cover_topk=True,
        diversity_mode=args.diversity_mode,
        relevance_strength=args.diversity_relevance,
        coverage_samples=args.coverage_samples,
    )
    risk_samples = int(
        round(getattr(args, "risk_branch_fraction", 0.0) * args.num_samples)
    )
    risk_samples = max(0, min(args.num_samples, risk_samples))
    risk_start = args.num_samples - risk_samples
    if risk_samples:
        retrieval_center = (
            retrieval.weights[..., None, None] * retrieval.values
        ).sum(dim=1, keepdim=True)
        references[:, risk_start:] = retrieval_center + (
            getattr(args, "risk_reference_shrinkage", 0.0)
            * (references[:, risk_start:] - retrieval_center)
        )
    residuals, latents = model.sample_prior_residual(
        batch["partial"], batch["seen"], references, sampling=args.prior_sampling
    )
    if risk_samples:
        risk_residuals, risk_latents = model.sample_prior_residual(
            batch["partial"],
            batch["seen"],
            references[:, risk_start:],
            sampling="mean",
        )
        residuals[:, risk_start:] = risk_residuals
        latents[:, risk_start:] = risk_latents
    if args.residual_scope == "history":
        residuals[:, :, 8:] = 0
    elif args.residual_scope == "future":
        residuals[:, :, :8] = 0
    anchor_samples = int(round(args.anchor_sample_fraction * args.num_samples))
    anchor_samples = max(0, min(args.num_samples, anchor_samples))
    if anchor_samples:
        residuals[:, :anchor_samples] = 0
        latents[:, :anchor_samples] = 0
    target_flat = batch["target"][:, None].expand(
        -1, args.num_samples, -1, -1
    ).reshape(batch_size * args.num_samples, 20, 2)
    seen_flat = batch["seen"][:, None].expand(
        -1, args.num_samples, -1
    ).reshape(batch_size * args.num_samples, 8)
    evaluation_noise = (
        args.source_noise if args.eval_source_noise < 0 else args.eval_source_noise
    )
    source_tensor = source_state(
        target_flat,
        seen_flat,
        (references + args.residual_scale * residuals).reshape(
            batch_size * args.num_samples, 20, 2
        ),
        evaluation_noise,
    ).reshape(batch_size, args.num_samples, 20, 2)

    result_chunks = []
    flow_trace_chunks = []
    chunk_size = max(1, min(args.sample_chunk_size, args.num_samples))
    for chunk_start in range(0, args.num_samples, chunk_size):
        chunk_end = min(args.num_samples, chunk_start + chunk_size)
        current_size = chunk_end - chunk_start
        state = source_tensor[:, chunk_start:chunk_end].reshape(
            batch_size * current_size, 20, 2
        )
        chunk_trace = (
            [state.reshape(batch_size, current_size, 20, 2).detach().cpu()]
            if args.dump_flow_trace else None
        )
        target = batch["target"][:, None].expand(
            -1, current_size, -1, -1
        ).reshape(batch_size * current_size, 20, 2)
        partial = batch["partial"][:, None].expand(
            -1, current_size, -1, -1
        ).reshape(batch_size * current_size, 8, 2)
        seen = batch["seen"][:, None].expand(
            -1, current_size, -1
        ).reshape(batch_size * current_size, 8)
        reference = references[:, chunk_start:chunk_end].reshape(
            batch_size * current_size, 20, 2
        )
        latent = latents[:, chunk_start:chunk_end].reshape(
            batch_size * current_size, -1
        )
        strength = (
            args.memory_flow_strength
            if args.memory_flow_strength >= 0
            else args.flow_strength
        )
        sample_strengths = torch.full(
            (args.num_samples,),
            strength,
            device=state.device,
            dtype=state.dtype,
        )
        if anchor_samples and args.anchor_flow_strength >= 0:
            sample_strengths[:anchor_samples] = args.anchor_flow_strength
        step_size = (
            sample_strengths[chunk_start:chunk_end]
            .repeat(batch_size)
            / args.flow_steps
        )
        for step in range(args.flow_steps):
            time = torch.full(
                (state.shape[0],),
                step / args.flow_steps,
                device=state.device,
                dtype=state.dtype,
            )
            velocity = model(
                state, time, partial, seen, reference, latent
            )
            state = state + step_size[:, None, None] * velocity
            state = clamp_observed(state, target, seen)
            if chunk_trace is not None:
                chunk_trace.append(
                    state.reshape(batch_size, current_size, 20, 2)
                    .detach().cpu()
                )
        reference_blend = float(getattr(args, "reference_blend", 0.0))
        if reference_blend:
            free = torch.ones(
                state.shape[0], state.shape[1], 1,
                device=state.device, dtype=state.dtype,
            )
            free[:, :8] = (~seen).to(state.dtype)[:, :, None]
            state = state + reference_blend * free * (reference - state)
            state = clamp_observed(state, target, seen)
        result_chunks.append(
            state.reshape(batch_size, current_size, 20, 2)
        )
        if chunk_trace is not None:
            flow_trace_chunks.append(torch.stack(chunk_trace, dim=2))
    predictions = torch.cat(result_chunks, dim=1)
    if flow_trace_chunks:
        model._last_flow_trace = {
            "states": torch.cat(flow_trace_chunks, dim=1),
            "references": references.detach().cpu(),
            "latents": latents.detach().cpu(),
        }
    memofm_predictions = predictions.clone()
    memofm_history = predictions[:, :, :8].clone()
    mantra_endpoint = None
    if args.mantra_source:
        predictions = mantra_proposals(
            model, memory, batch, args.num_samples, args=args
        )
        mantra_endpoint = predictions[:, :, -1].clone()
        references = predictions.clone()
    candidate_refiner_strength = float(args.candidate_refiner_strength)
    if candidate_refiner_strength:
        partial_refiner = batch["partial"][:, None].expand(
            -1, args.num_samples, -1, -1
        ).reshape(batch_size * args.num_samples, 8, 2)
        seen_refiner = batch["seen"][:, None].expand(
            -1, args.num_samples, -1
        ).reshape(batch_size * args.num_samples, 8)
        if args.set_aware_candidate_refiner:
            candidate_refined = model.refine_candidate_set(
                batch["partial"], batch["seen"], references,
                object_type=batch.get("object_type"),
            ).reshape(batch_size * args.num_samples, 20, 2)
        else:
            candidate_refined = model.refine_candidate_trajectory(
                partial_refiner,
                seen_refiner,
                references.reshape(batch_size * args.num_samples, 20, 2),
            )
        candidate_refined = clamp_observed(
            candidate_refined,
            target_flat,
            seen_refiner,
        ).reshape(batch_size, args.num_samples, 20, 2)
        predictions = predictions + candidate_refiner_strength * (
            candidate_refined - predictions
        )
    mode_strength = float(args.mode_head_refine_strength)
    mode_samples = int(round(args.mode_head_refine_fraction * args.num_samples))
    mode_samples = max(0, min(args.num_samples, mode_samples))
    if mode_strength and mode_samples:
        partial_mode = batch["partial"][:, None].expand(
            -1, mode_samples, -1, -1
        ).reshape(batch_size * mode_samples, 8, 2)
        seen_mode = batch["seen"][:, None].expand(
            -1, mode_samples, -1
        ).reshape(batch_size * mode_samples, 8)
        mode_references = model.predict_mode_trajectory(
            partial_mode,
            seen_mode,
            references[:, :mode_samples].reshape(
                batch_size * mode_samples, 20, 2
            ),
        )
        mode_targets = batch["target"][:, None].expand(
            -1, mode_samples, -1, -1
        ).reshape(batch_size * mode_samples, 20, 2)
        mode_references = clamp_observed(
            mode_references, mode_targets, seen_mode
        ).reshape(batch_size, mode_samples, 20, 2)
        predictions[:, :mode_samples] = (
            predictions[:, :mode_samples]
            + mode_strength
            * (mode_references - predictions[:, :mode_samples])
        )
    refine_strength = float(args.mean_head_refine_strength)
    refine_samples = int(round(args.mean_head_refine_fraction * args.num_samples))
    refine_samples = max(0, min(args.num_samples, refine_samples))
    if refine_strength and refine_samples:
        partial_flat = batch["partial"][:, None].expand(
            -1, args.num_samples, -1, -1
        ).reshape(batch_size * args.num_samples, 8, 2)
        seen_refine_flat = batch["seen"][:, None].expand(
            -1, args.num_samples, -1
        ).reshape(batch_size * args.num_samples, 8)
        refined_references = model.predict_mean_trajectory(
            partial_flat,
            seen_refine_flat,
            references.reshape(batch_size * args.num_samples, 20, 2),
        )
        refined_references = clamp_observed(
            refined_references,
            target_flat,
            seen_refine_flat,
        ).reshape(batch_size, args.num_samples, 20, 2)
        refine_start = args.num_samples - refine_samples
        predictions[:, refine_start:] = (
            predictions[:, refine_start:]
            + refine_strength
            * (
                refined_references[:, refine_start:]
                - predictions[:, refine_start:]
            )
        )
    mean_samples = int(round(args.mean_head_sample_fraction * args.num_samples))
    mean_samples = max(0, min(args.num_samples, mean_samples))
    center_blend_strength = float(
        getattr(args, "mean_head_center_blend_strength", 0.0)
    )
    if mean_samples or center_blend_strength:
        if args.mean_head_selector_preserve and mean_samples < args.num_samples:
            selector_logits = reference_selector_logits(
                model, batch, references, args
            )
            order = selector_logits.argsort(dim=-1, descending=True)
            rows = torch.arange(batch_size, device=predictions.device)[:, None]
            if args.selector_diverse_preserve:
                preserve_count = args.num_samples - mean_samples
                chosen_columns = [selector_logits.argmax(dim=-1)]
                endpoint = references[:, :, -1]
                minimum_distance = torch.linalg.vector_norm(
                    endpoint
                    - endpoint[
                        torch.arange(batch_size, device=endpoint.device),
                        chosen_columns[0],
                    ][:, None],
                    dim=-1,
                )
                relevance = selector_logits - selector_logits.amin(
                    dim=-1, keepdim=True
                )
                relevance = relevance / relevance.amax(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                for _ in range(1, preserve_count):
                    score = (
                        minimum_distance
                        + args.selector_diversity_weight * relevance
                    )
                    for previous in chosen_columns:
                        score[rows[:, 0], previous] = -torch.inf
                    next_column = score.argmax(dim=-1)
                    chosen_columns.append(next_column)
                    next_endpoint = endpoint[
                        torch.arange(batch_size, device=endpoint.device),
                        next_column,
                    ]
                    minimum_distance = torch.minimum(
                        minimum_distance,
                        torch.linalg.vector_norm(
                            endpoint - next_endpoint[:, None], dim=-1
                        ),
                    )
                remaining = selector_logits.argsort(dim=-1, descending=True)
                preserve_order = torch.stack(chosen_columns, dim=-1)
                selected_mask = torch.zeros_like(
                    selector_logits, dtype=torch.bool
                )
                selected_mask.scatter_(1, preserve_order, True)
                remainder_order = remaining.masked_select(
                    ~selected_mask.gather(1, remaining)
                ).reshape(batch_size, mean_samples)
                order = torch.cat([preserve_order, remainder_order], dim=-1)
            predictions = predictions[rows, order]
            references = references[rows, order]
            if mantra_endpoint is not None:
                mantra_endpoint = mantra_endpoint[rows, order]
            memofm_history = memofm_history[rows, order]
        mean_reference = (
            references.mean(dim=1)
            if args.mean_head_proposal_center and args.mantra_source
            else (
                retrieval.weights[..., None, None] * retrieval.values
            ).sum(dim=1)
        )
        if args.database_center_head:
            mean_prediction = model.predict_database_center(
                batch["partial"],
                batch["seen"],
                references,
                object_type=batch.get("object_type"),
            )
        elif args.set_aware_center_head:
            mean_prediction = model.predict_set_aware_center(
                batch["partial"],
                batch["seen"],
                references,
                object_type=batch.get("object_type"),
            )
        elif args.memory_set_mean_head:
            mean_prediction = model.predict_memory_set_center(
                batch["partial"], batch["seen"], references
            )
        else:
            if args.motion_mean_head:
                mean_prediction = model.predict_motion_mean_trajectory(
                    batch["partial"], batch["seen"], mean_reference,
                    object_type=batch.get("object_type"),
                )
            else:
                mean_prediction = model.predict_mean_trajectory(
                    batch["partial"], batch["seen"], mean_reference
                )
        mean_prediction = clamp_observed(
            mean_prediction, batch["target"], batch["seen"]
        )
        center_candidates = mean_prediction[:, None].expand(
            -1, args.num_samples, -1, -1
        ).clone()
        if args.endpoint_conditioned_center and mantra_endpoint is not None:
            endpoint_delta = (
                mantra_endpoint - mean_prediction[:, None, -1]
            )
            future_ramp = torch.linspace(
                1.0 / 12.0,
                1.0,
                12,
                device=predictions.device,
                dtype=predictions.dtype,
            ).pow(args.endpoint_warp_power)
            center_candidates[:, :, 8:] = (
                center_candidates[:, :, 8:]
                + future_ramp[None, None, :, None]
                * endpoint_delta[:, :, None]
            )
            coverage_samples = min(
                args.num_samples, int(args.endpoint_warp_coverage_samples)
            )
            if coverage_samples:
                coverage_ramp = torch.linspace(
                    1.0 / 12.0,
                    1.0,
                    12,
                    device=predictions.device,
                    dtype=predictions.dtype,
                ).pow(args.endpoint_warp_coverage_power)
                center_candidates[:, :coverage_samples, 8:] = (
                    mean_prediction[:, None, 8:]
                    + coverage_ramp[None, None, :, None]
                    * endpoint_delta[:, :coverage_samples, None]
                )
        if center_blend_strength:
            # Preserve the complete multimodal candidate set and move only
            # its free coordinates toward the conditional low-risk centre.
            # Observed history and retrieval endpoints are restored below,
            # so this gate cannot collapse endpoint coverage or repair input.
            predictions = predictions + center_blend_strength * (
                center_candidates - predictions
            )
        if mean_samples:
            predictions[:, -mean_samples:] = center_candidates[:, -mean_samples:]
    if args.mantra_source:
        predictions[:, :, :8] = memofm_history
        if args.mantra_preserve_endpoint:
            predictions[:, :, -1] = mantra_endpoint
    adaptive_strength = float(args.adaptive_selector_refine_strength)
    preserve_samples = min(
        args.num_samples, int(args.adaptive_selector_preserve_samples)
    )
    if adaptive_strength and preserve_samples < args.num_samples:
        selector_logits = reference_selector_logits(
            model, batch, references, args
        )
        preserve = selector_logits.topk(
            preserve_samples, dim=-1, largest=True
        ).indices
        preserve_mask = torch.zeros_like(selector_logits, dtype=torch.bool)
        preserve_mask.scatter_(1, preserve, True)
        partial_flat = batch["partial"][:, None].expand(
            -1, args.num_samples, -1, -1
        ).reshape(batch_size * args.num_samples, 8, 2)
        seen_flat = batch["seen"][:, None].expand(
            -1, args.num_samples, -1
        ).reshape(batch_size * args.num_samples, 8)
        refined = model.predict_mean_trajectory(
            partial_flat,
            seen_flat,
            references.reshape(batch_size * args.num_samples, 20, 2),
        ).reshape(batch_size, args.num_samples, 20, 2)
        gate = (~preserve_mask).to(predictions.dtype)[:, :, None, None]
        predictions = predictions + adaptive_strength * gate * (
            refined - predictions
        )
        predictions[:, :, :8] = memofm_history
    hermite_strength = float(args.endpoint_hermite_strength)
    if hermite_strength:
        start = predictions[:, :, 7]
        initial_velocity = predictions[:, :, 7] - predictions[:, :, 6]
        endpoint = (
            mantra_endpoint
            if mantra_endpoint is not None
            else references[:, :, -1]
        )
        terminal_velocity = references[:, :, -1] - references[:, :, -2]
        time = torch.linspace(
            1.0 / 12.0,
            1.0,
            12,
            device=predictions.device,
            dtype=predictions.dtype,
        )[None, None, :, None]
        h00 = 2 * time.pow(3) - 3 * time.pow(2) + 1
        h10 = time.pow(3) - 2 * time.pow(2) + time
        h01 = -2 * time.pow(3) + 3 * time.pow(2)
        h11 = time.pow(3) - time.pow(2)
        hermite_future = (
            h00 * start[:, :, None]
            + h10 * 12.0 * initial_velocity[:, :, None]
            + h01 * endpoint[:, :, None]
            + h11 * 12.0 * terminal_velocity[:, :, None]
        )
        predictions[:, :, 8:] = (
            predictions[:, :, 8:]
            + hermite_strength
            * (hermite_future - predictions[:, :, 8:])
        )
    replication_preserve = min(
        args.num_samples, int(args.candidate_replication_preserve_samples)
    )
    if 0 < replication_preserve < args.num_samples:
        source_count = max(
            1,
            min(replication_preserve, int(args.candidate_replication_source_samples)),
        )
        remainder = args.num_samples - replication_preserve
        source_index = torch.arange(
            remainder, device=predictions.device
        ).remainder(source_count)
        target_endpoint = predictions[:, replication_preserve:, -1].clone()
        replicated = predictions[:, source_index].clone()
        endpoint_delta = target_endpoint - replicated[:, :, -1]
        replication_ramp = torch.linspace(
            1.0 / 12.0,
            1.0,
            12,
            device=predictions.device,
            dtype=predictions.dtype,
        ).pow(args.candidate_replication_warp_power)
        replicated[:, :, 8:] = (
            replicated[:, :, 8:]
            + replication_ramp[None, None, :, None]
            * endpoint_delta[:, :, None]
        )
        predictions[:, replication_preserve:] = replicated
    final_refiner_strength = float(args.candidate_refiner_final_strength)
    final_refiner_samples = min(
        args.num_samples, int(args.candidate_refiner_final_samples)
    )
    if final_refiner_strength and final_refiner_samples:
        final_partial = batch["partial"][:, None].expand(
            -1, final_refiner_samples, -1, -1
        ).reshape(batch_size * final_refiner_samples, 8, 2)
        final_seen = batch["seen"][:, None].expand(
            -1, final_refiner_samples, -1
        ).reshape(batch_size * final_refiner_samples, 8)
        final_refiner = (
            model.refine_coverage_candidate_trajectory
            if args.coverage_refiner_checkpoint
            else model.refine_candidate_trajectory
        )
        final_refined = final_refiner(
            final_partial,
            final_seen,
            references[:, :final_refiner_samples].reshape(
                batch_size * final_refiner_samples, 20, 2
            ),
        ).reshape(batch_size, final_refiner_samples, 20, 2)
        predictions[:, :final_refiner_samples] = (
            predictions[:, :final_refiner_samples]
            + final_refiner_strength
            * (
                final_refined
                - predictions[:, :final_refiner_samples]
            )
        )
        predictions[:, :, :8] = memofm_history
    if args.mantra_source and args.mantra_preserve_endpoint:
        # Endpoint repair is intentionally last: adaptive refinement may
        # reshape high-risk paths, but it must not destroy the calibrated
        # destination coverage supplied by the retrieval proposal set.
        predictions[:, :, -1] = mantra_endpoint
    hybrid_flow_samples = min(args.num_samples, int(args.hybrid_flow_samples))
    if args.mantra_source and hybrid_flow_samples:
        # Keep a validation-selected subset of native flow hypotheses in the
        # fixed-size proposal set. Retrieval supplies endpoint coverage while
        # native flow contributes complementary low-FDE trajectories.
        predictions[:, :hybrid_flow_samples] = memofm_predictions[
            :, :hybrid_flow_samples
        ]
    external_center_samples = min(
        args.num_samples, int(args.external_center_samples)
    )
    if external_center_samples:
        if not hasattr(model, "external_center"):
            raise RuntimeError(
                "external center samples requested without a loaded checkpoint"
            )
        selector_preserve = args.num_samples - external_center_samples
        if args.external_center_selector_order and selector_preserve > 0:
            selector_logits = reference_selector_logits(
                model, batch, predictions, args
            )
            order = selector_logits.argsort(dim=-1, descending=True)
            rows = torch.arange(batch_size, device=predictions.device)[:, None]
            predictions = predictions[rows, order]
        center = external_center_trajectories(
            model.external_center,
            batch,
            args.external_center_kind,
            external_center_samples,
            args.external_center_stochastic,
        )
        if args.external_center_secondary_blend > 0:
            if not hasattr(model, "external_center_secondary"):
                raise RuntimeError(
                    "secondary external center blend requested without checkpoint"
                )
            secondary = external_center_trajectories(
                model.external_center_secondary,
                batch,
                args.external_center_secondary_kind,
                external_center_samples,
                args.external_center_secondary_stochastic,
            )
            blend = float(args.external_center_secondary_blend)
            center = center + blend * (secondary - center)
        refiner_strength = float(args.external_center_refiner_strength)
        if refiner_strength > 0:
            batch_size = center.shape[0]
            center_count = center.shape[1]
            partial = batch["partial"][:, None].expand(
                -1, center_count, -1, -1
            ).reshape(batch_size * center_count, 8, 2)
            seen = batch["seen"][:, None].expand(
                -1, center_count, -1
            ).reshape(batch_size * center_count, 8)
            refined = model.predict_motion_mean_trajectory(
                partial,
                seen,
                center.reshape(batch_size * center_count, 20, 2),
                object_type=(
                    batch["object_type"][:, None].expand(
                        -1, center_count
                    ).reshape(-1)
                    if "object_type" in batch else None
                ),
            ).reshape_as(center)
            center = center + refiner_strength * (refined - center)
        start = args.num_samples - external_center_samples
        center_future = center[:, :, 8:].clone()
        proposal_blend = float(args.external_center_proposal_blend)
        if proposal_blend > 0:
            # Inject proposal-specific waypoint structure around the trained
            # low-risk centre.  This preserves one shared conditional centre
            # while allowing the fixed candidate budget to cover distinct
            # database modes instead of differing only at the last point.
            proposal_future = predictions[:, start:, 8:]
            center_future = center_future + proposal_blend * (
                proposal_future - center_future
            )
        if args.external_center_endpoint_warp_power > 0:
            endpoint_delta = (
                predictions[:, start:, -1] - center_future[:, :, -1]
            )
            unwarped_center_future = center_future.clone()
            ramp = torch.linspace(
                1.0 / 12.0, 1.0, 12,
                device=predictions.device, dtype=predictions.dtype,
            ).pow(args.external_center_endpoint_warp_power)
            center_future = (
                center_future
                + ramp[None, None, :, None] * endpoint_delta[:, :, None]
            )
            coverage_samples = min(
                external_center_samples,
                int(args.external_center_coverage_samples),
            )
            if coverage_samples:
                coverage_ramp = torch.linspace(
                    1.0 / 12.0, 1.0, 12,
                    device=predictions.device, dtype=predictions.dtype,
                ).pow(args.external_center_coverage_warp_power)
                if args.external_center_diverse_coverage:
                    proposal_future = predictions[:, start:, 8:]
                    if args.external_center_coverage_mode == "endpoint":
                        coverage_feature = proposal_future[:, :, -1]
                    elif args.external_center_coverage_mode == "waypoints":
                        coverage_feature = proposal_future[:, :, [3, 7, 11]].reshape(
                            batch_size, external_center_samples, -1
                        )
                    else:
                        coverage_feature = proposal_future.reshape(
                            batch_size, external_center_samples, -1
                        )
                    rows = torch.arange(batch_size, device=predictions.device)
                    chosen = [torch.zeros(
                        batch_size, dtype=torch.long, device=predictions.device
                    )]
                    minimum_distance = torch.linalg.vector_norm(
                        coverage_feature - coverage_feature[:, :1], dim=-1
                    )
                    for _ in range(1, coverage_samples):
                        score = minimum_distance.clone()
                        for previous in chosen:
                            score[rows, previous] = -torch.inf
                        next_column = score.argmax(dim=-1)
                        chosen.append(next_column)
                        next_feature = coverage_feature[rows, next_column]
                        minimum_distance = torch.minimum(
                            minimum_distance,
                            torch.linalg.vector_norm(
                                coverage_feature - next_feature[:, None], dim=-1
                            ),
                        )
                    coverage_index = torch.stack(chosen, dim=1)
                else:
                    coverage_index = torch.arange(
                        coverage_samples, device=predictions.device
                    )[None].expand(batch_size, -1)
                    rows = torch.arange(batch_size, device=predictions.device)
                center_future[rows[:, None], coverage_index] = (
                    unwarped_center_future[rows[:, None], coverage_index]
                    + coverage_ramp[None, None, :, None]
                    * endpoint_delta[rows[:, None], coverage_index, None]
                )
        predictions[:, start:, 8:] = center_future
    mode_strength = float(args.conditional_mode_strength)
    if mode_strength > 0:
        preserved_history = predictions[:, :, :8].clone()
        center_prediction = predictions.mean(dim=1)
        if args.risk_center_checkpoint:
            base_mode_center = external_center_trajectories(
                model.external_center,
                batch,
                args.external_center_kind,
                1,
                False,
            )[:, 0]
            refined_mode_center = model.predict_motion_mean_trajectory(
                batch["partial"],
                batch["seen"],
                base_mode_center,
                object_type=batch.get("object_type"),
            )
            center_prediction = base_mode_center + args.risk_center_refiner_strength * (
                refined_mode_center - base_mode_center
            )
            if args.risk_center_memory_residual_strength > 0:
                mode_memory_residual = model.memory_set_center(
                    batch["partial"], batch["seen"], retrieval.values
                )
                center_prediction = center_prediction + (
                    args.risk_center_memory_residual_strength
                    * mode_memory_residual
                    * model.free_mask(
                        batch["seen"], mode_memory_residual.dtype
                    )
                )
            if args.risk_center_enhanced_residual_strength > 0:
                enhanced_residual = model.predict_enhanced_memory_center(
                    batch["partial"], batch["seen"], retrieval.values,
                    object_type=batch.get("object_type"),
                )
                center_prediction = center_prediction + (
                    args.risk_center_enhanced_residual_strength
                    * enhanced_residual
                    * model.free_mask(
                        batch["seen"], enhanced_residual.dtype
                    )
                )
        pool_center_copies = (
            1 if (
                args.conditional_selected_modes > 0
                or args.conditional_mixer_modes > 0
            )
            else args.conditional_center_copies
        )
        mode_predictions = model.predict_conditional_modes(
            batch["partial"], batch["seen"], center_prediction,
            args.num_samples, object_type=batch.get("object_type"),
            center_copies=pool_center_copies,
            teacher_modes=args.conditional_teacher_modes,
            split_decoder=args.conditional_split_decoder,
            coverage_modes=args.conditional_coverage_modes,
            memory_values=retrieval.values,
            memory_weights=retrieval.weights,
            memory_conditional=args.memory_conditional_mode,
            memory_retrieval_init_strength=(
                args.memory_conditional_retrieval_init_strength
            ),
        )
        if (
            args.dual_pool_selector
            or args.dual_pool_fixed_mixture
            or args.mode_alignment_gate
            or args.coverage_compression_mixer
            or args.direct_coverage_experts
            or args.retrieval_coverage_modes > 0
        ):
            dual_pool = model.predict_dual_pool(
                batch["partial"],
                batch["seen"],
                center_prediction,
                args.num_samples,
                object_type=batch.get("object_type"),
            )
            if args.retrieval_coverage_modes > 0:
                coverage = dual_pool[:, : args.num_samples]
                start = args.retrieval_coverage_expert_index * args.num_samples
                low_risk = dual_pool[:, start : start + args.num_samples]
                pairwise = torch.linalg.vector_norm(
                    coverage[:, :, None, 8:]
                    - retrieval.values[:, None, :, 8:],
                    dim=-1,
                )
                pairwise_ade = pairwise.mean(dim=-1)
                pairwise_fde = pairwise[:, :, :, -1]
                risk = pairwise_ade + args.retrieval_coverage_fde_weight * (
                    pairwise_fde
                )
                weighted_retrieval = args.retrieval_coverage_score in {
                    "weighted",
                    "mmr_weighted",
                }
                if weighted_retrieval:
                    weights = retrieval.weights / retrieval.weights.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-8)
                    score = (risk * weights[:, None]).sum(dim=-1)
                else:
                    score = risk.min(dim=-1).values
                if args.retrieval_coverage_score.startswith("mmr_"):
                    future = coverage[:, :, 8:]
                    if args.retrieval_coverage_diversity_feature == "endpoint":
                        diversity_feature = future[:, :, -1]
                    elif args.retrieval_coverage_diversity_feature == "waypoints":
                        diversity_feature = future[:, :, ::4].reshape(
                            future.shape[0], future.shape[1], -1
                        )
                    else:
                        diversity_feature = future.reshape(
                            future.shape[0], future.shape[1], -1
                        )
                    candidate_distance = torch.cdist(
                        diversity_feature, diversity_feature
                    )
                    relevance = (score - score.amin(dim=1, keepdim=True)) / (
                        score.amax(dim=1, keepdim=True)
                        - score.amin(dim=1, keepdim=True)
                    ).clamp_min(1e-6)
                    chosen = [relevance.argmin(dim=1)]
                    available = torch.ones_like(relevance, dtype=torch.bool)
                    rows = torch.arange(
                        coverage.shape[0], device=coverage.device
                    )
                    available[rows, chosen[0]] = False
                    for _ in range(1, args.retrieval_coverage_modes):
                        selected = torch.stack(chosen, dim=1)
                        selected_distance = candidate_distance.gather(
                            2,
                            selected[:, None, :].expand(
                                -1, candidate_distance.shape[1], -1
                            ),
                        ).amin(dim=2)
                        selected_distance = selected_distance / (
                            candidate_distance.amax(dim=(1, 2), keepdim=False)
                            .unsqueeze(1)
                            .clamp_min(1e-6)
                        )
                        criterion = relevance - (
                            args.retrieval_coverage_diversity_weight
                            * selected_distance
                        )
                        criterion = criterion.masked_fill(~available, float("inf"))
                        next_index = criterion.argmin(dim=1)
                        chosen.append(next_index)
                        available[rows, next_index] = False
                    coverage_index = torch.stack(chosen, dim=1)
                else:
                    coverage_index = score.topk(
                        args.retrieval_coverage_modes,
                        dim=1,
                        largest=False,
                        sorted=True,
                    ).indices
                low_feature = low_risk[:, :, 8:].reshape(
                    low_risk.shape[0], low_risk.shape[1], -1
                )
                consensus = low_feature.mean(dim=1, keepdim=True)
                low_count = args.num_samples - args.retrieval_coverage_modes
                low_index = torch.linalg.vector_norm(
                    low_feature - consensus, dim=-1
                ).topk(low_count, dim=1, largest=False, sorted=True).indices
                rows = torch.arange(
                    low_risk.shape[0], device=low_risk.device
                )[:, None]
                mode_predictions = torch.cat(
                    [
                        coverage[rows, coverage_index],
                        low_risk[rows, low_index],
                    ],
                    dim=1,
                )
            elif args.direct_coverage_experts:
                start = args.direct_coverage_expert_index * args.num_samples
                low_risk = dual_pool[:, start : start + args.num_samples]
                low_feature = low_risk[:, :, 8:].reshape(
                    low_risk.shape[0], low_risk.shape[1], -1
                )
                consensus = low_feature.mean(dim=1, keepdim=True)
                low_order = torch.linalg.vector_norm(
                    low_feature - consensus, dim=-1
                ).argsort(dim=1)
                rows = torch.arange(
                    low_risk.shape[0], device=low_risk.device
                )[:, None]
                center = center_prediction
                residual, gate = model.memory_conditional_mode_decoder(
                    batch["partial"],
                    batch["seen"],
                    center,
                    retrieval.values,
                    args.direct_coverage_modes,
                    memory_weights=retrieval.weights,
                    object_type=batch.get("object_type"),
                    retrieval_init_strength=(
                        args.direct_coverage_retrieval_strength
                    ),
                )
                experts = center[:, None] + residual * gate * model.free_mask(
                    batch["seen"], residual.dtype
                )[:, None]
                low_count = args.num_samples - args.direct_coverage_modes
                mode_predictions = torch.cat(
                    [experts, low_risk[rows, low_order[:, :low_count]]], dim=1
                )
            elif args.coverage_compression_mixer:
                start = (
                    args.coverage_compression_expert_index * args.num_samples
                )
                coverage = dual_pool[:, : args.num_samples]
                low_risk = dual_pool[:, start : start + args.num_samples]
                compressed = model.mix_conditional_modes(
                    batch["partial"],
                    batch["seen"],
                    coverage,
                    args.coverage_compression_modes,
                    temperature=args.coverage_compression_temperature,
                    object_type=batch.get("object_type"),
                )
                low_feature = low_risk[:, :, 8:].reshape(
                    low_risk.shape[0], low_risk.shape[1], -1
                )
                consensus = low_feature.mean(dim=1, keepdim=True)
                low_index = torch.linalg.vector_norm(
                    low_feature - consensus, dim=-1
                ).topk(
                    args.num_samples - args.coverage_compression_modes,
                    dim=1,
                    largest=False,
                    sorted=True,
                ).indices
                rows = torch.arange(
                    low_risk.shape[0], device=low_risk.device
                )[:, None]
                mode_predictions = torch.cat(
                    [compressed, low_risk[rows, low_index]], dim=1
                )
            elif args.mode_alignment_gate:
                start = args.mode_alignment_expert_index * args.num_samples
                mode_predictions, _, _ = model.align_coverage_modes(
                    batch["partial"],
                    batch["seen"],
                    dual_pool[:, : args.num_samples],
                    dual_pool[:, start : start + args.num_samples],
                    retrieval.values,
                    retrieval.weights,
                )
            else:
                mode_predictions = model.select_dual_pool(
                    batch["partial"],
                    batch["seen"],
                    dual_pool,
                    args.num_samples,
                    retrieval.values,
                    retrieval.weights,
                )
            if args.dual_pool_endpoint_warp_strength > 0:
                # Keep the low-risk trajectories from the mean expert while
                # transferring the complementary endpoints of the coverage
                # expert.  The fixed time ramp is target-free at inference.
                coverage_endpoint = dual_pool[:, : args.num_samples, -1]
                endpoint_delta = coverage_endpoint - mode_predictions[:, :, -1]
                endpoint_ramp = torch.linspace(
                    1.0 / 12.0,
                    1.0,
                    12,
                    device=mode_predictions.device,
                    dtype=mode_predictions.dtype,
                ).pow(args.dual_pool_endpoint_warp_power)
                mode_predictions = mode_predictions.clone()
                mode_predictions[:, :, 8:] = (
                    mode_predictions[:, :, 8:]
                    + args.dual_pool_endpoint_warp_strength
                    * endpoint_ramp[None, None, :, None]
                    * endpoint_delta[:, :, None]
                )
        if args.conditional_mixer_modes > 0:
            mixed = model.mix_conditional_modes(
                batch["partial"],
                batch["seen"],
                mode_predictions[:, 1:],
                args.conditional_mixer_modes,
                temperature=args.conditional_mixer_temperature,
                object_type=batch.get("object_type"),
            )
            mode_predictions = torch.cat([
                center_prediction[:, None].expand(
                    -1, args.conditional_center_copies, -1, -1
                ),
                mixed,
            ], dim=1)
        mode_predictions = select_conditional_mode_subset(
            model, batch, center_prediction, mode_predictions, args
        )
        reference_samples = min(
            args.num_samples, args.conditional_reference_samples
        )
        if reference_samples:
            # Keep a fixed retrieval subset in the same candidate set.  This
            # combines database coverage with learned low-risk modes without
            # selecting candidates using targets or test-set outcomes.
            mode_predictions[:, -reference_samples:] = references[
                :, :reference_samples
            ]
        conditional_flow_strength = float(
            args.conditional_flow_refine_strength
        )
        if conditional_flow_strength > 0:
            if mode_predictions.shape[1] != args.num_samples:
                raise ValueError(
                    "conditional flow refinement requires num_samples modes"
                )
            state = mode_predictions.reshape(
                batch_size * args.num_samples, 20, 2
            )
            partial_flow = batch["partial"][:, None].expand(
                -1, args.num_samples, -1, -1
            ).reshape(batch_size * args.num_samples, 8, 2)
            seen_flow = batch["seen"][:, None].expand(
                -1, args.num_samples, -1
            ).reshape(batch_size * args.num_samples, 8)
            target_flow = batch["target"][:, None].expand(
                -1, args.num_samples, -1, -1
            ).reshape(batch_size * args.num_samples, 20, 2)
            reference_flow = references.reshape(
                batch_size * args.num_samples, 20, 2
            )
            latent_flow = latents.reshape(batch_size * args.num_samples, -1)
            step_size = conditional_flow_strength / args.flow_steps
            for step in range(args.flow_steps):
                flow_time = torch.full(
                    (state.shape[0],),
                    step / args.flow_steps,
                    device=state.device,
                    dtype=state.dtype,
                )
                flow_velocity = model(
                    state,
                    flow_time,
                    partial_flow,
                    seen_flow,
                    reference_flow,
                    latent_flow,
                )
                state = state + step_size * flow_velocity
                state = clamp_observed(state, target_flow, seen_flow)
            mode_predictions = state.reshape_as(mode_predictions)
        predictions = predictions + mode_strength * (
            mode_predictions - predictions
        )
        predictions[:, :, :8] = preserved_history
    final_set_strength = float(args.external_center_set_refiner_strength)
    if final_set_strength > 0:
        preserved_history = predictions[:, :, :8].clone()
        set_refined = model.refine_candidate_set(
            batch["partial"], batch["seen"], predictions,
            object_type=batch.get("object_type"),
        )
        predictions = predictions + final_set_strength * (
            set_refined - predictions
        )
        predictions[:, :, :8] = preserved_history
    if args.risk_center_checkpoint:
        if not hasattr(model, "external_center"):
            raise RuntimeError(
                "risk-centre fusion requires an external centre checkpoint"
            )
        preserved_history = predictions[:, :, :8].clone()
        coverage_predictions = predictions.clone()
        base_center = external_center_trajectories(
            model.external_center,
            batch,
            args.external_center_kind,
            1,
            False,
        )[:, 0]
        refined_center = model.predict_motion_mean_trajectory(
            batch["partial"],
            batch["seen"],
            base_center,
            object_type=batch.get("object_type"),
        )
        risk_center = base_center + args.risk_center_refiner_strength * (
            refined_center - base_center
        )
        if args.risk_center_memory_residual_strength > 0:
            memory_residual = model.memory_set_center(
                batch["partial"], batch["seen"], retrieval.values
            )
            risk_center = (
                risk_center
                + args.risk_center_memory_residual_strength
                * memory_residual
                * model.free_mask(batch["seen"], memory_residual.dtype)
            )
        if args.risk_center_enhanced_residual_strength > 0:
            enhanced_residual = model.predict_enhanced_memory_center(
                batch["partial"], batch["seen"], retrieval.values,
                object_type=batch.get("object_type"),
            )
            risk_center = (
                risk_center
                + args.risk_center_enhanced_residual_strength
                * enhanced_residual
                * model.free_mask(batch["seen"], enhanced_residual.dtype)
            )
        risk_center = clamp_observed(
            risk_center, batch["target"], batch["seen"]
        )
        alpha = float(args.risk_center_blend_strength)
        temporal_power = float(args.risk_center_temporal_power)
        if temporal_power > 0:
            temporal_gate = torch.linspace(
                1.0 / 12.0,
                1.0,
                12,
                device=predictions.device,
                dtype=predictions.dtype,
            ).pow(temporal_power)[None, None, :, None]
        else:
            temporal_gate = 1.0
        if args.adaptive_risk_gate:
            gate_rank_scores = None
            if args.retrieval_projection_risk_gate:
                projection_topk = (
                    retrieval.values.shape[1]
                    if args.retrieval_projection_topk <= 0
                    else min(
                        int(args.retrieval_projection_topk),
                        retrieval.values.shape[1],
                    )
                )
                projection_weights = retrieval.weights[:, :projection_topk]
                projection_weights = projection_weights / projection_weights.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)
                pseudo_target = (
                    projection_weights[:, :, None, None]
                    * retrieval.values[:, :projection_topk]
                ).sum(dim=1)
                branch_residual = (
                    coverage_predictions[:, :, 8:]
                    - risk_center[:, None, 8:]
                )
                pseudo_residual = (
                    pseudo_target[:, None, 8:]
                    - risk_center[:, None, 8:]
                )
                learned_gate = (
                    (branch_residual * pseudo_residual).sum(dim=(-1, -2))
                    / branch_residual.square().sum(
                        dim=(-1, -2)
                    ).clamp_min(1e-8)
                ).clamp(0.0, 1.0)
                learned_gate = (
                    args.retrieval_projection_scale
                    * learned_gate.pow(args.retrieval_projection_power)
                ).clamp(0.0, 1.0)
            elif args.dual_risk_gate:
                learned_gate = model.predict_coverage_risk_gate(
                    batch["partial"], batch["seen"], args.num_samples
                )
                gate_rank_scores = model.select_set_reference_logits(
                    batch["partial"],
                    batch["seen"],
                    coverage_predictions,
                    object_type=batch.get("object_type"),
                )
            elif args.memory_aware_risk_gate:
                learned_gate = model.predict_retrieval_projection_gate(
                    batch["partial"],
                    batch["seen"],
                    risk_center,
                    coverage_predictions,
                    retrieval.values,
                    retrieval.weights,
                    object_type=batch.get("object_type"),
                )
            elif args.candidate_aware_risk_gate:
                learned_gate = torch.sigmoid(
                    model.select_set_reference_logits(
                        batch["partial"],
                        batch["seen"],
                        coverage_predictions,
                        object_type=batch.get("object_type"),
                    )
                )
            else:
                learned_gate = model.predict_coverage_risk_gate(
                    batch["partial"], batch["seen"], args.num_samples
                )
            if args.adaptive_risk_gate_topk > 0:
                active = min(
                    args.adaptive_risk_gate_topk,
                    args.num_samples - args.conditional_center_copies,
                )
                coverage_gate = learned_gate[
                    :, args.conditional_center_copies:
                ]
                hard_gate = torch.zeros_like(coverage_gate).scatter_(
                    1,
                    coverage_gate.topk(active, dim=1).indices,
                    1.0,
                )
                learned_gate = learned_gate.clone()
                learned_gate[:, args.conditional_center_copies:] = hard_gate
            if args.adaptive_risk_gate_preserve_topk > 0:
                coverage_start = min(
                    max(int(args.conditional_center_copies), 0),
                    args.num_samples - 1,
                )
                preserve_count = min(
                    args.adaptive_risk_gate_preserve_topk,
                    args.num_samples - coverage_start,
                )
                preserve_scores = (
                    gate_rank_scores[:, coverage_start:]
                    if gate_rank_scores is not None
                    else learned_gate[:, coverage_start:]
                )
                preserve_index = preserve_scores.topk(
                    preserve_count, dim=1
                ).indices
                preserve_mask = torch.zeros_like(preserve_scores).scatter_(
                    1, preserve_index, 1.0
                )
                learned_gate = learned_gate.clone()
                learned_gate[:, coverage_start:] = (
                    preserve_mask
                    + (1.0 - preserve_mask) * preserve_scores
                )
            learned_gate = args.adaptive_risk_gate_min + (
                1.0 - args.adaptive_risk_gate_min
            ) * learned_gate
            temporal_gate = temporal_gate * learned_gate[:, :, None, None]
        predictions[:, :, 8:] = (
            risk_center[:, None, 8:]
            + alpha * temporal_gate
            * (
                coverage_predictions[:, :, 8:]
                - risk_center[:, None, 8:]
            )
        )
        preserve = min(
            args.num_samples,
            int(args.risk_center_memory_preserve_samples),
        )
        if preserve:
            ranking_values = (
                references
                if args.risk_center_ranking_source == "proposals"
                else retrieval.values
            )
            memory_count = min(
                ranking_values.shape[1],
                int(args.risk_center_memory_topk),
            )
            memory_values = ranking_values[:, :memory_count, 8:]
            candidate_future = coverage_predictions[:, :, 8:]
            pairwise = torch.linalg.vector_norm(
                candidate_future[:, :, None]
                - memory_values[:, None],
                dim=-1,
            )
            pairwise_ade = pairwise.mean(dim=-1)
            pairwise_fde = pairwise[:, :, :, -1]
            if args.risk_center_memory_score == "learned":
                ranking_weights = (
                    torch.full_like(
                        pairwise_ade[:, 0], 1.0 / max(memory_count, 1)
                    )
                    if args.risk_center_ranking_source == "proposals"
                    else retrieval.weights[:, :memory_count]
                )
                ranking_weights = ranking_weights / ranking_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                candidate_score = model.score_candidates_with_memory(
                    batch["partial"],
                    batch["seen"],
                    coverage_predictions,
                    ranking_values[:, :memory_count],
                    ranking_weights,
                    object_type=batch.get("object_type"),
                )
            elif args.risk_center_memory_score == "weighted":
                memory_weights = (
                    torch.ones_like(pairwise_ade[:, 0])
                    if args.risk_center_ranking_source == "proposals"
                    else retrieval.weights[:, :memory_count]
                )
                memory_weights = memory_weights / memory_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                candidate_score = (
                    pairwise_ade
                    + args.risk_center_memory_fde_weight * pairwise_fde
                ).mul(memory_weights[:, None]).sum(dim=-1)
            else:
                candidate_score = (
                    pairwise_ade
                    + args.risk_center_memory_fde_weight * pairwise_fde
                ).min(dim=-1).values
            preserve_index = candidate_score.topk(
                preserve, dim=1, largest=False, sorted=True
            ).indices
            rows = torch.arange(batch_size, device=predictions.device)[:, None]
            predictions[rows, preserve_index, 8:] = coverage_predictions[
                rows, preserve_index, 8:
            ]
        if args.risk_center_retain_slots:
            # Slot identities are learned parameters of the conditional mode
            # decoder.  A fixed subset chosen on validation data retains the
            # complementary hypotheses, while every other slot follows the
            # low-risk centre.  The same frozen slot set is then used for all
            # independent test folds; no target-dependent inference occurs.
            retain_index = torch.as_tensor(
                args.risk_center_retain_slots,
                device=predictions.device,
                dtype=torch.long,
            )
            predictions[:, retain_index, 8:] = coverage_predictions[
                :, retain_index, 8:
            ]
        retrieval_preserve = min(
            int(args.risk_center_retrieval_preserve_samples),
            args.num_samples,
            retrieval.values.shape[1],
        )
        if retrieval_preserve:
            # Train-supervised listwise ranking chooses complete database
            # records from the incomplete query alone.  Targets are used only
            # to construct labels during training and never enter inference.
            retrieval_score = model.score_candidates_with_memory(
                batch["partial"],
                batch["seen"],
                retrieval.values,
                retrieval.values,
                retrieval.weights,
                object_type=batch.get("object_type"),
            )
            retrieval_index = retrieval_score.topk(
                retrieval_preserve, dim=1, largest=False, sorted=True
            ).indices
            rows = torch.arange(
                batch_size, device=predictions.device
            )[:, None]
            selected_retrieval = retrieval.values[rows, retrieval_index]
            predictions[:, -retrieval_preserve:, 8:] = selected_retrieval[
                :, :, 8:
            ]
        if args.risk_center_set_refiner_strength > 0:
            set_refined = model.refine_candidate_set(
                batch["partial"],
                batch["seen"],
                predictions,
                object_type=batch.get("object_type"),
            )
            predictions = predictions + args.risk_center_set_refiner_strength * (
                set_refined - predictions
            )
        if args.risk_center_share_history:
            predictions[:, :, :8] = risk_center[:, None, :8]
        else:
            predictions[:, :, :8] = preserved_history
    endpoint_samples = min(
        args.num_samples, int(args.final_mantra_endpoint_samples)
    )
    if endpoint_samples:
        if mantra_endpoint is None:
            raise RuntimeError(
                "final MANTRA endpoint restoration requires --mantra-source"
            )
        current_endpoint = predictions[:, -endpoint_samples:, -1]
        predictions[:, -endpoint_samples:, -1] = current_endpoint + (
            args.final_mantra_endpoint_strength
            * (mantra_endpoint[:, :endpoint_samples] - current_endpoint)
        )
    if args.future_amplitude_scale != 1.0:
        # Target-free validation-calibrated motion amplitude.  Scaling about
        # each candidate's repaired current position preserves history repair
        # while correcting systematic future over/under-shoot.
        origin = predictions[:, :, 7:8].clone()
        predictions[:, :, 8:] = origin + args.future_amplitude_scale * (
            predictions[:, :, 8:] - origin
        )
    return predictions


@torch.no_grad()
def evaluate(model, memory, loader, args, device, coordinate_scale: float) -> Dict[str, float]:
    model.eval()
    sums = {
        "minADE": 0.0,
        "minFDE": 0.0,
        "meanADE": 0.0,
        "impute_minADE": 0.0,
        "MR@2m": 0.0,
        "JointADE": 0.0,
    }
    rows = 0
    impute_rows = 0
    p90_values = []
    gate_alpha_prediction = []
    gate_alpha_target = []
    gate_suffix_absolute_error = []
    candidate_error_rows = (
        {
            "ade": [], "fde": [], "imputation": [], "joint": [],
            "samples": [], "target": [], "future_valid": [],
            "impute_mask": [], "partial": [], "seen": [], "key": [],
            "index": [], "object_type": [], "retrieval_weights": [],
            "retrieval_indices": [], "retrieval_values": [],
        }
        if args.dump_candidate_errors else None
    )
    flow_trace_rows = (
        {
            "states": [], "references": [], "latents": [], "target": [],
            "partial": [], "seen": [], "future_valid": [], "impute_mask": [],
            "index": [], "object_type": [],
        }
        if args.dump_flow_trace else None
    )
    type_sums: Dict[int, Dict[str, float]] = {}
    type_rows: Dict[int, int] = {}
    type_impute_rows: Dict[int, int] = {}
    for raw_batch in tqdm(loader, desc="evaluate", leave=False):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        if args.model == "flow":
            samples = sample_flow(model, memory, batch, args)
        elif args.model == "anchor_flow":
            samples = sample_anchor_flow(model, memory, batch, args)
        elif args.model == "variational_flow":
            samples = sample_variational_flow(model, memory, batch, args)
        else:
            context, _ = memory_for_batch(memory, batch, args.use_memory, training=False)
            samples = model.sample(
                batch["partial"], batch["seen"], context, args.num_samples
            )
            samples[:, :, :8] = torch.where(
                batch["seen"][:, None, :, None],
                batch["target"][:, None, :8],
                samples[:, :, :8],
            )

        if flow_trace_rows is not None:
            trace = getattr(model, "_last_flow_trace", None)
            if trace is None:
                raise RuntimeError("flow trace requested but the sampler produced none")
            for key in ("states", "references", "latents"):
                flow_trace_rows[key].append(trace[key])
            for key in ("target", "partial", "seen", "future_valid", "impute_mask", "index"):
                value = batch.get(key)
                if value is not None:
                    flow_trace_rows[key].append(value.detach().cpu())
            flow_trace_rows["object_type"].append(
                batch.get(
                    "object_type",
                    torch.ones(batch["target"].shape[0], dtype=torch.long, device=device),
                ).detach().cpu()
            )

        if (
            args.mode_alignment_gate
            and hasattr(model, "mode_alignment_gate")
            and hasattr(model.mode_alignment_gate, "last_alpha")
        ):
            gate = model.mode_alignment_gate
            coverage = gate.last_coverage
            aligned_mean = gate.last_aligned_mean
            future_valid_for_gate = batch.get(
                "future_valid",
                torch.ones_like(batch["target"][:, 8:, 0], dtype=torch.bool),
            ).to(coverage.dtype)
            direction = coverage[:, :, 8:] - aligned_mean[:, :, 8:]
            target_delta = (
                batch["target"][:, None, 8:] - aligned_mean[:, :, 8:]
            )
            valid_for_gate = future_valid_for_gate[:, None, :, None]
            oracle_alpha = (
                (direction * target_delta * valid_for_gate).sum(dim=(-1, -2))
                / direction.square().mul(valid_for_gate).sum(
                    dim=(-1, -2)
                ).clamp_min(1e-8)
            ).clamp(0, 1)
            predicted_alpha = gate.last_alpha
            gate_alpha_prediction.append(predicted_alpha.detach().cpu().flatten())
            gate_alpha_target.append(oracle_alpha.detach().cpu().flatten())
            suffix_pattern = torch.cat(
                [
                    torch.ones(4, device=device, dtype=torch.bool),
                    torch.zeros(4, device=device, dtype=torch.bool),
                ]
            )
            suffix_rows = (batch["seen"] == suffix_pattern[None]).all(dim=1)
            if suffix_rows.any():
                gate_suffix_absolute_error.append(
                    (predicted_alpha[suffix_rows] - oracle_alpha[suffix_rows])
                    .abs().detach().cpu().flatten()
                )

        future_error = torch.linalg.vector_norm(
            samples[:, :, 8:] - batch["target"][:, None, 8:],
            dim=-1,
        )
        future_valid = batch.get(
            "future_valid",
            torch.ones_like(batch["target"][:, 8:, 0], dtype=torch.bool),
        )
        ade = (
            (future_error * future_valid[:, None].float()).sum(dim=-1)
            / future_valid.sum(dim=-1)[:, None].clamp_min(1)
        )
        reverse_index = torch.flip(future_valid, dims=[1]).float().argmax(dim=1)
        final_index = future_valid.shape[1] - 1 - reverse_index
        fde = future_error.gather(
            2,
            final_index[:, None, None].expand(-1, future_error.shape[1], 1),
        ).squeeze(-1)
        missing = batch.get("impute_mask", ~batch["seen"]).float()
        history_error = torch.linalg.vector_norm(
            samples[:, :, :8] - batch["target"][:, None, :8],
            dim=-1,
        )
        missing_count = missing.sum(dim=-1)
        imputation = (
            (history_error * missing[:, None]).sum(dim=-1)
            / missing_count[:, None].clamp_min(1)
        )
        joint_count = missing_count + future_valid.sum(dim=-1)
        joint = (
            (history_error * missing[:, None]).sum(dim=-1)
            + (future_error * future_valid[:, None].float()).sum(dim=-1)
        ) / joint_count[:, None].clamp_min(1)
        if candidate_error_rows is not None:
            audit_retrieval = memory.retrieve(
                batch["key"], query_indices=batch.get("index")
            )
            candidate_error_rows["ade"].append(ade.detach().cpu())
            candidate_error_rows["fde"].append(fde.detach().cpu())
            candidate_error_rows["imputation"].append(
                imputation.detach().cpu()
            )
            candidate_error_rows["joint"].append(joint.detach().cpu())
            candidate_error_rows["samples"].append(samples.detach().cpu())
            candidate_error_rows["target"].append(batch["target"].detach().cpu())
            candidate_error_rows["future_valid"].append(
                future_valid.detach().cpu()
            )
            candidate_error_rows["impute_mask"].append(missing.detach().cpu())
            candidate_error_rows["partial"].append(
                batch["partial"].detach().cpu()
            )
            candidate_error_rows["seen"].append(batch["seen"].detach().cpu())
            candidate_error_rows["key"].append(batch["key"].detach().cpu())
            candidate_error_rows["index"].append(
                batch["index"].detach().cpu()
            )
            candidate_error_rows["object_type"].append(
                batch.get(
                    "object_type",
                    torch.ones(
                        batch["target"].shape[0],
                        dtype=torch.long,
                        device=batch["target"].device,
                    ),
                ).detach().cpu()
            )
            candidate_error_rows["retrieval_weights"].append(
                audit_retrieval.weights.detach().cpu()
            )
            candidate_error_rows["retrieval_indices"].append(
                audit_retrieval.indices.detach().cpu()
            )
            candidate_error_rows["retrieval_values"].append(
                audit_retrieval.values.detach().cpu()
            )
        has_missing = missing_count > 0
        batch_count = batch["target"].shape[0]
        row_min_ade = ade.min(dim=1).values
        row_min_fde = fde.min(dim=1).values
        row_mean_ade = ade.mean(dim=1)
        row_impute_min_ade = imputation.min(dim=1).values
        sums["MR@2m"] += float((row_min_fde * coordinate_scale > 2.0).float().sum().item())
        sums["JointADE"] += float(joint.min(dim=1).values.sum().item())
        p90_values.extend((row_mean_ade * coordinate_scale).detach().cpu().tolist())
        sums["minADE"] += float(row_min_ade.sum().item())
        sums["minFDE"] += float(row_min_fde.sum().item())
        sums["meanADE"] += float(row_mean_ade.sum().item())
        if has_missing.any():
            sums["impute_minADE"] += float(
                row_impute_min_ade[has_missing].sum().item()
            )
            impute_rows += int(has_missing.sum().item())
        if "object_type" in batch:
            for type_tensor in torch.unique(batch["object_type"]):
                type_id = int(type_tensor.item())
                selected = batch["object_type"] == type_tensor
                selected_missing = selected & has_missing
                current = type_sums.setdefault(
                    type_id,
                    {
                        "minADE": 0.0,
                        "minFDE": 0.0,
                        "meanADE": 0.0,
                        "impute_minADE": 0.0,
                    },
                )
                current["minADE"] += float(row_min_ade[selected].sum().item())
                current["minFDE"] += float(row_min_fde[selected].sum().item())
                current["meanADE"] += float(row_mean_ade[selected].sum().item())
                current["impute_minADE"] += float(
                    row_impute_min_ade[selected_missing].sum().item()
                )
                type_rows[type_id] = type_rows.get(type_id, 0) + int(
                    selected.sum().item()
                )
                type_impute_rows[type_id] = type_impute_rows.get(
                    type_id, 0
                ) + int(selected_missing.sum().item())
        rows += batch_count
        if args.batch_sleep_ms > 0:
            time_module.sleep(args.batch_sleep_ms / 1000.0)
    result = {
        key: sums[key] / max(rows, 1) * coordinate_scale
        for key in ("minADE", "minFDE", "meanADE", "JointADE")
    }
    result["MR@2m"] = sums["MR@2m"] / max(rows, 1) * 100.0
    result["P90-ADE"] = float(np.percentile(np.asarray(p90_values), 90))
    result["impute_minADE"] = (
        sums["impute_minADE"] / max(impute_rows, 1) * coordinate_scale
    )
    if gate_alpha_prediction:
        predicted = torch.cat(gate_alpha_prediction).float()
        target_alpha = torch.cat(gate_alpha_target).float()
        centered_prediction = predicted - predicted.mean()
        centered_target = target_alpha - target_alpha.mean()
        correlation = (
            (centered_prediction * centered_target).mean()
            / (
                centered_prediction.square().mean().sqrt()
                * centered_target.square().mean().sqrt()
            ).clamp_min(1e-8)
        )
        result["gate_alpha_mean"] = float(predicted.mean())
        result["gate_oracle_alpha_mean"] = float(target_alpha.mean())
        result["gate_alpha_mae"] = float((predicted - target_alpha).abs().mean())
        result["gate_alpha_corr"] = float(correlation)
        if gate_suffix_absolute_error:
            result["gate_suffix_alpha_mae"] = float(
                torch.cat(gate_suffix_absolute_error).mean()
            )
    for type_id, current in sorted(type_sums.items()):
        name = OBJECT_TYPE_NAMES.get(type_id, f"type_{type_id}")
        for metric in ("minADE", "minFDE", "meanADE"):
            result[f"type_{name}_{metric}"] = (
                current[metric] / max(type_rows[type_id], 1) * coordinate_scale
            )
        result[f"type_{name}_impute_minADE"] = (
            current["impute_minADE"]
            / max(type_impute_rows[type_id], 1)
            * coordinate_scale
        )
        result[f"type_{name}_count"] = float(type_rows[type_id])
    if candidate_error_rows is not None:
        dump_path = Path(args.dump_candidate_errors)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                key: torch.cat(value, dim=0)
                for key, value in candidate_error_rows.items()
            }
            | {"coordinate_scale": float(coordinate_scale)},
            dump_path,
        )
    if flow_trace_rows is not None:
        trace_path = Path(args.dump_flow_trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                key: torch.cat(value, dim=0)
                for key, value in flow_trace_rows.items() if value
            }
            | {
                "coordinate_scale": float(coordinate_scale),
                "flow_steps": int(args.flow_steps),
                "flow_strength": float(args.flow_strength),
                "memory_flow_strength": float(args.memory_flow_strength),
            },
            trace_path,
        )
    return result


def warm_start_variational_flow(
    model: VariationalAnchorFlow,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Embed an Anchor Flow checkpoint into the latent-conditioned flow.

    The pre-latent feature columns and time columns are copied exactly, while
    the newly inserted latent columns are zeroed.  Together with the
    zero-initialized residual decoder this makes epoch zero functionally
    identical to the supplied Anchor Flow.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source = checkpoint["model"]
    target = model.flow.state_dict()
    latent_dim = model.latent_dim
    copied = {}
    for name, value in target.items():
        if name == "input_projection.weight":
            anchor_weight = source[name]
            expanded = torch.zeros_like(value)
            conditioning_width = 2 + 2 + 1 + 2
            expanded[:, :conditioning_width] = anchor_weight[:, :conditioning_width]
            expanded[:, conditioning_width + latent_dim :] = anchor_weight[
                :, conditioning_width:
            ]
            copied[name] = expanded
        elif name in source and source[name].shape == value.shape:
            copied[name] = source[name]
        elif name.startswith(
            (
                "memory_input.",
                "memory_encoder.",
                "memory_cross_attention.",
                "memory_gate.",
                "motion_projection.",
                "goal_projection.",
                "candidate_score_head.",
            )
        ):
            # Structural-fusion parameters do not exist in legacy anchor
            # checkpoints; keep their zero-gated initialization.
            continue
        else:
            raise ValueError(f"cannot warm-start flow parameter {name}")
    # New structural-fusion parameters are absent from legacy anchor
    # checkpoints; they keep their zero-gated initialization and are learned
    # during the structural-fusion fine-tuning run.
    model.flow.load_state_dict(copied, strict=False)


def load_mantra_refiner(
    model: VariationalAnchorFlow,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.mantra_refine.load_state_dict(
        {
            "0.weight": state["refine.0.weight"],
            "0.bias": state["refine.0.bias"],
            "2.weight": state["refine.2.weight"],
            "2.bias": state["refine.2.bias"],
        },
        strict=True,
    )


def load_auxiliary_heads(
    model: VariationalAnchorFlow,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load validation-trained proposal heads without replacing the flow trunk."""
    state = torch.load(checkpoint_path, map_location=device)
    state = state["model"] if "model" in state else state
    prefixes = (
        "mean_trajectory_network.",
        "mode_trajectory_network.",
        "candidate_selector_network.",
        "candidate_score_head.",
        "candidate_refiner.",
        "coverage_candidate_refiner.",
        "set_candidate_refiner.",
        "set_candidate_ranker.",
        "conditional_mode_decoder.",
        "conditional_mode_mixer.",
        "mean_motion_refiner.",
        "memory_set_center.",
        "database_center_decoder.",
    )
    selected = {key: value for key, value in state.items() if key.startswith(prefixes)}
    if not selected:
        raise ValueError(f"no proposal-head weights in {checkpoint_path}")
    model.load_state_dict(selected, strict=False)


def load_risk_center_head(
    model: VariationalAnchorFlow,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load only the independently fitted low-risk centre refiner."""
    state = torch.load(checkpoint_path, map_location=device)
    state = state["model"] if "model" in state else state
    selected = {
        key: value
        for key, value in state.items()
        if key.startswith("mean_motion_refiner.")
    }
    if not selected:
        raise ValueError(f"no low-risk centre weights in {checkpoint_path}")
    model.load_state_dict(selected, strict=False)


def load_dual_pool_experts(
    model: VariationalAnchorFlow,
    checkpoint_paths: list[str],
    device: torch.device,
) -> None:
    """Load three frozen conditional decoders beside the primary decoder."""
    if len(checkpoint_paths) != len(model.dual_pool_decoders):
        raise ValueError(
            "dual-pool selector requires exactly three expert checkpoints"
        )
    prefix = "conditional_mode_decoder."
    for decoder, checkpoint_path in zip(
        model.dual_pool_decoders, checkpoint_paths
    ):
        state = torch.load(checkpoint_path, map_location=device)
        state = state["model"] if "model" in state else state
        decoder_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not decoder_state:
            raise ValueError(
                f"no conditional-mode decoder in {checkpoint_path}"
            )
        incompatible = decoder.load_state_dict(decoder_state, strict=False)
        # Checkpoints produced before the endpoint-intention refinement do not
        # contain that zero-initialized auxiliary head.  They remain exactly
        # compatible with the original decoder path, so permit only those
        # newly introduced keys and reject every other schema mismatch.
        allowed_missing = {
            key
            for key in decoder.state_dict()
            if key.startswith("endpoint_intention.")
        }
        unexpected = set(incompatible.unexpected_keys)
        missing = set(incompatible.missing_keys)
        if unexpected or not missing.issubset(allowed_missing):
            raise RuntimeError(
                "incompatible dual-pool decoder checkpoint "
                f"{checkpoint_path}: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        decoder.eval()
        for parameter in decoder.parameters():
            parameter.requires_grad = False


def load_retrieval_key_encoder(
    model: VariationalAnchorFlow,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load a validation-trained retrieval encoder without replacing decoders."""
    state = torch.load(checkpoint_path, map_location=device)
    state = state["model"] if "model" in state else state
    selected = {
        key: value
        for key, value in state.items()
        if key.startswith("retrieval_key_encoder.")
    }
    if not selected:
        raise ValueError(f"no retrieval-key weights in {checkpoint_path}")
    model.load_state_dict(selected, strict=False)


def load_candidate_refiner(
    model: VariationalAnchorFlow,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load only the validation-trained candidate-refinement branch."""
    state = torch.load(checkpoint_path, map_location=device)
    state = state["model"] if "model" in state else state
    selected = {
        key: value
        for key, value in state.items()
        if key.startswith("candidate_refiner.")
    }
    if not selected:
        raise ValueError(f"no candidate-refiner weights in {checkpoint_path}")
    model.load_state_dict(selected, strict=False)


def load_coverage_refiner(
    model: VariationalAnchorFlow,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load a low-oracle-risk refiner into the coverage branch."""
    state = torch.load(checkpoint_path, map_location=device)
    state = state["model"] if "model" in state else state
    selected = {
        key.replace("candidate_refiner.", "coverage_candidate_refiner.", 1): value
        for key, value in state.items()
        if key.startswith("candidate_refiner.")
    }
    if not selected:
        raise ValueError(f"no candidate-refiner weights in {checkpoint_path}")
    model.load_state_dict(selected, strict=False)


def load_external_center_model(
    checkpoint_path: str, kind: str, device: torch.device
) -> nn.Module:
    """Load a matched joint-completion adapter as a frozen center branch."""
    from av2_table_baselines_remote import (
        MPEAdapter, MaskedTransformerCompletionAdapter,
    )

    center = (
        MPEAdapter(hidden=128, cycles=2)
        if kind == "mpe"
        else MaskedTransformerCompletionAdapter(hidden=128)
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    center.load_state_dict(state)
    center.eval()
    for parameter in center.parameters():
        parameter.requires_grad = False
    return center


def external_center_trajectories(
    center: nn.Module,
    batch: Dict[str, torch.Tensor],
    kind: str,
    samples: int,
    stochastic: bool,
) -> torch.Tensor:
    """Return deterministic or stochastic trajectories from a matched adapter."""
    batch_size = batch["partial"].shape[0]
    values = torch.zeros(
        batch_size, 20, 2,
        device=batch["partial"].device, dtype=batch["partial"].dtype,
    )
    values[:, :8] = batch["partial"]
    observed = torch.zeros(
        batch_size, 20, dtype=torch.bool, device=values.device
    )
    observed[:, :8] = batch["seen"]
    latent = (
        torch.randn(
            batch_size, samples, 16,
            device=values.device, dtype=values.dtype,
        )
        if stochastic
        else torch.zeros(
            batch_size, samples, 16,
            device=values.device, dtype=values.dtype,
        )
    )
    if kind == "mpe":
        repaired = center.repair_history(values, observed)
        encoded, _ = center.gru(repaired)
        context = encoded[:, -1]
        position = center.future_pos.expand(
            batch_size, -1, -1
        )[:, None].expand(-1, samples, -1, -1)
        future_input = torch.cat(
            [context[:, None, None].expand(-1, samples, 12, -1),
             latent[:, :, None].expand(-1, -1, 12, -1), position], dim=-1,
        )
        future = center.future(future_input)
    else:
        encoded = center.backbone(values, observed)
        repaired = torch.where(
            observed[:, :8, None], values[:, :8], center.impute(encoded[:, :8])
        )
        tokens = torch.cat([repaired, values[:, 8:]], dim=1)
        token_mask = torch.cat(
            [torch.ones_like(observed[:, :8]),
             torch.zeros_like(observed[:, 8:])], dim=1,
        )
        encoded = center.backbone(tokens, token_mask)
        context = encoded[:, :8].mean(dim=1)
        condition = torch.cat(
            [context[:, None].expand(-1, samples, -1), latent], dim=-1
        )
        future_hidden = encoded[:, None, 8:].expand(
            -1, samples, -1, -1
        )
        future_input = torch.cat(
            [future_hidden, condition[:, :, None].expand(-1, -1, 12, -1)],
            dim=-1,
        )
        future = center.future(future_input)
    history = repaired[:, None].expand(-1, samples, -1, -1)
    return torch.cat([history, future], dim=2)


def append_csv(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="",
        help="Optional JSON file containing argument names and values.",
    )
    parser.add_argument("--dataset", default="ZARA1")
    parser.add_argument("--difficulty", default="Easy", choices=["Easy", "Hard"])
    parser.add_argument(
        "--model",
        default="flow",
        choices=["flow", "cvae", "anchor_flow", "variational_flow"],
    )
    parser.add_argument("--use-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--variational-dim", type=int, default=128)
    parser.add_argument("--residual-loss-weight", type=float, default=0.1)
    parser.add_argument("--prior-residual-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--residual-scope",
        choices=["all", "history", "future"],
        default="all",
        help="Coordinates modified by the variational residual module.",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Gate applied to the decoded variational residual before transport.",
    )
    parser.add_argument(
        "--flow-latent-source",
        choices=["prior", "posterior", "mixed", "mean"],
        default="prior",
        help="Latent distribution used by the flow path during training.",
    )
    parser.add_argument(
        "--prior-flow-probability",
        type=float,
        default=1.0,
        help="Per-example prior probability when --flow-latent-source=mixed.",
    )
    parser.add_argument(
        "--prior-sampling",
        choices=["sample", "mean"],
        default="sample",
        help="Use a stochastic prior sample or its conditional mean at inference.",
    )
    parser.add_argument(
        "--anchor-checkpoint",
        default="",
        help="Warm-start a variational flow from a trained Anchor Flow checkpoint.",
    )
    parser.add_argument(
        "--flow-lr-scale",
        type=float,
        default=0.1,
        help="Learning-rate multiplier for the warm-started flow submodule.",
    )
    parser.add_argument(
        "--anchor-sample-fraction",
        type=float,
        default=0.0,
        help="Fraction of inference samples kept as exact zero-residual anchors.",
    )
    parser.add_argument(
        "--anchor-flow-strength",
        type=float,
        default=-1.0,
        help="Optional flow strength for the zero-residual anchor samples; negative reuses memory flow strength.",
    )
    parser.add_argument("--anchor-smooth-weight", type=float, default=1.0)
    parser.add_argument("--anchor-extrapolation-weight", type=float, default=1.0)
    parser.add_argument("--memory-k", type=int, default=10)
    parser.add_argument("--memory-temperature", type=float, default=0.1)
    parser.add_argument(
        "--retrieval-key-variant",
        type=str,
        default="legacy",
        choices=[
            "legacy",
            "suffix_motion25",
            "suffix_motion50",
            "suffix_nomask50",
            "suffix_block",
            "hybrid_motion25",
            "hybrid_motion50",
            "hybrid_nomask50",
            "hybrid_block",
        ],
        help="Deterministic retrieval key; non-legacy variants extrapolate missing suffix motion.",
    )
    parser.add_argument("--learned-retrieval-keys", action="store_true")
    parser.add_argument("--retrieval-key-checkpoint", type=str, default="")
    parser.add_argument("--retrieval-key-encoder-only", action="store_true")
    parser.add_argument("--retrieval-key-temperature", type=float, default=0.07)
    parser.add_argument("--retrieval-target-temperature", type=float, default=0.5)
    parser.add_argument("--retrieval-key-positive-k", type=int, default=8)
    parser.add_argument("--retrieval-key-fde-weight", type=float, default=0.5)
    parser.add_argument(
        "--retrieval-key-suffix-weight",
        type=float,
        default=1.0,
        help="Train-only weight for exact 00001111 suffix-missing queries.",
    )
    parser.add_argument(
        "--reranker-loss-weight",
        type=float,
        default=0.0,
        help="Training weight for candidate future-risk regression.",
    )
    parser.add_argument(
        "--reranker-fde-weight",
        type=float,
        default=0.5,
        help="Endpoint contribution to train-only candidate risk targets.",
    )
    parser.add_argument(
        "--reranker-strength",
        type=float,
        default=0.0,
        help="Validation-selected strength of risk-aware retrieval reranking.",
    )
    parser.add_argument(
        "--reranker-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to predicted candidate risk at inference.",
    )
    parser.add_argument("--refinement-reranker-strength", type=float, default=0.0)
    parser.add_argument("--refinement-reranker-fde-weight", type=float, default=0.5)
    parser.add_argument("--selector-loss-weight", type=float, default=0.0)
    parser.add_argument("--selector-strength", type=float, default=0.0)
    parser.add_argument("--selector-temperature", type=float, default=0.2)
    parser.add_argument("--selector-refined-cost", action="store_true")
    parser.add_argument("--set-aware-selector", action="store_true")
    parser.add_argument("--selector-diverse-preserve", action="store_true")
    parser.add_argument("--selector-diversity-weight", type=float, default=0.25)
    parser.add_argument(
        "--selector-only",
        action="store_true",
        help="Freeze transport/refinement and fit only the intention selector.",
    )
    parser.add_argument("--mantra-source", action="store_true")
    parser.add_argument("--mantra-checkpoint", default="")
    parser.add_argument("--mantra-preserve-endpoint", action="store_true")
    parser.add_argument("--mantra-velocity-weight", type=float, default=0.0)
    parser.add_argument("--mantra-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--mantra-recency-power", type=float, default=0.0)
    parser.add_argument(
        "--mantra-candidate-pool-multiplier", type=int, default=1
    )
    parser.add_argument("--mantra-diversity-samples", type=int, default=0)
    parser.add_argument("--mantra-diversity-relevance", type=float, default=0.25)
    parser.add_argument(
        "--mantra-training-source",
        action="store_true",
        help="Train refinement/selection heads on leave-one-out MANTRA proposals.",
    )
    parser.add_argument("--adaptive-selector-refine-strength", type=float, default=0.0)
    parser.add_argument("--adaptive-selector-preserve-samples", type=int, default=12)
    parser.add_argument(
        "--hybrid-flow-samples",
        type=int,
        default=0,
        help="Number of native MemoFM hypotheses retained alongside retrieval proposals.",
    )
    parser.add_argument("--external-center-checkpoint", default="")
    parser.add_argument(
        "--external-center-kind",
        choices=["missformer", "mpe"],
        default="missformer",
    )
    parser.add_argument("--external-center-samples", type=int, default=0)
    parser.add_argument("--external-center-stochastic", action="store_true")
    parser.add_argument("--external-center-training-source", action="store_true")
    parser.add_argument(
        "--external-center-candidate-training-source", action="store_true"
    )
    parser.add_argument(
        "--external-center-refiner-strength", type=float, default=0.0
    )
    parser.add_argument("--external-center-secondary-checkpoint", default="")
    parser.add_argument(
        "--external-center-secondary-kind",
        choices=["missformer", "mpe"],
        default="mpe",
    )
    parser.add_argument(
        "--external-center-secondary-stochastic", action="store_true"
    )
    parser.add_argument(
        "--external-center-secondary-blend", type=float, default=0.0
    )
    parser.add_argument(
        "--external-center-endpoint-warp-power", type=float, default=0.0
    )
    parser.add_argument(
        "--external-center-proposal-blend", type=float, default=0.0
    )
    parser.add_argument(
        "--external-center-set-refiner-strength", type=float, default=0.0
    )
    parser.add_argument("--conditional-mode-strength", type=float, default=0.0)
    parser.add_argument("--future-amplitude-scale", type=float, default=1.0)
    parser.add_argument(
        "--conditional-flow-refine-strength", type=float, default=0.0
    )
    parser.add_argument("--conditional-mode-decoder-only", action="store_true")
    parser.add_argument(
        "--conditional-mode-mask-aware-query", action="store_true"
    )
    parser.add_argument("--conditional-mode-decoder-v2", action="store_true")
    parser.add_argument("--conditional-ranker-only", action="store_true")
    parser.add_argument("--conditional-selected-modes", type=int, default=0)
    parser.add_argument("--conditional-mixer-only", action="store_true")
    parser.add_argument("--conditional-mixer-modes", type=int, default=0)
    parser.add_argument("--conditional-mixer-temperature", type=float, default=.25)
    parser.add_argument(
        "--conditional-mixer-distill-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--conditional-mixer-distill-endpoint-weight", type=float, default=1.0
    )
    parser.add_argument("--memory-conditional-mode", action="store_true")
    parser.add_argument("--memory-conditional-mode-v2", action="store_true")
    parser.add_argument(
        "--memory-conditional-retrieval-init-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument("--coverage-mode-decoder-only", action="store_true")
    parser.add_argument("--teacher-ranker-only", action="store_true")
    parser.add_argument("--adaptive-risk-gate", action="store_true")
    parser.add_argument("--adaptive-risk-gate-only", action="store_true")
    parser.add_argument("--candidate-aware-risk-gate", action="store_true")
    parser.add_argument("--memory-aware-risk-gate", action="store_true")
    parser.add_argument(
        "--retrieval-projection-risk-gate", action="store_true"
    )
    parser.add_argument("--retrieval-projection-topk", type=int, default=0)
    parser.add_argument("--retrieval-projection-scale", type=float, default=1.0)
    parser.add_argument("--retrieval-projection-power", type=float, default=1.0)
    parser.add_argument("--dual-risk-gate", action="store_true")
    parser.add_argument("--adaptive-risk-gate-min", type=float, default=0.0)
    parser.add_argument("--adaptive-risk-gate-topk", type=int, default=0)
    parser.add_argument(
        "--adaptive-risk-gate-preserve-topk", type=int, default=0
    )
    parser.add_argument(
        "--adaptive-risk-gate-winner-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--adaptive-risk-gate-winner-fde-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--adaptive-risk-gate-winner-positive-weight", type=float, default=5.0
    )
    parser.add_argument(
        "--adaptive-risk-gate-projection-weight", type=float, default=0.0
    )
    parser.add_argument("--risk-target-hinge-weight", type=float, default=0.0)
    parser.add_argument("--risk-target-minade", type=float, default=3.0575)
    parser.add_argument("--risk-target-minfde", type=float, default=5.7301)
    parser.add_argument("--risk-target-meanade", type=float, default=6.9343)
    parser.add_argument("--risk-target-mr", type=float, default=67.1875)
    parser.add_argument("--risk-target-p90", type=float, default=14.9002)
    parser.add_argument("--risk-target-impute", type=float, default=0.3661)
    parser.add_argument("--risk-target-jointade", type=float, default=2.4573)
    parser.add_argument(
        "--risk-target-mr-temperature", type=float, default=0.25
    )
    parser.add_argument("--risk-center-share-history", action="store_true")
    parser.add_argument("--final-mantra-endpoint-samples", type=int, default=0)
    parser.add_argument(
        "--final-mantra-endpoint-strength", type=float, default=1.0
    )
    parser.add_argument("--memory-candidate-ranker-only", action="store_true")
    parser.add_argument("--memory-ranker-v2", action="store_true")
    parser.add_argument(
        "--risk-center-memory-residual-only", action="store_true"
    )
    parser.add_argument(
        "--risk-center-enhanced-residual-only", action="store_true"
    )
    parser.add_argument("--risk-center-set-refiner-only", action="store_true")
    parser.add_argument("--conditional-center-copies", type=int, default=1)
    parser.add_argument("--conditional-coverage-modes", type=int, default=0)
    parser.add_argument("--conditional-split-decoder", action="store_true")
    parser.add_argument("--conditional-teacher-modes", type=int, default=0)
    parser.add_argument("--conditional-reference-samples", type=int, default=0)
    parser.add_argument("--dual-pool-selector", action="store_true")
    parser.add_argument("--dual-pool-selector-only", action="store_true")
    parser.add_argument(
        "--dual-pool-fixed-mixture",
        action="store_true",
        help="Use a fixed coverage/mean pool mixture without learned ranking.",
    )
    parser.add_argument("--dual-pool-fixed-coverage-count", type=int, default=6)
    parser.add_argument(
        "--dual-pool-endpoint-warp-strength", type=float, default=0.0
    )
    parser.add_argument(
        "--dual-pool-endpoint-warp-power", type=float, default=2.0
    )
    parser.add_argument(
        "--dual-pool-oracle-mixture",
        action="store_true",
        help="Diagnostic only: choose a coverage/mean count using validation targets.",
    )
    parser.add_argument("--dual-pool-fixed-coverage-index", type=int, default=-1)
    parser.add_argument(
        "--dual-pool-fixed-coverage-slots",
        default="",
        help="Comma-separated validation-selected coverage mode slots.",
    )
    parser.add_argument("--dual-pool-fixed-retrieval-mode", type=int, default=0)
    parser.add_argument(
        "--dual-pool-fixed-rank-by-memory",
        action="store_true",
        help="Rank fixed-mixture coverage modes by distance to the retrieval center.",
    )
    parser.add_argument(
        "--dual-pool-fixed-learned-ranker",
        action="store_true",
        help="Rank coverage candidates with the retrieval-conditioned risk model.",
    )
    parser.add_argument(
        "--dual-pool-ranker-selector-ensemble", action="store_true"
    )
    parser.add_argument(
        "--dual-pool-ranker-selector-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--dual-pool-ranker-mixture", action="store_true"
    )
    parser.add_argument(
        "--dual-pool-ranker-mixture-temperature", type=float, default=0.2
    )
    parser.add_argument(
        "--dual-pool-adaptive-mixture",
        action="store_true",
        help="Choose the coverage/mean mixture per validation/test record by retrieval-distance confidence.",
    )
    parser.add_argument("--dual-pool-adaptive-min-count", type=int, default=3)
    parser.add_argument("--dual-pool-adaptive-max-count", type=int, default=10)
    parser.add_argument("--dual-pool-adaptive-threshold", type=float, default=0.0)
    parser.add_argument(
        "--dual-pool-adaptive-gate",
        action="store_true",
        help="Train and use a record-level gate for the coverage/mean mixture.",
    )
    parser.add_argument(
        "--dual-pool-adaptive-gate-gain-quantile", type=float, default=0.75
    )
    parser.add_argument(
        "--dual-pool-adaptive-multiclass-gate",
        action="store_true",
        help="Use a validation-supervised six-way coverage-count gate.",
    )
    parser.add_argument(
        "--dual-pool-adaptive-gate-threshold", type=float, default=0.0
    )
    parser.add_argument(
        "--dual-pool-adaptive-gate-loss-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--dual-pool-adaptive-gate-mean-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--dual-pool-role-constrained",
        action="store_true",
        help="Reserve coverage roles for minADE/minFDE and mean roles for mean risk.",
    )
    parser.add_argument(
        "--dual-pool-expert-checkpoints",
        default="",
        help="Comma-separated complementary conditional-decoder checkpoints.",
    )
    parser.add_argument(
        "--dual-pool-repair-joint-weight", type=float, default=0.02
    )
    parser.add_argument(
        "--dual-pool-forecast-fde-weight", type=float, default=0.1
    )
    parser.add_argument(
        "--dual-pool-forecast-joint-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--dual-pool-target-temperature", type=float, default=0.25
    )
    parser.add_argument("--dual-pool-topk-weight", type=float, default=2.0)
    parser.add_argument("--dual-pool-pairwise-weight", type=float, default=0.0)
    parser.add_argument(
        "--dual-pool-pairwise-temperature", type=float, default=0.5
    )
    parser.add_argument("--dual-pool-regression-weight", type=float, default=0.0)
    parser.add_argument(
        "--dual-pool-coverage-specialist-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--dual-pool-coverage-specialist-only",
        action="store_true",
        help="Train each dual-pool head only on its inference-reserved mode pool.",
    )
    parser.add_argument(
        "--dual-pool-minade-selector-only",
        action="store_true",
        help="Train only the coverage minADE selector and mean-risk head.",
    )
    parser.add_argument(
        "--dual-pool-composite-selector-only",
        action="store_true",
        help="Train one coverage selector against normalized ADE/FDE/joint/repair risk.",
    )
    parser.add_argument(
        "--dual-pool-coverage-specialist-soft-weight",
        type=float,
        default=0.0,
        help="Add restricted soft target supervision to the role-only selector loss.",
    )
    parser.add_argument(
        "--dual-pool-coverage-regression-weight",
        type=float,
        default=0.0,
        help="Add normalized candidate-error regression to restricted selector heads.",
    )
    parser.add_argument("--mode-alignment-gate", action="store_true")
    parser.add_argument("--mode-alignment-gate-only", action="store_true")
    parser.add_argument(
        "--mode-alignment-retrieval-proxy-strength",
        type=float,
        default=0.0,
        help="Blend per-record retrieval projection into the alignment gate.",
    )
    parser.add_argument(
        "--mode-alignment-direct-output",
        action="store_true",
        help="Predict the projection strength directly with a sigmoid gate.",
    )
    parser.add_argument(
        "--mode-alignment-preserve-count",
        type=int,
        default=0,
        help="Keep this many high-alpha coverage candidates unprojected.",
    )
    parser.add_argument(
        "--mode-alignment-hard-mixture",
        action="store_true",
        help="Use low-risk modes except for the rank-selected preserved coverage candidates.",
    )
    parser.add_argument(
        "--mode-alignment-retrieval-specialists",
        action="store_true",
        help="Fill preserved specialist slots directly with top complete retrieved records.",
    )
    parser.add_argument("--mode-alignment-expert-index", type=int, default=3)
    parser.add_argument(
        "--mode-alignment-projection-weight", type=float, default=10.0
    )
    parser.add_argument(
        "--mode-alignment-projection-loss",
        choices=["smooth_l1", "bce"],
        default="smooth_l1",
    )
    parser.add_argument(
        "--mode-alignment-projection-binary-target", action="store_true"
    )
    parser.add_argument(
        "--mode-alignment-projection-positive-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--mode-alignment-hard-output-threshold", type=float, default=-1.0
    )
    parser.add_argument("--mode-alignment-mean-weight", type=float, default=2.0)
    parser.add_argument("--mode-alignment-oracle-weight", type=float, default=5.0)
    parser.add_argument("--mode-alignment-fde-weight", type=float, default=2.0)
    parser.add_argument("--mode-alignment-cvar-weight", type=float, default=1.0)
    parser.add_argument(
        "--mode-alignment-ranking-weight",
        type=float,
        default=0.0,
        help="Train-only supervision for which coverage candidates hard top-k preserves.",
    )
    parser.add_argument(
        "--mode-alignment-ranking-fde-weight", type=float, default=0.5
    )
    parser.add_argument("--coverage-compression-mixer", action="store_true")
    parser.add_argument("--coverage-compression-mixer-only", action="store_true")
    parser.add_argument("--coverage-compression-modes", type=int, default=2)
    parser.add_argument("--coverage-compression-expert-index", type=int, default=3)
    parser.add_argument(
        "--coverage-compression-temperature", type=float, default=0.25
    )
    parser.add_argument("--coverage-compression-ade-weight", type=float, default=10.0)
    parser.add_argument("--coverage-compression-fde-weight", type=float, default=10.0)
    parser.add_argument("--coverage-compression-aux-weight", type=float, default=1.0)
    parser.add_argument(
        "--coverage-compression-distill-weight", type=float, default=5.0
    )
    parser.add_argument(
        "--coverage-compression-selector-weight", type=float, default=20.0
    )
    parser.add_argument(
        "--coverage-compression-diversity-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--coverage-compression-diversity-margin", type=float, default=2.0
    )
    parser.add_argument("--direct-coverage-experts", action="store_true")
    parser.add_argument("--direct-coverage-experts-only", action="store_true")
    parser.add_argument("--direct-coverage-expert-index", type=int, default=3)
    parser.add_argument("--direct-coverage-modes", type=int, default=3)
    parser.add_argument(
        "--direct-coverage-retrieval-strength", type=float, default=1.0
    )
    parser.add_argument("--direct-coverage-oracle-weight", type=float, default=10.0)
    parser.add_argument("--direct-coverage-fde-weight", type=float, default=5.0)
    parser.add_argument("--direct-coverage-mean-weight", type=float, default=1.0)
    parser.add_argument("--direct-coverage-diversity-weight", type=float, default=1.0)
    parser.add_argument("--direct-coverage-diversity-margin", type=float, default=0.5)
    parser.add_argument("--retrieval-coverage-modes", type=int, default=0)
    parser.add_argument("--retrieval-coverage-expert-index", type=int, default=3)
    parser.add_argument("--retrieval-coverage-fde-weight", type=float, default=0.5)
    parser.add_argument(
        "--retrieval-coverage-score",
        choices=["nearest", "weighted", "mmr_nearest", "mmr_weighted"],
        default="nearest",
    )
    parser.add_argument(
        "--retrieval-coverage-diversity-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--retrieval-coverage-diversity-feature",
        choices=["endpoint", "waypoints", "full"],
        default="waypoints",
    )
    parser.add_argument(
        "--risk-center-checkpoint",
        default="",
        help="Checkpoint supplying only the frozen low-risk centre refiner.",
    )
    parser.add_argument(
        "--risk-center-refiner-strength", type=float, default=1.0
    )
    parser.add_argument(
        "--risk-center-memory-residual-strength", type=float, default=0.0
    )
    parser.add_argument(
        "--risk-center-enhanced-residual-strength", type=float, default=0.0
    )
    parser.add_argument(
        "--risk-center-set-refiner-strength", type=float, default=0.0
    )
    parser.add_argument(
        "--risk-center-blend-strength",
        type=float,
        default=1.0,
        help="Residual share retained around the low-risk centre.",
    )
    parser.add_argument(
        "--risk-center-temporal-power",
        type=float,
        default=0.0,
        help=(
            "If positive, retain coverage residuals with a late-branching "
            "time ramp while preserving their endpoints."
        ),
    )
    parser.add_argument(
        "--risk-center-memory-preserve-samples", type=int, default=0
    )
    parser.add_argument(
        "--risk-center-retain-slots",
        default="",
        help=(
            "Comma-separated, validation-selected candidate slots whose "
            "coverage residuals are retained after risk-centre fusion."
        ),
    )
    parser.add_argument("--risk-center-memory-topk", type=int, default=20)
    parser.add_argument(
        "--risk-center-ranking-source",
        choices=["memory", "proposals"],
        default="memory",
    )
    parser.add_argument(
        "--risk-center-memory-score",
        choices=["weighted", "nearest", "learned"],
        default="weighted",
    )
    parser.add_argument(
        "--risk-center-memory-fde-weight", type=float, default=0.5
    )
    parser.add_argument("--memory-ranker-fde-weight", type=float, default=0.5)
    parser.add_argument(
        "--memory-ranker-candidate-source",
        choices=["conditional", "retrieval", "mantra", "dual_pool"],
        default="conditional",
    )
    parser.add_argument(
        "--risk-center-retrieval-preserve-samples", type=int, default=0
    )
    parser.add_argument("--memory-ranker-temperature", type=float, default=0.2)
    parser.add_argument(
        "--memory-ranker-target-temperature", type=float, default=0.2
    )
    parser.add_argument(
        "--memory-ranker-regression-weight", type=float, default=0.25
    )
    parser.add_argument("--mode-set-mean-weight", type=float, default=1.0)
    parser.add_argument("--mode-set-early-mean-weight", type=float, default=0.0)
    parser.add_argument("--mode-set-early-mean-steps", type=int, default=6)
    parser.add_argument("--mode-set-coverage-mean-weight", type=float, default=0.0)
    parser.add_argument("--mode-set-suffix-weight", type=float, default=1.0)
    parser.add_argument(
        "--mode-set-risk-slot-indices", type=int, nargs="*", default=[]
    )
    parser.add_argument(
        "--mode-set-risk-slot-mean-weight", type=float, default=0.0
    )
    parser.add_argument("--mode-set-oracle-weight", type=float, default=1.0)
    parser.add_argument("--mode-set-oracle-topk", type=int, default=1)
    parser.add_argument("--mode-set-fde-weight", type=float, default=0.5)
    parser.add_argument("--mode-set-fde-topk", type=int, default=1)
    parser.add_argument("--mode-set-mean-fde-weight", type=float, default=0.1)
    parser.add_argument("--mode-set-cvar-weight", type=float, default=0.5)
    parser.add_argument("--mode-set-cvar-fraction", type=float, default=0.2)
    parser.add_argument("--mode-set-diversity-weight", type=float, default=0.05)
    parser.add_argument("--mode-set-diversity-margin", type=float, default=0.5)
    parser.add_argument("--mode-set-assignment-weight", type=float, default=0.0)
    parser.add_argument(
        "--mode-set-assignment-fde-weight", type=float, default=0.0
    )
    parser.add_argument("--mode-set-rank-weight", type=float, default=0.0)
    parser.add_argument("--mode-set-rank-fde-weight", type=float, default=0.25)
    parser.add_argument(
        "--mode-set-rank-target-temperature", type=float, default=0.5
    )
    parser.add_argument("--external-center-coverage-samples", type=int, default=0)
    parser.add_argument(
        "--external-center-coverage-warp-power", type=float, default=1.5
    )
    parser.add_argument(
        "--external-center-diverse-coverage", action="store_true"
    )
    parser.add_argument(
        "--external-center-selector-order", action="store_true"
    )
    parser.add_argument(
        "--external-center-coverage-mode",
        choices=["endpoint", "waypoints", "full"],
        default="full",
    )
    parser.add_argument("--mean-head-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--mean-head-loss-type",
        choices=["ade", "mse", "smooth_l1"],
        default="ade",
    )
    parser.add_argument("--mean-head-velocity-weight", type=float, default=0.0)
    parser.add_argument("--mean-head-acceleration-weight", type=float, default=0.0)
    parser.add_argument(
        "--mean-head-center-only",
        action="store_true",
        help="Train the deterministic branch on the leave-one-out proposal center.",
    )
    parser.add_argument("--mean-head-history-weight", type=float, default=0.25)
    parser.add_argument("--mean-head-endpoint-weight", type=float, default=2.0)
    parser.add_argument("--mean-head-cvar-weight", type=float, default=0.0)
    parser.add_argument("--mean-head-cvar-fraction", type=float, default=0.2)
    parser.add_argument(
        "--mean-head-direct-candidate-risk", action="store_true"
    )
    parser.add_argument("--mean-head-oracle-weight", type=float, default=0.0)
    parser.add_argument("--motion-mean-head", action="store_true")
    parser.add_argument("--memory-set-mean-head", action="store_true")
    parser.add_argument("--set-aware-center-head", action="store_true")
    parser.add_argument("--database-center-head", action="store_true")
    parser.add_argument("--endpoint-conditioned-center", action="store_true")
    parser.add_argument("--endpoint-warp-power", type=float, default=1.0)
    parser.add_argument("--endpoint-warp-coverage-samples", type=int, default=0)
    parser.add_argument("--endpoint-warp-coverage-power", type=float, default=1.5)
    parser.add_argument(
        "--mean-head-endpoint-warp-training", action="store_true"
    )
    parser.add_argument(
        "--mean-head-sample-fraction",
        type=float,
        default=0.0,
        help="Fraction of samples replaced by the conditional mean prediction.",
    )
    parser.add_argument("--mean-head-selector-preserve", action="store_true")
    parser.add_argument("--mean-head-proposal-center", action="store_true")
    parser.add_argument(
        "--mean-head-center-blend-strength", type=float, default=0.0
    )
    parser.add_argument(
        "--mean-head-refine-strength",
        type=float,
        default=0.0,
        help="Gate from Flow outputs toward train-supervised refined candidates.",
    )
    parser.add_argument(
        "--mean-head-refine-fraction",
        type=float,
        default=1.0,
        help="Fraction of samples receiving candidate refinement.",
    )
    parser.add_argument(
        "--mean-head-only",
        action="store_true",
        help="Freeze the existing transport model while fitting the mean head.",
    )
    parser.add_argument(
        "--reset-refinement-heads",
        action="store_true",
        help="Reinitialize proposal refinement/selection heads after loading a trunk checkpoint.",
    )
    parser.add_argument("--mode-head-loss-weight", type=float, default=0.0)
    parser.add_argument("--mode-head-temperature", type=float, default=0.1)
    parser.add_argument("--mode-head-fde-weight", type=float, default=0.5)
    parser.add_argument("--mode-head-preservation-weight", type=float, default=0.25)
    parser.add_argument("--mode-head-refine-strength", type=float, default=0.0)
    parser.add_argument("--mode-head-refine-fraction", type=float, default=0.0)
    parser.add_argument(
        "--candidate-refiner-only",
        action="store_true",
        help="Freeze the existing model and fit only the candidate-specific refiner.",
    )
    parser.add_argument("--candidate-refiner-loss-weight", type=float, default=0.0)
    parser.add_argument("--candidate-refiner-oracle-weight", type=float, default=0.0)
    parser.add_argument("--candidate-refiner-cvar-weight", type=float, default=0.0)
    parser.add_argument("--candidate-refiner-cvar-fraction", type=float, default=0.1)
    parser.add_argument("--candidate-refiner-target-fraction", type=float, default=1.0)
    parser.add_argument("--candidate-refiner-history-weight", type=float, default=0.25)
    parser.add_argument("--candidate-refiner-endpoint-weight", type=float, default=0.0)
    parser.add_argument("--candidate-refiner-preservation-weight", type=float, default=0.05)
    parser.add_argument("--candidate-refiner-diversity-weight", type=float, default=0.25)
    parser.add_argument("--candidate-refiner-diversity-ratio", type=float, default=0.8)
    parser.add_argument(
        "--candidate-refiner-strength",
        type=float,
        default=0.0,
        help="Validation-selected interpolation toward candidate-specific refinement.",
    )
    parser.add_argument("--candidate-refiner-final-strength", type=float, default=0.0)
    parser.add_argument("--candidate-refiner-final-samples", type=int, default=0)
    parser.add_argument("--coverage-refiner-checkpoint", default="")
    parser.add_argument("--set-aware-candidate-refiner", action="store_true")
    parser.add_argument("--endpoint-hermite-strength", type=float, default=0.0)
    parser.add_argument("--candidate-replication-preserve-samples", type=int, default=0)
    parser.add_argument("--candidate-replication-source-samples", type=int, default=1)
    parser.add_argument("--candidate-replication-warp-power", type=float, default=1.0)
    parser.add_argument(
        "--diversity-mode",
        choices=["endpoint", "waypoints", "full"],
        default="endpoint",
    )
    parser.add_argument("--diversity-relevance", type=float, default=0.2)
    parser.add_argument(
        "--coverage-samples",
        type=int,
        default=0,
        help="Number of unique samples reserved for diversity coverage; zero uses all.",
    )
    parser.add_argument("--source-noise", type=float, default=0.15)
    parser.add_argument(
        "--eval-source-noise",
        type=float,
        default=-1.0,
        help="Negative reuses --source-noise; zero preserves empirical memory anchors.",
    )
    parser.add_argument(
        "--flow-strength",
        type=float,
        default=1.0,
        help="Fraction of the learned flow integrated at evaluation.",
    )
    parser.add_argument(
        "--memory-flow-strength",
        type=float,
        default=-1.0,
        help="Optional separate flow strength for empirical memory anchors.",
    )
    parser.add_argument(
        "--reference-blend",
        type=float,
        default=0.0,
        help="Validation-time fraction of the retrieved reference mixed back after flow; zero preserves the learned flow output.",
    )
    parser.add_argument("--memory-source-ratio", type=float, default=0.5)
    parser.add_argument("--ot-temperature", type=float, default=0.1)
    parser.add_argument(
        "--uniform-anchor-exposure",
        type=float,
        default=0.0,
        help="Training probability mass assigned uniformly across retrieved anchors.",
    )
    parser.add_argument(
        "--ot-fde-weight",
        type=float,
        default=0.0,
        help="Endpoint share in the train-only memory coupling cost.",
    )
    parser.add_argument(
        "--endpoint-loss-weight",
        type=float,
        default=1.0,
        help="Relative flow-matching weight for the final prediction step.",
    )
    parser.add_argument(
        "--rollout-loss-weight",
        type=float,
        default=0.0,
        help="Weight for direct reconstruction after differentiable flow integration.",
    )
    parser.add_argument(
        "--risk-rollout-loss-weight",
        type=float,
        default=0.0,
        help="Future-only direct-risk weight for the retrieval-centre branch.",
    )
    parser.add_argument(
        "--risk-endpoint-loss-weight",
        type=float,
        default=1.0,
        help="Endpoint weight inside the retrieval-centre risk loss.",
    )
    parser.add_argument(
        "--risk-branch-fraction",
        type=float,
        default=0.0,
        help="Inference fraction assigned to prior-mean risk branches.",
    )
    parser.add_argument(
        "--risk-reference-shrinkage",
        type=float,
        default=0.0,
        help="Risk-reference residual retained around the retrieval centre.",
    )
    parser.add_argument("--training-rollout-steps", type=int, default=2)
    parser.add_argument(
        "--training-flow-strength",
        type=float,
        default=1.0,
        help="Integrated flow strength used by the direct rollout loss.",
    )
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--sample-chunk-size",
        type=int,
        default=1,
        help="Vectorize this many flow samples per integration chunk.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use cpu for smoke tests; guarded launchers select cuda only when idle.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fp16 autocast on CUDA; ignored on CPU.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--batch-sleep-ms",
        type=float,
        default=0.0,
        help="Voluntary delay after each train/eval batch to yield shared GPU time.",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.03,
        help="Per-process CUDA allocator cap; 0 disables the cap.",
    )
    parser.add_argument(
        "--av2-root",
        default="data/raw/Argoverse2_Motion_Forecasting",
    )
    parser.add_argument("--av2-train-size", type=int, default=8192)
    parser.add_argument("--av2-val-size", type=int, default=1024)
    parser.add_argument("--av2-test-size", type=int, default=1024)
    parser.add_argument(
        "--av2-test-offset",
        type=int,
        default=None,
        help="Offset into hash-ordered official validation scenarios for an untouched audit split.",
    )
    parser.add_argument("--av2-selection-seed", type=int, default=2026)
    parser.add_argument(
        "--mask-mode",
        choices=["random", "block", "tail", "mixed"],
        default="mixed",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.5,
        choices=[0.0, 0.1, 0.25, 0.5, 0.75],
        help="History missingness ratio; 0.0 enables complete-history forecasting.",
    )
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--av2-cache-root", default="")
    parser.add_argument(
        "--scene-aware-retrieval",
        action="store_true",
        help="Fuse the audited AV2 HD-map scene descriptor into memory retrieval keys.",
    )
    parser.add_argument("--scene-cache-root", default="")
    parser.add_argument("--scene-key-weight", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--selection-impute-ceiling",
        type=float,
        default=float("inf"),
        help="Only checkpoints at or below this validation imputation minADE are eligible.",
    )
    parser.add_argument(
        "--selection-mean-weight",
        type=float,
        default=0.0,
        help="Validation-only meanADE contribution to checkpoint selection.",
    )
    parser.add_argument(
        "--selection-p90-weight",
        type=float,
        default=0.0,
        help="Validation-only P90-ADE contribution to checkpoint selection.",
    )
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--num-folds", type=int, default=1)
    parser.add_argument("--output-root", default="results/memoflow")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Train and select on validation only; do not inspect the test split.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["train", "val", "test", "both"],
        default="both",
        help="Split evaluated by --eval-only; use val while selecting hyperparameters.",
    )
    parser.add_argument(
        "--dump-candidate-errors",
        default="",
        help=(
            "Optional validation-audit path for per-row, per-candidate ADE/FDE/"
            "imputation/joint errors. Use only with a single eval split."
        ),
    )
    parser.add_argument(
        "--dump-flow-trace",
        default="",
        help="Optional path for true per-integration-step variational-flow states.",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--auxiliary-head-checkpoint",
        default="",
        help="Load only proposal refinement/selection heads after the primary checkpoint.",
    )
    parser.add_argument(
        "--candidate-refiner-checkpoint",
        default="",
        help="Load only a separately trained candidate-refinement branch.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Initialize the complete model from a training checkpoint before fine-tuning.",
    )
    parser.add_argument(
        "--save-every-eval",
        action="store_true",
        help="Retain an epoch checkpoint at every validation evaluation.",
    )
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        config_path = Path(preliminary.config)
        config = json.loads(config_path.read_text())
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            raise ValueError(
                f"unknown configuration keys in {config_path}: "
                + ", ".join(unknown_keys)
            )
        parser.set_defaults(**config)
    args = parser.parse_args()

    args.dual_pool_expert_checkpoints = tuple(
        value.strip()
        for value in args.dual_pool_expert_checkpoints.split(",")
        if value.strip()
    )
    args.dual_pool_fixed_coverage_slots = tuple(
        int(value.strip())
        for value in args.dual_pool_fixed_coverage_slots.split(",")
        if value.strip()
    )
    if (
        args.dual_pool_selector
        or args.dual_pool_selector_only
        or args.mode_alignment_gate
        or args.mode_alignment_gate_only
        or args.coverage_compression_mixer
        or args.coverage_compression_mixer_only
        or args.direct_coverage_experts
        or args.direct_coverage_experts_only
        or args.retrieval_coverage_modes > 0
    ):
        if len(args.dual_pool_expert_checkpoints) != 3:
            raise ValueError(
                "dual-pool selection requires exactly three expert checkpoints"
            )
        if args.model != "variational_flow" or args.num_samples != 12:
            raise ValueError(
                "dual-pool selection requires 12-sample variational_flow"
            )
    if not 1 <= args.mode_alignment_expert_index <= 3:
        raise ValueError("--mode-alignment-expert-index must lie in [1, 3]")
    if not 1 <= args.coverage_compression_expert_index <= 3:
        raise ValueError("--coverage-compression-expert-index must lie in [1, 3]")
    if not 2 <= args.coverage_compression_modes <= 4:
        raise ValueError("--coverage-compression-modes must lie in [2, 4]")
    if args.coverage_compression_temperature <= 0:
        raise ValueError("--coverage-compression-temperature must be positive")
    if not 1 <= args.direct_coverage_expert_index <= 3:
        raise ValueError("--direct-coverage-expert-index must lie in [1, 3]")
    if not 1 <= args.direct_coverage_modes < args.num_samples:
        raise ValueError("--direct-coverage-modes must lie in [1, num_samples)")
    if args.direct_coverage_retrieval_strength < 0:
        raise ValueError("--direct-coverage-retrieval-strength must be non-negative")
    if not 0 <= args.retrieval_coverage_modes < args.num_samples:
        raise ValueError("--retrieval-coverage-modes must lie in [0, num_samples)")
    if not 1 <= args.retrieval_coverage_expert_index <= 3:
        raise ValueError("--retrieval-coverage-expert-index must lie in [1, 3]")
    if args.retrieval_coverage_fde_weight < 0:
        raise ValueError("--retrieval-coverage-fde-weight must be non-negative")
    if not 0 <= args.mode_alignment_retrieval_proxy_strength <= 1:
        raise ValueError(
            "--mode-alignment-retrieval-proxy-strength must lie in [0, 1]"
        )
    if args.mode_alignment_preserve_count < 0:
        raise ValueError("--mode-alignment-preserve-count must be non-negative")
    if args.mode_alignment_ranking_weight < 0:
        raise ValueError("--mode-alignment-ranking-weight must be non-negative")
    if args.mode_alignment_ranking_fde_weight < 0:
        raise ValueError("--mode-alignment-ranking-fde-weight must be non-negative")
    if args.retrieval_coverage_diversity_weight < 0:
        raise ValueError(
            "--retrieval-coverage-diversity-weight must be non-negative"
        )
    if args.dual_pool_target_temperature <= 0:
        raise ValueError("--dual-pool-target-temperature must be positive")
    if args.dual_pool_pairwise_temperature <= 0:
        raise ValueError("--dual-pool-pairwise-temperature must be positive")

    if args.model in {"anchor_flow", "variational_flow"} and not args.use_memory:
        raise ValueError(f"{args.model} requires --use-memory")

    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be positive")
    if args.sample_chunk_size < 1:
        raise ValueError("--sample-chunk-size must be positive")
    if args.coverage_samples < 0:
        raise ValueError("--coverage-samples must be non-negative")
    if args.training_rollout_steps < 1:
        raise ValueError("--training-rollout-steps must be positive")
    if args.rollout_loss_weight < 0:
        raise ValueError("--rollout-loss-weight must be non-negative")
    if args.risk_rollout_loss_weight < 0:
        raise ValueError("--risk-rollout-loss-weight must be non-negative")
    if args.reranker_loss_weight < 0:
        raise ValueError("--reranker-loss-weight must be non-negative")
    if args.reranker_fde_weight < 0:
        raise ValueError("--reranker-fde-weight must be non-negative")
    if args.reranker_strength < 0:
        raise ValueError("--reranker-strength must be non-negative")
    if args.reranker_temperature <= 0:
        raise ValueError("--reranker-temperature must be positive")
    if args.refinement_reranker_strength < 0:
        raise ValueError("--refinement-reranker-strength must be non-negative")
    if args.refinement_reranker_fde_weight < 0:
        raise ValueError("--refinement-reranker-fde-weight must be non-negative")
    if args.selector_loss_weight < 0:
        raise ValueError("--selector-loss-weight must be non-negative")
    if args.selector_strength < 0:
        raise ValueError("--selector-strength must be non-negative")
    if args.selector_temperature <= 0:
        raise ValueError("--selector-temperature must be positive")
    if args.selector_diversity_weight < 0:
        raise ValueError("--selector-diversity-weight must be non-negative")
    if args.mantra_source and not args.mantra_checkpoint:
        raise ValueError("--mantra-source requires --mantra-checkpoint")
    if args.mantra_preserve_endpoint and not args.mantra_source:
        raise ValueError("--mantra-preserve-endpoint requires --mantra-source")
    if args.mantra_training_source and not args.mantra_checkpoint:
        raise ValueError("--mantra-training-source requires --mantra-checkpoint")
    if not 0 <= args.adaptive_selector_refine_strength <= 1:
        raise ValueError("--adaptive-selector-refine-strength must lie in [0, 1]")
    if not 0 <= args.adaptive_selector_preserve_samples <= args.num_samples:
        raise ValueError("--adaptive-selector-preserve-samples must lie in [0, num_samples]")
    if not 0 <= args.hybrid_flow_samples <= args.num_samples:
        raise ValueError("--hybrid-flow-samples must lie in [0, num_samples]")
    if not 0 <= args.external_center_samples <= args.num_samples:
        raise ValueError("--external-center-samples must lie in [0, num_samples]")
    if not 0 <= args.external_center_refiner_strength <= 1:
        raise ValueError("--external-center-refiner-strength must lie in [0, 1]")
    if args.external_center_endpoint_warp_power < 0:
        raise ValueError(
            "--external-center-endpoint-warp-power must be non-negative"
        )
    if not 0 <= args.external_center_proposal_blend <= 1:
        raise ValueError("--external-center-proposal-blend must lie in [0, 1]")
    if not 0 <= args.external_center_coverage_samples <= args.external_center_samples:
        raise ValueError(
            "--external-center-coverage-samples must lie in [0, external_center_samples]"
        )
    if args.external_center_coverage_warp_power <= 0:
        raise ValueError("--external-center-coverage-warp-power must be positive")
    if args.external_center_samples and not args.external_center_checkpoint:
        raise ValueError(
            "--external-center-samples requires --external-center-checkpoint"
        )
    if args.external_center_training_source and not args.external_center_checkpoint:
        raise ValueError(
            "--external-center-training-source requires a checkpoint"
        )
    if (
        args.external_center_candidate_training_source
        and not args.external_center_checkpoint
    ):
        raise ValueError(
            "--external-center-candidate-training-source requires a checkpoint"
        )
    if not 0 <= args.external_center_set_refiner_strength <= 1:
        raise ValueError(
            "--external-center-set-refiner-strength must lie in [0, 1]"
        )
    if not 0 <= args.conditional_mode_strength <= 1:
        raise ValueError("--conditional-mode-strength must lie in [0, 1]")
    if args.conditional_flow_refine_strength < 0:
        raise ValueError(
            "--conditional-flow-refine-strength must be non-negative"
        )
    if not 1 <= args.conditional_center_copies < args.num_samples:
        raise ValueError(
            "--conditional-center-copies must lie in [1, num-samples)"
        )
    if not 0 <= args.conditional_selected_modes < args.num_samples:
        raise ValueError(
            "--conditional-selected-modes must lie in [0, num-samples)"
        )
    if args.conditional_selected_modes > 0 and args.conditional_center_copies != (
        args.num_samples - args.conditional_selected_modes
    ):
        raise ValueError(
            "with conditional selection, center copies must equal "
            "num-samples minus selected modes"
        )
    if not 0 <= args.conditional_mixer_modes < args.num_samples:
        raise ValueError(
            "--conditional-mixer-modes must lie in [0, num-samples)"
        )
    if args.conditional_mixer_modes > 0 and args.conditional_center_copies != (
        args.num_samples - args.conditional_mixer_modes
    ):
        raise ValueError(
            "with conditional mixing, center copies must equal "
            "num-samples minus mixer modes"
        )
    if args.conditional_selected_modes > 0 and args.conditional_mixer_modes > 0:
        raise ValueError(
            "conditional hard selection and differentiable mixing are exclusive"
        )
    if args.conditional_mixer_temperature <= 0:
        raise ValueError("--conditional-mixer-temperature must be positive")
    if args.conditional_mixer_distill_weight < 0:
        raise ValueError("--conditional-mixer-distill-weight must be non-negative")
    if args.conditional_mixer_distill_endpoint_weight < 0:
        raise ValueError(
            "--conditional-mixer-distill-endpoint-weight must be non-negative"
        )
    if args.conditional_mixer_only and (
        args.model != "variational_flow" or args.conditional_mixer_modes <= 0
    ):
        raise ValueError(
            "--conditional-mixer-only requires variational_flow and positive "
            "conditional mixer modes"
        )
    if args.conditional_ranker_only and (
        args.model != "variational_flow"
        or args.conditional_selected_modes <= 0
        or args.mode_set_rank_weight <= 0
    ):
        raise ValueError(
            "--conditional-ranker-only requires variational_flow, positive "
            "selected modes, and positive mode-set rank weight"
        )
    if not 0 <= args.conditional_teacher_modes <= (
        args.num_samples - args.conditional_center_copies
    ):
        raise ValueError(
            "--conditional-teacher-modes must lie in "
            "[0, num-samples - conditional-center-copies]"
        )
    if not 0 <= args.conditional_coverage_modes <= (
        args.num_samples - args.conditional_center_copies
    ):
        raise ValueError(
            "--conditional-coverage-modes must lie in "
            "[0, num-samples - conditional-center-copies]"
        )
    if args.conditional_split_decoder and not args.conditional_coverage_modes:
        raise ValueError(
            "--conditional-split-decoder requires "
            "--conditional-coverage-modes > 0"
        )
    if args.memory_conditional_mode and (
        args.conditional_split_decoder or args.conditional_teacher_modes
    ):
        raise ValueError(
            "--memory-conditional-mode cannot be combined with split or "
            "teacher modes"
        )
    if not 0 <= args.memory_conditional_retrieval_init_strength <= 1:
        raise ValueError(
            "--memory-conditional-retrieval-init-strength must lie in [0, 1]"
        )
    if not 0 <= args.conditional_reference_samples < args.num_samples:
        raise ValueError(
            "--conditional-reference-samples must lie in [0, num-samples)"
        )
    if not 0 <= args.risk_center_refiner_strength <= 1:
        raise ValueError("--risk-center-refiner-strength must lie in [0, 1]")
    if not 0 <= args.risk_center_memory_residual_strength <= 1:
        raise ValueError(
            "--risk-center-memory-residual-strength must lie in [0, 1]"
        )
    if not 0 <= args.risk_center_enhanced_residual_strength <= 1:
        raise ValueError(
            "--risk-center-enhanced-residual-strength must lie in [0, 1]"
        )
    if not 0 <= args.risk_center_set_refiner_strength <= 1:
        raise ValueError(
            "--risk-center-set-refiner-strength must lie in [0, 1]"
        )
    if not 0 <= args.risk_center_blend_strength <= 1:
        raise ValueError("--risk-center-blend-strength must lie in [0, 1]")
    if args.risk_center_temporal_power < 0:
        raise ValueError("--risk-center-temporal-power must be non-negative")
    if not 0 <= args.adaptive_risk_gate_min <= 1:
        raise ValueError("--adaptive-risk-gate-min must lie in [0, 1]")
    if not 0 <= args.adaptive_risk_gate_topk <= args.num_samples:
        raise ValueError(
            "--adaptive-risk-gate-topk must lie in [0, num_samples]"
        )
    if not 0 <= args.adaptive_risk_gate_preserve_topk <= args.num_samples:
        raise ValueError(
            "--adaptive-risk-gate-preserve-topk must lie in [0, num_samples]"
        )
    if args.adaptive_risk_gate_projection_weight < 0:
        raise ValueError(
            "--adaptive-risk-gate-projection-weight must be non-negative"
        )
    if args.risk_target_hinge_weight < 0:
        raise ValueError("--risk-target-hinge-weight must be non-negative")
    for name in (
        "risk_target_minade", "risk_target_minfde", "risk_target_meanade",
        "risk_target_mr", "risk_target_p90", "risk_target_impute",
        "risk_target_jointade",
        "risk_target_mr_temperature",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.adaptive_risk_gate_only and not args.adaptive_risk_gate:
        raise ValueError(
            "--adaptive-risk-gate-only requires --adaptive-risk-gate"
        )
    if args.candidate_aware_risk_gate and not args.adaptive_risk_gate:
        raise ValueError(
            "--candidate-aware-risk-gate requires --adaptive-risk-gate"
        )
    if args.memory_aware_risk_gate and not args.adaptive_risk_gate:
        raise ValueError(
            "--memory-aware-risk-gate requires --adaptive-risk-gate"
        )
    if args.retrieval_projection_risk_gate and not args.adaptive_risk_gate:
        raise ValueError(
            "--retrieval-projection-risk-gate requires --adaptive-risk-gate"
        )
    if args.retrieval_projection_topk < 0:
        raise ValueError("--retrieval-projection-topk must be non-negative")
    if args.retrieval_projection_scale <= 0:
        raise ValueError("--retrieval-projection-scale must be positive")
    if args.retrieval_projection_power <= 0:
        raise ValueError("--retrieval-projection-power must be positive")
    if args.dual_risk_gate and not args.adaptive_risk_gate:
        raise ValueError("--dual-risk-gate requires --adaptive-risk-gate")
    if args.dual_risk_gate and args.candidate_aware_risk_gate:
        raise ValueError(
            "--dual-risk-gate and --candidate-aware-risk-gate are exclusive"
        )
    if args.memory_aware_risk_gate and (
        args.candidate_aware_risk_gate or args.dual_risk_gate
    ):
        raise ValueError(
            "--memory-aware-risk-gate is exclusive with the other learned gates"
        )
    if args.retrieval_projection_risk_gate and (
        args.candidate_aware_risk_gate
        or args.memory_aware_risk_gate
        or args.dual_risk_gate
    ):
        raise ValueError(
            "--retrieval-projection-risk-gate is exclusive with learned gates"
        )
    if args.retrieval_projection_risk_gate and args.adaptive_risk_gate_only:
        raise ValueError(
            "retrieval projection is analytic and does not use gate-only training"
        )
    if not 0 <= args.final_mantra_endpoint_samples <= args.num_samples:
        raise ValueError(
            "--final-mantra-endpoint-samples must lie in [0, num-samples]"
        )
    if not 0 <= args.final_mantra_endpoint_strength <= 1:
        raise ValueError(
            "--final-mantra-endpoint-strength must lie in [0, 1]"
        )
    if not 0 <= args.risk_center_memory_preserve_samples <= args.num_samples:
        raise ValueError(
            "--risk-center-memory-preserve-samples must lie in [0, num-samples]"
        )
    try:
        args.risk_center_retain_slots = tuple(
            int(value.strip())
            for value in args.risk_center_retain_slots.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError(
            "--risk-center-retain-slots must be comma-separated integers"
        ) from exc
    if len(set(args.risk_center_retain_slots)) != len(
        args.risk_center_retain_slots
    ) or any(
        value < 0 or value >= args.num_samples
        for value in args.risk_center_retain_slots
    ):
        raise ValueError(
            "--risk-center-retain-slots must contain unique values in "
            "[0, num-samples)"
        )
    if args.risk_center_memory_topk < 1:
        raise ValueError("--risk-center-memory-topk must be positive")
    if args.risk_center_memory_fde_weight < 0:
        raise ValueError("--risk-center-memory-fde-weight must be non-negative")
    if args.memory_ranker_fde_weight < 0:
        raise ValueError("--memory-ranker-fde-weight must be non-negative")
    if not 0 <= args.risk_center_retrieval_preserve_samples <= args.num_samples:
        raise ValueError(
            "--risk-center-retrieval-preserve-samples must lie in "
            "[0, num-samples]"
        )
    if args.memory_ranker_temperature <= 0:
        raise ValueError("--memory-ranker-temperature must be positive")
    if args.memory_ranker_target_temperature <= 0:
        raise ValueError("--memory-ranker-target-temperature must be positive")
    if args.memory_ranker_regression_weight < 0:
        raise ValueError("--memory-ranker-regression-weight must be non-negative")
    if args.retrieval_key_temperature <= 0:
        raise ValueError("--retrieval-key-temperature must be positive")
    if args.retrieval_target_temperature <= 0:
        raise ValueError("--retrieval-target-temperature must be positive")
    if args.retrieval_key_positive_k < 1:
        raise ValueError("--retrieval-key-positive-k must be positive")
    if args.retrieval_key_fde_weight < 0:
        raise ValueError("--retrieval-key-fde-weight must be non-negative")
    if args.retrieval_key_suffix_weight <= 0:
        raise ValueError("--retrieval-key-suffix-weight must be positive")
    if args.retrieval_key_encoder_only and not args.learned_retrieval_keys:
        raise ValueError(
            "--retrieval-key-encoder-only requires --learned-retrieval-keys"
        )
    if args.retrieval_key_checkpoint and not args.learned_retrieval_keys:
        raise ValueError(
            "--retrieval-key-checkpoint requires --learned-retrieval-keys"
        )
    if args.risk_center_checkpoint and not args.external_center_checkpoint:
        raise ValueError(
            "--risk-center-checkpoint requires --external-center-checkpoint"
        )
    if args.memory_candidate_ranker_only and not args.risk_center_checkpoint:
        raise ValueError(
            "--memory-candidate-ranker-only requires --risk-center-checkpoint"
        )
    if args.risk_center_memory_residual_only and (
        not args.risk_center_checkpoint
        or args.risk_center_memory_residual_strength <= 0
    ):
        raise ValueError(
            "--risk-center-memory-residual-only requires a risk checkpoint "
            "and positive residual strength"
        )
    if args.risk_center_enhanced_residual_only and (
        not args.risk_center_checkpoint
        or args.risk_center_enhanced_residual_strength <= 0
    ):
        raise ValueError(
            "--risk-center-enhanced-residual-only requires a risk checkpoint "
            "and positive enhanced residual strength"
        )
    if args.risk_center_set_refiner_only and (
        not args.risk_center_checkpoint
        or args.risk_center_set_refiner_strength <= 0
    ):
        raise ValueError(
            "--risk-center-set-refiner-only requires a risk checkpoint and "
            "positive set-refiner strength"
        )
    for name in (
        "mode_set_mean_weight", "mode_set_coverage_mean_weight",
        "mode_set_early_mean_weight",
        "mode_set_oracle_weight", "mode_set_fde_weight",
        "mode_set_mean_fde_weight", "mode_set_cvar_weight",
        "mode_set_diversity_weight", "mode_set_diversity_margin",
        "mode_set_assignment_weight", "mode_set_assignment_fde_weight",
        "mode_set_rank_weight",
        "mode_set_rank_fde_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not 0 < args.mode_set_cvar_fraction <= 1:
        raise ValueError("--mode-set-cvar-fraction must lie in (0, 1]")
    if args.mode_set_early_mean_steps < 1:
        raise ValueError("--mode-set-early-mean-steps must be positive")
    if args.mode_set_rank_target_temperature <= 0:
        raise ValueError(
            "--mode-set-rank-target-temperature must be positive"
        )
    if args.mode_set_oracle_topk < 1 or args.mode_set_fde_topk < 1:
        raise ValueError("mode-set top-k values must be positive")
    if not 0 <= args.external_center_secondary_blend <= 1:
        raise ValueError("--external-center-secondary-blend must lie in [0, 1]")
    if (
        args.external_center_secondary_blend > 0
        and not args.external_center_secondary_checkpoint
    ):
        raise ValueError(
            "secondary blend requires --external-center-secondary-checkpoint"
        )
    if args.mean_head_loss_weight < 0:
        raise ValueError("--mean-head-loss-weight must be non-negative")
    if args.mean_head_velocity_weight < 0:
        raise ValueError("--mean-head-velocity-weight must be non-negative")
    if args.mean_head_acceleration_weight < 0:
        raise ValueError("--mean-head-acceleration-weight must be non-negative")
    if args.mean_head_history_weight < 0:
        raise ValueError("--mean-head-history-weight must be non-negative")
    if args.mean_head_endpoint_weight <= 0:
        raise ValueError("--mean-head-endpoint-weight must be positive")
    if args.mean_head_cvar_weight < 0:
        raise ValueError("--mean-head-cvar-weight must be non-negative")
    if args.mean_head_oracle_weight < 0:
        raise ValueError("--mean-head-oracle-weight must be non-negative")
    if args.mean_head_direct_candidate_risk and not (
        args.memory_set_mean_head
        or args.set_aware_center_head
        or args.database_center_head
    ):
        raise ValueError(
            "--mean-head-direct-candidate-risk requires a set center head"
        )
    if not 0 < args.mean_head_cvar_fraction <= 1:
        raise ValueError("--mean-head-cvar-fraction must lie in (0, 1]")
    if args.endpoint_warp_power <= 0:
        raise ValueError("--endpoint-warp-power must be positive")
    if args.endpoint_warp_coverage_samples < 0:
        raise ValueError("--endpoint-warp-coverage-samples must be nonnegative")
    if args.endpoint_warp_coverage_power <= 0:
        raise ValueError("--endpoint-warp-coverage-power must be positive")
    if not 0 <= args.mean_head_sample_fraction <= 1:
        raise ValueError("--mean-head-sample-fraction must lie in [0, 1]")
    if not 0 <= args.mean_head_refine_strength <= 1:
        raise ValueError("--mean-head-refine-strength must lie in [0, 1]")
    if not 0 <= args.mean_head_refine_fraction <= 1:
        raise ValueError("--mean-head-refine-fraction must lie in [0, 1]")
    if args.selection_mean_weight < 0:
        raise ValueError("--selection-mean-weight must be non-negative")
    if args.selection_p90_weight < 0:
        raise ValueError("--selection-p90-weight must be non-negative")
    if args.mode_head_loss_weight < 0:
        raise ValueError("--mode-head-loss-weight must be non-negative")
    if args.mode_head_temperature <= 0:
        raise ValueError("--mode-head-temperature must be positive")
    if args.mode_head_fde_weight < 0:
        raise ValueError("--mode-head-fde-weight must be non-negative")
    if args.mode_head_preservation_weight < 0:
        raise ValueError("--mode-head-preservation-weight must be non-negative")
    if not 0 <= args.mode_head_refine_strength <= 1:
        raise ValueError("--mode-head-refine-strength must lie in [0, 1]")
    if not 0 <= args.mode_head_refine_fraction <= 1:
        raise ValueError("--mode-head-refine-fraction must lie in [0, 1]")
    if args.candidate_refiner_loss_weight < 0:
        raise ValueError("--candidate-refiner-loss-weight must be non-negative")
    if args.candidate_refiner_oracle_weight < 0:
        raise ValueError("--candidate-refiner-oracle-weight must be non-negative")
    if args.candidate_refiner_cvar_weight < 0:
        raise ValueError("--candidate-refiner-cvar-weight must be non-negative")
    if not 0 < args.candidate_refiner_cvar_fraction <= 1:
        raise ValueError("--candidate-refiner-cvar-fraction must lie in (0, 1]")
    if not 0 < args.candidate_refiner_target_fraction <= 1:
        raise ValueError("--candidate-refiner-target-fraction must lie in (0, 1]")
    if args.candidate_refiner_history_weight < 0:
        raise ValueError("--candidate-refiner-history-weight must be non-negative")
    if args.candidate_refiner_endpoint_weight < 0:
        raise ValueError("--candidate-refiner-endpoint-weight must be non-negative")
    if args.candidate_refiner_preservation_weight < 0:
        raise ValueError("--candidate-refiner-preservation-weight must be non-negative")
    if args.candidate_refiner_diversity_weight < 0:
        raise ValueError("--candidate-refiner-diversity-weight must be non-negative")
    if not 0 <= args.candidate_refiner_diversity_ratio <= 1:
        raise ValueError("--candidate-refiner-diversity-ratio must lie in [0, 1]")
    if not 0 <= args.candidate_refiner_strength <= 1:
        raise ValueError("--candidate-refiner-strength must lie in [0, 1]")
    if not 0 <= args.candidate_refiner_final_strength <= 1:
        raise ValueError("--candidate-refiner-final-strength must lie in [0, 1]")
    if args.candidate_refiner_final_samples < 0:
        raise ValueError("--candidate-refiner-final-samples must be non-negative")
    if not 0 <= args.endpoint_hermite_strength <= 1:
        raise ValueError("--endpoint-hermite-strength must lie in [0, 1]")
    if args.candidate_replication_preserve_samples < 0:
        raise ValueError("--candidate-replication-preserve-samples must be non-negative")
    if args.candidate_replication_source_samples <= 0:
        raise ValueError("--candidate-replication-source-samples must be positive")
    if args.candidate_replication_warp_power <= 0:
        raise ValueError("--candidate-replication-warp-power must be positive")
    if not 0 <= args.risk_branch_fraction <= 1:
        raise ValueError("--risk-branch-fraction must lie in [0, 1]")
    if not 0 <= args.risk_reference_shrinkage <= 1:
        raise ValueError("--risk-reference-shrinkage must lie in [0, 1]")
    if not 0 <= args.uniform_anchor_exposure <= 1:
        raise ValueError("--uniform-anchor-exposure must lie in [0, 1]")
    if not 0 <= args.anchor_sample_fraction <= 1:
        raise ValueError("--anchor-sample-fraction must lie in [0, 1]")
    if args.anchor_flow_strength < -1:
        raise ValueError("--anchor-flow-strength must be -1 or non-negative")
    if args.batch_sleep_ms < 0:
        raise ValueError("--batch-sleep-ms must be non-negative")
    set_retrieval_key_variant(args.retrieval_key_variant)
    if args.dataset.upper() in {"AV2", "ARGOVERSE2", "ARGOVERSE_2"}:
        protocol_output = (
            Path(args.output_root)
            / (
                f"protocol_{args.dataset.upper()}_{args.mask_mode}_"
                f"{int(args.mask_ratio * 100):02d}_seed{args.mask_seed}.json"
            )
        )
        bundle = load_bundle(
            REPO_ROOT,
            args.dataset,
            args.difficulty,
            av2_root=args.av2_root,
            av2_train_size=args.av2_train_size,
            av2_val_size=args.av2_val_size,
            av2_test_size=args.av2_test_size,
            av2_test_offset=args.av2_test_offset,
            av2_selection_seed=args.av2_selection_seed,
            mask_mode=args.mask_mode,
            mask_ratio=args.mask_ratio,
            mask_seed=args.mask_seed,
            av2_cache_root=args.av2_cache_root or None,
            protocol_output=protocol_output,
        )
    else:
        bundle = load_bundle(REPO_ROOT, args.dataset, args.difficulty)
    if args.scene_aware_retrieval:
        if not args.scene_cache_root:
            raise ValueError("--scene-aware-retrieval requires --scene-cache-root")
        attach_scene_retrieval_keys(bundle, args)
    args.runtime_coordinate_scale = float(bundle.coordinate_scale)
    train_indices = None
    if args.num_folds > 1:
        if not 0 <= args.fold < args.num_folds:
            raise ValueError("--fold must be in [0, --num-folds)")
        fold_rng = np.random.default_rng(args.seed)
        all_indices = np.arange(len(bundle.train))
        fold_rng.shuffle(all_indices)
        folds = np.array_split(all_indices, args.num_folds)
        val_indices = folds[args.fold]
        train_indices_np = np.concatenate([fold for i, fold in enumerate(folds) if i != args.fold])
        train_indices = torch.as_tensor(train_indices_np, dtype=torch.long)
        train_dataset = torch.utils.data.Subset(bundle.train, train_indices_np.tolist())
        val_dataset = torch.utils.data.Subset(bundle.train, val_indices.tolist())
    else:
        train_dataset = bundle.train
        val_dataset = bundle.val
    if args.limit_train:
        train_dataset = torch.utils.data.Subset(
            bundle.train, range(min(args.limit_train, len(bundle.train)))
        )
    test_dataset = bundle.test
    if args.limit_val:
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(args.limit_val, len(val_dataset))))
    if args.limit_test:
        test_dataset = torch.utils.data.Subset(test_dataset, range(min(args.limit_test, len(test_dataset))))

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    memory = make_memory(bundle, args, device, train_indices=train_indices)
    if args.model in {"flow", "anchor_flow"}:
        model = JointFlowModel(
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "variational_flow":
        model = VariationalAnchorFlow(
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            heads=args.heads,
            dropout=args.dropout,
            latent_dim=args.latent_dim,
            variational_dim=args.variational_dim,
        ).to(device)
        if args.anchor_checkpoint:
            warm_start_variational_flow(model, args.anchor_checkpoint, device)
        if args.resume_checkpoint:
            resume = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(resume["model"], strict=False)
            if not any(
                key.startswith("coverage_mode_decoder.")
                for key in resume["model"]
            ):
                model.initialize_coverage_mode_decoder()
        if args.reset_refinement_heads:
            model.reset_refinement_heads()
        if args.mantra_checkpoint:
            load_mantra_refiner(model, args.mantra_checkpoint, device)
        if args.candidate_refiner_checkpoint:
            load_candidate_refiner(
                model, args.candidate_refiner_checkpoint, device
            )
        if args.coverage_refiner_checkpoint:
            load_coverage_refiner(
                model, args.coverage_refiner_checkpoint, device
            )
    else:
        model = JointCVAE(
            hidden_dim=max(args.hidden_dim, 128),
            latent_dim=args.latent_dim,
        ).to(device)
    if args.model == "variational_flow" and args.external_center_checkpoint:
        model.external_center = load_external_center_model(
            args.external_center_checkpoint,
            args.external_center_kind,
            device,
        )
    if (
        args.model == "variational_flow"
        and args.external_center_secondary_checkpoint
    ):
        model.external_center_secondary = load_external_center_model(
            args.external_center_secondary_checkpoint,
            args.external_center_secondary_kind,
            device,
        )
    if args.model == "variational_flow" and args.risk_center_checkpoint:
        load_risk_center_head(model, args.risk_center_checkpoint, device)
    if args.model == "variational_flow" and args.dual_pool_expert_checkpoints:
        load_dual_pool_experts(
            model, list(args.dual_pool_expert_checkpoints), device
        )
    if hasattr(model, "mode_alignment_gate"):
        model.mode_alignment_gate.retrieval_proxy_strength = (
            args.mode_alignment_retrieval_proxy_strength
        )
        model.mode_alignment_gate.direct_output = args.mode_alignment_direct_output
        model.mode_alignment_gate.preserve_count = args.mode_alignment_preserve_count
        model.mode_alignment_gate.hard_mixture = args.mode_alignment_hard_mixture
        model.mode_alignment_gate.retrieval_specialists = (
            args.mode_alignment_retrieval_specialists
        )
        model.mode_alignment_gate.hard_output_threshold = (
            args.mode_alignment_hard_output_threshold
        )
    if hasattr(model, "conditional_mode_decoder"):
        mask_aware_decoders = [model.conditional_mode_decoder]
        if hasattr(model, "coverage_mode_decoder"):
            mask_aware_decoders.append(model.coverage_mode_decoder)
        if hasattr(model, "dual_pool_decoders"):
            mask_aware_decoders.extend(list(model.dual_pool_decoders))
        for decoder in mask_aware_decoders:
            decoder.mask_aware_query = args.conditional_mode_mask_aware_query
    if hasattr(model, "dual_pool_selector"):
        model.dual_pool_role_constrained = args.dual_pool_role_constrained
        model.dual_pool_fixed_mixture = args.dual_pool_fixed_mixture
        model.dual_pool_fixed_coverage_count = args.dual_pool_fixed_coverage_count
        model.dual_pool_fixed_coverage_index = args.dual_pool_fixed_coverage_index
        model.dual_pool_fixed_coverage_slots = (
            args.dual_pool_fixed_coverage_slots
        )
        model.dual_pool_fixed_retrieval_mode = args.dual_pool_fixed_retrieval_mode
        model.dual_pool_fixed_selector_mode = (
            args.dual_pool_minade_selector_only
            or args.dual_pool_composite_selector_only
        )
        model.dual_pool_fixed_rank_by_memory = args.dual_pool_fixed_rank_by_memory
        model.dual_pool_fixed_learned_ranker = (
            args.dual_pool_fixed_learned_ranker
        )
        model.memory_ranker_v2 = args.memory_ranker_v2
        model.conditional_mode_decoder_v2_enabled = (
            args.conditional_mode_decoder_v2
        )
        model.memory_conditional_mode_decoder_v2_enabled = (
            args.memory_conditional_mode_v2
        )
        model.dual_pool_ranker_selector_ensemble = (
            args.dual_pool_ranker_selector_ensemble
        )
        model.dual_pool_ranker_selector_weight = (
            args.dual_pool_ranker_selector_weight
        )
        model.dual_pool_ranker_mixture = args.dual_pool_ranker_mixture
        model.dual_pool_ranker_mixture_temperature = (
            args.dual_pool_ranker_mixture_temperature
        )
        model.dual_pool_adaptive_mixture = args.dual_pool_adaptive_mixture
        model.dual_pool_adaptive_min_count = args.dual_pool_adaptive_min_count
        model.dual_pool_adaptive_max_count = args.dual_pool_adaptive_max_count
        model.dual_pool_adaptive_threshold = args.dual_pool_adaptive_threshold
        model.dual_pool_adaptive_gate = args.dual_pool_adaptive_gate
        model.dual_pool_adaptive_multiclass_gate = (
            args.dual_pool_adaptive_multiclass_gate
        )
        model.dual_pool_selector.dual_pool_adaptive_multiclass_gate = (
            args.dual_pool_adaptive_multiclass_gate
        )
        model.dual_pool_adaptive_gate_threshold = (
            args.dual_pool_adaptive_gate_threshold
        )
    if args.model == "variational_flow" and args.retrieval_key_checkpoint:
        load_retrieval_key_encoder(
            model, args.retrieval_key_checkpoint, device
        )
    if args.learned_retrieval_keys:
        if args.model != "variational_flow" or memory is None:
            raise ValueError(
                "--learned-retrieval-keys requires memory variational_flow"
            )
        refresh_retrieval_index(model, memory, args)
    if args.direct_coverage_experts_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.memory_conditional_mode_decoder.parameters():
            parameter.requires_grad = True
    elif args.coverage_compression_mixer_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.conditional_mode_mixer.parameters():
            parameter.requires_grad = True
    elif args.mode_alignment_gate_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.mode_alignment_gate.parameters():
            parameter.requires_grad = True
    elif args.dual_pool_selector_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.dual_pool_selector.parameters():
            parameter.requires_grad = True
    elif args.retrieval_key_encoder_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.retrieval_key_encoder.parameters():
            parameter.requires_grad = True
    elif args.adaptive_risk_gate_only:
        if (
            args.candidate_aware_risk_gate
            or args.memory_aware_risk_gate
            or args.dual_risk_gate
        ):
            valid_gate_setup = (
                args.model == "variational_flow"
                and bool(args.risk_center_checkpoint)
                and args.conditional_mode_strength > 0
            )
        else:
            valid_gate_setup = (
                args.model == "variational_flow"
                and args.conditional_split_decoder
                and args.conditional_coverage_modes > 0
                and bool(args.risk_center_checkpoint)
            )
        if not valid_gate_setup:
            raise ValueError(
                "adaptive risk-gate training requires a compatible "
                "variational-flow coverage setup and risk-center checkpoint"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        gate_modules = (
            (model.coverage_risk_gate, model.set_candidate_ranker)
            if args.dual_risk_gate
            else (
                model.retrieval_projection_gate
                if args.memory_aware_risk_gate
                else model.set_candidate_ranker
                if args.candidate_aware_risk_gate
                else model.coverage_risk_gate,
            )
        )
        for gate_module in gate_modules:
            for parameter in gate_module.parameters():
                parameter.requires_grad = True
        if args.candidate_aware_risk_gate:
            for parameter in model.set_candidate_ranker.parameters():
                parameter.requires_grad = True
    elif args.teacher_ranker_only:
        if (
            args.model != "variational_flow"
            or args.conditional_teacher_modes <= 0
            or args.mode_set_rank_weight <= 0
        ):
            raise ValueError(
                "--teacher-ranker-only requires variational_flow, positive "
                "teacher modes, and positive rank weight"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.set_candidate_ranker.parameters():
            parameter.requires_grad = True
    elif args.coverage_mode_decoder_only:
        if args.model != "variational_flow" or not args.conditional_split_decoder:
            raise ValueError(
                "--coverage-mode-decoder-only requires split variational_flow"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.coverage_mode_decoder.parameters():
            parameter.requires_grad = True
    elif args.risk_center_set_refiner_only:
        if args.model != "variational_flow":
            raise ValueError(
                "--risk-center-set-refiner-only requires variational_flow"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.set_candidate_refiner.parameters():
            parameter.requires_grad = True
    elif args.risk_center_memory_residual_only:
        if args.model != "variational_flow":
            raise ValueError(
                "--risk-center-memory-residual-only requires variational_flow"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.memory_set_center.parameters():
            parameter.requires_grad = True
    elif args.risk_center_enhanced_residual_only:
        if args.model != "variational_flow":
            raise ValueError(
                "--risk-center-enhanced-residual-only requires variational_flow"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.enhanced_memory_center.parameters():
            parameter.requires_grad = True
    elif args.memory_candidate_ranker_only:
        if args.model != "variational_flow":
            raise ValueError(
                "--memory-candidate-ranker-only requires variational_flow"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        active_ranker = (
            model.set_candidate_ranker
            if args.memory_ranker_v2
            else model.memory_candidate_ranker
        )
        for parameter in active_ranker.parameters():
            parameter.requires_grad = True
    elif args.conditional_mixer_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.conditional_mode_mixer.parameters():
            parameter.requires_grad = True
    elif args.conditional_ranker_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.set_candidate_ranker.parameters():
            parameter.requires_grad = True
    elif args.conditional_mode_decoder_only:
        if args.model != "variational_flow":
            raise ValueError(
                "--conditional-mode-decoder-only requires variational_flow"
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        active_decoder = (
            model.memory_conditional_mode_decoder_v2
            if args.memory_conditional_mode and args.memory_conditional_mode_v2
            else model.memory_conditional_mode_decoder
            if args.memory_conditional_mode
            else model.conditional_mode_decoder_v2
            if args.conditional_mode_decoder_v2
            else model.conditional_mode_decoder
        )
        for parameter in active_decoder.parameters():
            parameter.requires_grad = True
        if args.conditional_split_decoder and not args.memory_conditional_mode:
            for parameter in model.coverage_mode_decoder.parameters():
                parameter.requires_grad = True
        for parameter in model.set_candidate_ranker.parameters():
            parameter.requires_grad = True
    elif args.selector_only:
        if args.model != "variational_flow":
            raise ValueError("--selector-only requires --model variational_flow")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.candidate_selector_network.parameters():
            parameter.requires_grad = True
        for parameter in model.set_candidate_ranker.parameters():
            parameter.requires_grad = True
    elif args.mean_head_only:
        if args.model != "variational_flow":
            raise ValueError("--mean-head-only requires --model variational_flow")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.mean_trajectory_network.parameters():
            parameter.requires_grad = True
        for parameter in model.mean_motion_refiner.parameters():
            parameter.requires_grad = True
        for parameter in model.memory_set_center.parameters():
            parameter.requires_grad = True
        for parameter in model.set_candidate_refiner.parameters():
            parameter.requires_grad = True
        for parameter in model.database_center_decoder.parameters():
            parameter.requires_grad = True
        for parameter in model.mode_trajectory_network.parameters():
            parameter.requires_grad = True
        for parameter in model.candidate_selector_network.parameters():
            parameter.requires_grad = True
        for parameter in model.set_candidate_ranker.parameters():
            parameter.requires_grad = True
    elif args.candidate_refiner_only:
        if args.model != "variational_flow":
            raise ValueError("--candidate-refiner-only requires --model variational_flow")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.candidate_refiner.parameters():
            parameter.requires_grad = True
        for parameter in model.set_candidate_refiner.parameters():
            parameter.requires_grad = True
    if args.model == "variational_flow" and (
        args.anchor_checkpoint or args.resume_checkpoint
    ):
        flow_parameters = list(model.flow.parameters())
        flow_parameter_ids = {id(parameter) for parameter in flow_parameters}
        variational_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in flow_parameter_ids
        ]
        optimizer = AdamW(
            [
                {"params": variational_parameters, "lr": args.lr},
                {"params": flow_parameters, "lr": args.lr * args.flow_lr_scale},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and device.type == "cuda"
    )

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.retrieval_key_checkpoint:
            load_retrieval_key_encoder(
                model, args.retrieval_key_checkpoint, device
            )
        if args.risk_center_checkpoint:
            load_risk_center_head(model, args.risk_center_checkpoint, device)
        if args.auxiliary_head_checkpoint:
            load_auxiliary_heads(model, args.auxiliary_head_checkpoint, device)
        if args.candidate_refiner_checkpoint:
            load_candidate_refiner(
                model, args.candidate_refiner_checkpoint, device
            )
        if args.coverage_refiner_checkpoint:
            load_coverage_refiner(
                model, args.coverage_refiner_checkpoint, device
            )
        if args.external_center_checkpoint:
            model.external_center = load_external_center_model(
                args.external_center_checkpoint,
                args.external_center_kind,
                device,
            )
        if args.external_center_secondary_checkpoint:
            model.external_center_secondary = load_external_center_model(
                args.external_center_secondary_checkpoint,
                args.external_center_secondary_kind,
                device,
            )
        refresh_retrieval_index(model, memory, args)
        train_metrics = (
            evaluate(model, memory, train_loader, args, device, bundle.coordinate_scale)
            if args.eval_split == "train"
            else {}
        )
        validation = (
            evaluate(model, memory, val_loader, args, device, bundle.coordinate_scale)
            if args.eval_split in {"val", "both"}
            else {}
        )
        test_metrics = (
            evaluate(model, memory, test_loader, args, device, bundle.coordinate_scale)
            if args.eval_split in {"test", "both"}
            else {}
        )
        run_name = (
            f"{args.dataset}_{args.difficulty}_{args.model}_"
            f"{'memory' if args.use_memory else 'no_memory'}_seed{args.seed}"
        )
        if args.dataset.upper() in {"AV2", "ARGOVERSE2", "ARGOVERSE_2"}:
            run_name += f"_{args.mask_mode}{int(args.mask_ratio * 100):02d}_maskseed{args.mask_seed}"
        if args.num_folds > 1:
            run_name += f"_fold{args.fold}of{args.num_folds}"
        output_dir = Path(args.output_root) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_summary = json.loads((output_dir / "summary.json").read_text()) if (output_dir / "summary.json").exists() else {
            "run": run_name,
            "dataset": args.dataset,
            "difficulty": args.difficulty,
            "model": args.model,
            "use_memory": args.use_memory,
            "mask_mode": args.mask_mode,
            "mask_ratio": args.mask_ratio,
            "mask_seed": args.mask_seed,
            "checkpoint": args.checkpoint,
            "eval_source_noise": args.eval_source_noise,
        }
        eval_summary.update({f"val_{key}": value for key, value in validation.items()})
        eval_summary.update({f"test_{key}": value for key, value in test_metrics.items()})
        eval_summary.update({f"train_{key}": value for key, value in train_metrics.items()})
        eval_summary["extended_metrics_source"] = args.checkpoint
        (output_dir / "summary.json").write_text(
            json.dumps(eval_summary, indent=2, sort_keys=True) + "\n"
        )
        print(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "flow_strength": args.flow_strength,
                    "memory_flow_strength": args.memory_flow_strength,
                    "memory_source_ratio": args.memory_source_ratio,
                    "eval_source_noise": args.eval_source_noise,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in validation.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                },
                sort_keys=True,
            )
        )
        return

    run_name = (
        f"{args.dataset}_{args.difficulty}_{args.model}_"
        f"{'memory' if args.use_memory else 'no_memory'}_seed{args.seed}"
    )
    if args.dataset.upper() in {"AV2", "ARGOVERSE2", "ARGOVERSE_2"}:
        run_name += (
            f"_{args.mask_mode}{int(args.mask_ratio * 100):02d}_"
            f"maskseed{args.mask_seed}"
        )
    if args.num_folds > 1:
        run_name += f"_fold{args.fold}of{args.num_folds}"
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True)
    )
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    history_path = output_dir / "history.csv"

    if args.model == "variational_flow" and (
        args.anchor_checkpoint or args.resume_checkpoint
    ):
        initial_validation = evaluate(
            model, memory, val_loader, args, device, bundle.coordinate_scale
        )
        best_score = 0.5 * (
            initial_validation["minADE"] + initial_validation["minFDE"]
        ) + args.selection_mean_weight * initial_validation["meanADE"] \
            + args.selection_p90_weight * initial_validation["P90-ADE"]
        initial_row = {"epoch": 0, "train_loss": 0.0, **initial_validation}
        append_csv(history_path, initial_row)
        print(json.dumps(initial_row, sort_keys=True))
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": 0,
                "score": best_score,
                "coordinate_scale": bundle.coordinate_scale,
                "args": vars(args),
            },
            output_dir / "best.pt",
        )

    for epoch in range(1, args.epochs + 1):
        if args.direct_coverage_experts_only:
            train_loss = train_direct_coverage_experts_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.coverage_compression_mixer_only:
            train_loss = train_coverage_compression_mixer_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.mode_alignment_gate_only:
            train_loss = train_mode_alignment_gate_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.dual_pool_selector_only:
            train_loss = train_dual_pool_selector_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.retrieval_key_encoder_only:
            train_loss = train_retrieval_key_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
            refresh_retrieval_index(model, memory, args)
        elif args.adaptive_risk_gate_only:
            train_loss = train_adaptive_risk_gate_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.teacher_ranker_only:
            train_loss = train_conditional_mode_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.coverage_mode_decoder_only:
            train_loss = train_conditional_mode_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.risk_center_set_refiner_only:
            train_loss = train_risk_center_set_refiner_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.risk_center_memory_residual_only:
            train_loss = train_risk_center_memory_residual_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.risk_center_enhanced_residual_only:
            train_loss = train_enhanced_memory_center_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.memory_candidate_ranker_only:
            train_loss = train_memory_candidate_ranker_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif (
            args.conditional_mode_decoder_only
            or args.conditional_ranker_only
            or args.conditional_mixer_only
        ):
            train_loss = train_conditional_mode_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.mean_head_only or args.selector_only or args.candidate_refiner_only:
            train_loss = train_mean_head_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.model == "flow":
            train_loss = train_flow_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.model == "anchor_flow":
            train_loss = train_anchor_flow_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        elif args.model == "variational_flow":
            train_loss = train_variational_flow_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        else:
            train_loss = train_cvae_epoch(
                model, memory, train_loader, optimizer, scaler, args, device
            )
        if epoch % args.eval_every:
            continue
        validation = evaluate(
            model, memory, val_loader, args, device, bundle.coordinate_scale
        )
        score = (
            0.5 * (validation["minADE"] + validation["minFDE"])
            + args.selection_mean_weight * validation["meanADE"]
            + args.selection_p90_weight * validation["P90-ADE"]
        )
        row = {"epoch": epoch, "train_loss": train_loss, **validation}
        append_csv(history_path, row)
        print(json.dumps(row, sort_keys=True))
        if args.save_every_eval:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "score": score,
                    "coordinate_scale": bundle.coordinate_scale,
                    "args": vars(args),
                },
                output_dir / f"epoch_{epoch:03d}.pt",
            )
        eligible = validation["impute_minADE"] <= args.selection_impute_ceiling
        if eligible and score < best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "score": score,
                    "coordinate_scale": bundle.coordinate_scale,
                    "args": vars(args),
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    refresh_retrieval_index(model, memory, args)
    test_metrics = (
        {}
        if args.skip_final_test
        else evaluate(model, memory, test_loader, args, device, bundle.coordinate_scale)
    )
    summary = {
        "run": run_name,
        "dataset": args.dataset,
        "mask_mode": args.mask_mode,
        "mask_ratio": args.mask_ratio,
        "mask_seed": args.mask_seed,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "device": str(device),
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "peak_gpu_reserved_mib": (
            torch.cuda.max_memory_reserved() / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    append_csv(Path(args.output_root) / "all_results.csv", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
