from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RetrievalResult:
    values: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor
    context: torch.Tensor


class TrajectoryMemory:
    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        top_k: int = 10,
        temperature: float = 0.1,
        source_indices: Optional[torch.Tensor] = None,
    ) -> None:
        self.raw_keys = keys.float()
        self.keys = F.normalize(self.raw_keys, dim=-1)
        self.key_encoder = None
        self.values = values.float()
        self.source_indices = (
            torch.arange(len(keys), dtype=torch.long)
            if source_indices is None
            else source_indices.detach().clone().long()
        )
        if self.source_indices.numel() != len(keys):
            raise ValueError("source_indices must align one-to-one with memory rows")
        self.top_k = int(top_k)
        self.temperature = float(temperature)

    def to(self, device: torch.device) -> "TrajectoryMemory":
        self.raw_keys = self.raw_keys.to(device)
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)
        self.source_indices = self.source_indices.to(device)
        return self

    @torch.no_grad()
    def set_key_encoder(self, encoder: Optional[nn.Module]) -> None:
        """Freeze a train-supervised query encoder into the searchable index."""
        self.key_encoder = encoder
        if encoder is None:
            self.keys = F.normalize(self.raw_keys, dim=-1)
            return
        self.keys = F.normalize(encoder(self.raw_keys), dim=-1)

    @torch.no_grad()
    def retrieve(
        self,
        query_keys: torch.Tensor,
        query_indices: Optional[torch.Tensor] = None,
    ) -> RetrievalResult:
        if self.key_encoder is not None:
            query_keys = self.key_encoder(query_keys)
        similarity = F.normalize(query_keys, dim=-1) @ self.keys.T
        if query_indices is not None:
            # In cross-validation the memory is a compacted training subset,
            # whereas each dataset item retains its original AV2 row index.
            # Compare original IDs explicitly so a training query can never
            # retrieve its own complete target from the compacted memory.
            own_row = query_indices[:, None] == self.source_indices[None, :]
            similarity = similarity.masked_fill(own_row, -torch.inf)
        count = min(self.top_k, self.keys.shape[0] - (1 if query_indices is not None else 0))
        scores, indices = torch.topk(similarity, k=count, dim=-1)
        weights = torch.softmax(scores / self.temperature, dim=-1)
        values = self.values[indices]
        context = (weights[..., None, None] * values).sum(dim=1)
        return RetrievalResult(values=values, weights=weights, indices=indices, context=context)

    @staticmethod
    def sample_anchors(
        result: RetrievalResult,
        samples: int = 1,
        cover_topk: bool = False,
        diversity_mode: str = "endpoint",
        relevance_strength: float = 0.2,
        coverage_samples: int = 0,
    ) -> torch.Tensor:
        batch = result.weights.shape[0]
        candidates = result.weights.shape[1]
        if cover_topk and samples >= candidates:
            chosen = (
                torch.arange(samples, device=result.weights.device)[None, :]
                .remainder(candidates)
                .expand(batch, -1)
            )
        elif cover_topk:
            # Relevance-aware farthest-point traversal covers distinct future
            # modes from a broader history-retrieved candidate set. A bounded
            # coverage budget preserves rare modes while the remaining unique
            # samples follow retrieval relevance for probability calibration.
            coverage_count = min(
                samples,
                candidates,
                coverage_samples if coverage_samples > 0 else samples,
            )
            future_trajectory = result.values[:, :, 8:]
            if diversity_mode == "endpoint":
                future = future_trajectory[:, :, -1]
            elif diversity_mode == "waypoints":
                future = future_trajectory[:, :, [5, 11]].reshape(batch, candidates, -1)
            elif diversity_mode == "full":
                future = future_trajectory.reshape(batch, candidates, -1)
            else:
                raise ValueError(f"unknown diversity mode: {diversity_mode}")
            chosen_columns = [
                torch.zeros(batch, dtype=torch.long, device=result.weights.device)
            ]
            rows = torch.arange(batch, device=result.weights.device)
            first = future[:, 0]
            minimum_distance = torch.linalg.vector_norm(
                future - first[:, None],
                dim=-1,
            )
            relevance = torch.log(result.weights.clamp_min(1e-8))
            relevance = relevance - relevance.amin(dim=-1, keepdim=True)
            relevance = relevance / relevance.amax(dim=-1, keepdim=True).clamp_min(1e-6)
            for _ in range(1, coverage_count):
                score = minimum_distance + relevance_strength * relevance
                for previous in chosen_columns:
                    score[rows, previous] = -torch.inf
                next_column = score.argmax(dim=-1)
                chosen_columns.append(next_column)
                next_value = future[rows, next_column]
                distance = torch.linalg.vector_norm(
                    future - next_value[:, None],
                    dim=-1,
                )
                minimum_distance = torch.minimum(minimum_distance, distance)
            for _ in range(coverage_count, samples):
                score = result.weights.clone()
                for previous in chosen_columns:
                    score[rows, previous] = -torch.inf
                chosen_columns.append(score.argmax(dim=-1))
            chosen = torch.stack(chosen_columns, dim=-1)
        else:
            chosen = torch.multinomial(result.weights, num_samples=samples, replacement=True)
        rows = torch.arange(batch, device=result.weights.device)[:, None]
        return result.values[rows, chosen]


class RetrievalKeyEncoder(nn.Module):
    """Map observable incomplete-history descriptors to retrieval embeddings."""

    def __init__(
        self, input_dim: int = 32, hidden_dim: int = 128, output_dim: int = 64
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.skip = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(keys) + self.skip(keys), dim=-1)


@torch.no_grad()
def adapt_retrieved_anchors(
    values: torch.Tensor,
    partial: torch.Tensor,
    seen: torch.Tensor,
    smooth_weight: float = 1.0,
    extrapolation_weight: float = 1.0,
) -> torch.Tensor:
    """Smoothly deform retrieved complete trajectories toward visible points."""
    squeeze = values.ndim == 3
    if squeeze:
        values = values[:, None]
    if values.ndim != 4 or values.shape[-2:] != (20, 2):
        raise ValueError(f"expected [B,K,20,2] values, got {tuple(values.shape)}")
    batch, candidates = values.shape[:2]
    work = values.float()
    device = work.device
    length = work.shape[2]

    observed_mask = torch.zeros(batch, length, device=device, dtype=torch.float32)
    observed_mask[:, :8] = seen.float()
    system = torch.diag_embed(observed_mask)

    second = torch.zeros(length - 2, length, device=device)
    row = torch.arange(length - 2, device=device)
    second[row, row] = 1.0
    second[row, row + 1] = -2.0
    second[row, row + 2] = 1.0
    system = system + float(smooth_weight) * (second.T @ second)[None]

    future_difference = torch.zeros(12, length, device=device)
    future_row = torch.arange(12, device=device)
    future_index = torch.arange(8, 20, device=device)
    future_difference[future_row, future_index - 1] = -1.0
    future_difference[future_row, future_index] = 1.0
    system = system + float(extrapolation_weight) * (
        future_difference.T @ future_difference
    )[None]
    system = system + 1e-4 * torch.eye(length, device=device)[None]

    observed = torch.zeros(batch, length, 2, device=device)
    observed[:, :8] = partial.float()
    right_hand = observed_mask[:, None, :, None] * (
        observed[:, None] - work
    )
    expanded_system = system[:, None].expand(
        batch, candidates, length, length
    ).reshape(batch * candidates, length, length)
    displacement = torch.linalg.solve(
        expanded_system,
        right_hand.reshape(batch * candidates, length, 2),
    ).reshape(batch, candidates, length, 2)
    adapted = (work + displacement).to(values.dtype)
    return adapted[:, 0] if squeeze else adapted


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequency = torch.exp(
            torch.linspace(0, -6, half, device=time.device, dtype=time.dtype)
        )
        angles = time[:, None] * frequency[None, :] * 2 * torch.pi
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class JointFlowModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        latent_dim: int = 0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.time_embedding = SinusoidalTimeEmbedding(32)
        self.position_embedding = nn.Parameter(torch.randn(1, 20, hidden_dim) * 0.02)
        self.input_projection = nn.Linear(
            2 + 2 + 1 + 2 + self.latent_dim + 32,
            hidden_dim,
        )
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        # Retrieval-conditioned fusion branch. It is zero-gated at
        # initialization so legacy checkpoints remain valid warm starts.
        memory_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.memory_input = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.memory_encoder = nn.TransformerEncoder(memory_block, num_layers=1)
        self.memory_cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.memory_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.motion_projection = nn.Linear(6, hidden_dim)
        self.goal_projection = nn.Linear(hidden_dim, hidden_dim)
        for module in (self.memory_gate, self.motion_projection, self.goal_projection):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def conditioning_sequence(partial: torch.Tensor) -> torch.Tensor:
        future_zeros = torch.zeros(
            partial.shape[0],
            12,
            2,
            dtype=partial.dtype,
            device=partial.device,
        )
        return torch.cat([partial, future_zeros], dim=1)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        partial: torch.Tensor,
        seen: torch.Tensor,
        memory_context: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = state.shape[0]
        condition = self.conditioning_sequence(partial)
        seen_full = torch.zeros(batch, 20, 1, device=state.device, dtype=state.dtype)
        seen_full[:, :8, 0] = seen.float()
        time_features = self.time_embedding(time)[:, None, :].expand(-1, 20, -1)
        features = [state, condition, seen_full, memory_context]
        if self.latent_dim:
            if latent is None:
                raise ValueError("latent is required when latent_dim > 0")
            features.append(latent[:, None, :].expand(-1, 20, -1))
        features.append(time_features)
        inputs = torch.cat(features, dim=-1)
        hidden = self.input_projection(inputs) + self.position_embedding
        hidden = self.transformer(hidden)

        # Encode retrieved position, velocity and acceleration, then use a
        # query/key/value fusion path rather than broadcast-only conditioning.
        memory_velocity = torch.cat(
            [memory_context[:, :1] * 0.0,
             memory_context[:, 1:] - memory_context[:, :-1]], dim=1
        )
        memory_acceleration = torch.cat(
            [memory_velocity[:, :1] * 0.0,
             memory_velocity[:, 1:] - memory_velocity[:, :-1]], dim=1
        )
        memory_features = torch.cat(
            [memory_context, memory_velocity, memory_acceleration], dim=-1
        )
        memory_tokens = self.memory_encoder(
            self.memory_input(memory_features) + self.position_embedding
        )
        attended, _ = self.memory_cross_attention(
            hidden, memory_tokens, memory_tokens, need_weights=False
        )
        gate = torch.tanh(
            self.memory_gate(torch.cat([hidden, attended], dim=-1))
        )
        hidden = hidden + gate * attended

        # Future waypoint/goal summary provides an explicit intention signal
        # without using ground-truth future points at inference.
        future_memory = memory_tokens[:, 8:]
        goal = future_memory[:, [5, 9, 11]].mean(dim=1)
        hidden = hidden + self.goal_projection(goal)[:, None, :]

        state_velocity = torch.cat(
            [state[:, :1] * 0.0, state[:, 1:] - state[:, :-1]], dim=1
        )
        state_acceleration = torch.cat(
            [state_velocity[:, :1] * 0.0,
             state_velocity[:, 1:] - state_velocity[:, :-1]], dim=1
        )
        hidden = hidden + self.motion_projection(
            torch.cat([state, state_velocity, state_acceleration], dim=-1)
        )
        return self.output(hidden)


class MotionCandidateRefiner(nn.Module):
    """Candidate-specific residual refiner with explicit motion features."""

    def __init__(self, hidden_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        # Candidate position/velocity/acceleration (6), visible-query position
        # and mask (3), and a normalized time coordinate (1).
        self.input_projection = nn.Sequential(
            nn.Linear(10, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.query_projection = nn.Sequential(
            nn.Linear(8 * 7, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.object_embedding = nn.Embedding(10, hidden_dim)
        nn.init.zeros_(self.object_embedding.weight)
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 3,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=2)
        self.output = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        reference: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        velocity = torch.cat(
            [reference[:, :1] * 0.0, reference[:, 1:] - reference[:, :-1]], dim=1
        )
        acceleration = torch.cat(
            [velocity[:, :1] * 0.0, velocity[:, 1:] - velocity[:, :-1]], dim=1
        )
        query_position = torch.zeros_like(reference)
        query_position[:, :8] = partial
        query_mask = torch.zeros(
            reference.shape[0], 20, 1,
            device=reference.device, dtype=reference.dtype,
        )
        query_mask[:, :8, 0] = seen.to(reference.dtype)
        time = torch.linspace(
            0.0, 1.0, 20, device=reference.device, dtype=reference.dtype
        )[None, :, None].expand(reference.shape[0], -1, -1)
        token = self.input_projection(
            torch.cat(
                [reference, velocity, acceleration, query_position, query_mask, time],
                dim=-1,
            )
        )
        query_velocity = torch.cat(
            [partial[:, :1] * 0.0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        query_acceleration = torch.cat(
            [
                query_velocity[:, :1] * 0.0,
                query_velocity[:, 1:] - query_velocity[:, :-1],
            ],
            dim=1,
        )
        query_feature = torch.cat(
            [
                partial,
                query_velocity,
                query_acceleration,
                seen[..., None].to(partial.dtype),
            ],
            dim=-1,
        ).reshape(partial.shape[0], -1)
        query_embedding = self.query_projection(query_feature)
        if object_type is not None:
            query_embedding = query_embedding + self.object_embedding(
                object_type.long().clamp(0, 9)
            )
        token = token + query_embedding[:, None]
        output = self.output(self.encoder(token))
        residual = output[..., :2]
        gate = torch.sigmoid(output[..., 2:3])
        return residual, gate


class SetAwareCandidateRefiner(nn.Module):
    """Refine all candidates jointly using temporal and cross-candidate context."""

    def __init__(self, hidden_dim: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(10, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.query_projection = nn.Sequential(
            nn.Linear(8 * 7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.object_embedding = nn.Embedding(10, hidden_dim)
        temporal_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        candidate_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 3,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_block, num_layers=3
        )
        self.candidate_encoder = nn.TransformerEncoder(
            candidate_block, num_layers=2
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, candidates = references.shape[:2]
        velocity = torch.cat(
            [references[:, :, :1] * 0.0,
             references[:, :, 1:] - references[:, :, :-1]],
            dim=2,
        )
        acceleration = torch.cat(
            [velocity[:, :, :1] * 0.0,
             velocity[:, :, 1:] - velocity[:, :, :-1]],
            dim=2,
        )
        query_position = torch.zeros_like(references)
        query_position[:, :, :8] = partial[:, None]
        query_mask = torch.zeros(
            batch, candidates, 20, 1,
            device=references.device, dtype=references.dtype,
        )
        query_mask[:, :, :8, 0] = seen[:, None].to(references.dtype)
        time = torch.linspace(
            0.0, 1.0, 20,
            device=references.device, dtype=references.dtype,
        )[None, None, :, None].expand(batch, candidates, -1, -1)
        token = self.input_projection(
            torch.cat(
                [references, velocity, acceleration,
                 query_position, query_mask, time],
                dim=-1,
            )
        ).reshape(batch * candidates, 20, -1)

        query_velocity = torch.cat(
            [partial[:, :1] * 0.0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        query_acceleration = torch.cat(
            [query_velocity[:, :1] * 0.0,
             query_velocity[:, 1:] - query_velocity[:, :-1]],
            dim=1,
        )
        query_feature = torch.cat(
            [partial, query_velocity, query_acceleration,
             seen[..., None].to(partial.dtype)],
            dim=-1,
        ).reshape(batch, -1)
        query_embedding = self.query_projection(query_feature)
        if object_type is not None:
            query_embedding = query_embedding + self.object_embedding(
                object_type.long().clamp(0, 9)
            )
        token = token + query_embedding[:, None, :].expand(
            -1, candidates, -1
        ).reshape(batch * candidates, 1, -1)
        token = self.temporal_encoder(token).reshape(
            batch, candidates, 20, -1
        )
        candidate_context = self.candidate_encoder(
            token.mean(dim=2) + query_embedding[:, None]
        )
        fused = self.fusion(
            torch.cat(
                [token, candidate_context[:, :, None].expand(-1, -1, 20, -1)],
                dim=-1,
            )
        )
        output = self.output(fused)
        return output[..., :2], torch.sigmoid(output[..., 2:3])


class SetAwareCandidateRanker(nn.Module):
    """Rank retrieval candidates with temporal and cross-candidate context."""

    def __init__(self, hidden_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(10, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.query_projection = nn.Sequential(
            nn.Linear(8 * 7, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.object_embedding = nn.Embedding(10, hidden_dim)
        nn.init.zeros_(self.object_embedding.weight)
        temporal_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 3, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        candidate_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 3, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_block, 2)
        self.candidate_encoder = nn.TransformerEncoder(candidate_block, 2)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, candidates = references.shape[:2]
        velocity = torch.cat(
            [references[:, :, :1] * 0,
             references[:, :, 1:] - references[:, :, :-1]], dim=2,
        )
        acceleration = torch.cat(
            [velocity[:, :, :1] * 0,
             velocity[:, :, 1:] - velocity[:, :, :-1]], dim=2,
        )
        query_position = torch.zeros_like(references)
        query_position[:, :, :8] = partial[:, None]
        query_mask = torch.zeros(
            batch, candidates, 20, 1,
            device=references.device, dtype=references.dtype,
        )
        query_mask[:, :, :8, 0] = seen[:, None].to(references.dtype)
        time = torch.linspace(
            0, 1, 20, device=references.device, dtype=references.dtype
        )[None, None, :, None].expand(batch, candidates, -1, -1)
        token = self.input_projection(torch.cat(
            [references, velocity, acceleration,
             query_position, query_mask, time], dim=-1,
        )).reshape(batch * candidates, 20, -1)
        query_velocity = torch.cat(
            [partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        query_acceleration = torch.cat(
            [query_velocity[:, :1] * 0,
             query_velocity[:, 1:] - query_velocity[:, :-1]], dim=1,
        )
        query = self.query_projection(torch.cat(
            [partial, query_velocity, query_acceleration,
             seen[..., None].to(partial.dtype)], dim=-1,
        ).reshape(batch, -1))
        if object_type is not None:
            query = query + self.object_embedding(object_type.long().clamp(0, 9))
        token = token + query[:, None, :].expand(
            -1, candidates, -1
        ).reshape(batch * candidates, 1, -1)
        temporal = self.temporal_encoder(token).mean(dim=1).reshape(
            batch, candidates, -1
        )
        candidate = self.candidate_encoder(temporal + query[:, None])
        return self.output(candidate).squeeze(-1)


class DualResponsibilitySelector(nn.Module):
    """Select one repair specialist and complementary forecast hypotheses."""

    def __init__(self, hidden_dim: int = 192, experts: int = 4) -> None:
        super().__init__()
        self.experts = int(experts)
        self.expert_embedding = nn.Embedding(self.experts, 16)
        input_dim = 40 + 40 + 8 * 7 + 16 + 40 + 8
        self.scorers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(5)
            ]
        )
        self.gate_network = nn.Sequential(
            nn.Linear(7 + 40 + 20, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            # The first output preserves the historical binary-gate path;
            # the remaining outputs support validation-trained multi-count
            # coverage/mean mixtures.
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        candidates: torch.Tensor,
        memory_values: torch.Tensor,
        memory_weights: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, count = candidates.shape[:2]
        if count % self.experts:
            raise ValueError(
                "dual-responsibility candidate count must be divisible by experts"
            )
        velocity = torch.cat(
            [partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        acceleration = torch.cat(
            [velocity[:, :1] * 0, velocity[:, 1:] - velocity[:, :-1]], dim=1
        )
        query = torch.cat(
            [partial, velocity, acceleration, seen[..., None].to(partial.dtype)],
            dim=-1,
        ).reshape(batch, -1)
        consensus = candidates.mean(dim=1).reshape(batch, -1)
        memory_weights = memory_weights / memory_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        memory_center = (
            memory_values * memory_weights[:, :, None, None]
        ).sum(dim=1)
        center_distance = torch.linalg.vector_norm(
            candidates - memory_center[:, None], dim=-1
        )
        pairwise_distance = torch.linalg.vector_norm(
            candidates[:, :, None] - memory_values[:, None], dim=-1
        )
        pairwise_ade = pairwise_distance.mean(dim=-1)
        pairwise_fde = pairwise_distance[:, :, :, -1]
        consistency = torch.stack(
            [
                center_distance[:, :, :8].mean(dim=-1),
                center_distance[:, :, 8:].mean(dim=-1),
                center_distance[:, :, -1],
                pairwise_ade.min(dim=-1).values,
                (pairwise_ade * memory_weights[:, None]).sum(dim=-1),
                pairwise_fde.min(dim=-1).values,
                (pairwise_fde * memory_weights[:, None]).sum(dim=-1),
                torch.linalg.vector_norm(
                    candidates - candidates.mean(dim=1, keepdim=True), dim=-1
                ).mean(dim=-1),
            ],
            dim=-1,
        )
        modes_per_expert = count // self.experts
        expert_id = torch.arange(
            self.experts, device=candidates.device
        ).repeat_interleave(modes_per_expert)
        expert = self.expert_embedding(expert_id)[None].expand(batch, -1, -1)
        feature = torch.cat(
            [
                candidates.reshape(batch, count, -1),
                consensus[:, None].expand(-1, count, -1),
                query[:, None].expand(-1, count, -1),
                expert,
                memory_center.reshape(batch, -1)[:, None].expand(
                    -1, count, -1
                ),
                consistency,
            ],
            dim=-1,
        )
        return tuple(
            scorer(feature).squeeze(-1) for scorer in self.scorers
        )

    def forward_mixture_gate(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        memory_values: torch.Tensor,
        memory_weights: torch.Tensor,
    ) -> torch.Tensor:
        velocity = torch.cat(
            [partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        acceleration = torch.cat(
            [velocity[:, :1] * 0, velocity[:, 1:] - velocity[:, :-1]], dim=1
        )
        query = torch.cat(
            [
                partial.mean(dim=1),
                velocity.mean(dim=1),
                acceleration.mean(dim=1),
                seen.to(partial.dtype).mean(dim=1, keepdim=True),
            ],
            dim=-1,
        )
        weights = memory_weights / memory_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        memory_center = (
            memory_values * weights[:, :, None, None]
        ).sum(dim=1).reshape(partial.shape[0], -1)
        feature = torch.cat([query, memory_center, weights], dim=-1)
        logits = self.gate_network(feature)
        if getattr(self, "dual_pool_adaptive_multiclass_gate", False):
            return logits
        return logits[:, 0]


class ModeAlignmentGate(nn.Module):
    """Predict mode-wise residual strength between risk and coverage experts."""

    def __init__(self, hidden_dim: int = 192) -> None:
        super().__init__()
        self.retrieval_proxy_strength = 0.0
        self.direct_output = False
        self.preserve_count = 0
        self.hard_mixture = False
        self.retrieval_specialists = False
        self.hard_output_threshold = -1.0
        self.network = nn.Sequential(
            nn.Linear(8 * 7 + 40 * 4 + 15, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.rank_network = nn.Sequential(
            nn.Linear(8 * 7 + 40 * 4 + 15, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        nn.init.zeros_(self.rank_network[-1].weight)
        nn.init.zeros_(self.rank_network[-1].bias)

    @staticmethod
    def _mask_aware_kinematics(
        partial: torch.Tensor,
        seen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fill missing history without introducing zero-boundary velocities."""
        batch, steps, _ = partial.shape
        timeline = torch.arange(steps, device=partial.device)[None].expand(batch, -1)
        previous = torch.where(seen, timeline, -torch.ones_like(timeline)).cummax(1).values
        following = torch.flip(
            torch.flip(
                torch.where(seen, timeline, torch.full_like(timeline, steps)),
                dims=[1],
            ).cummin(1).values,
            dims=[1],
        )
        safe_previous = previous.clamp(min=0)
        safe_following = following.clamp(max=steps - 1)
        rows = torch.arange(batch, device=partial.device)[:, None]
        before = partial[rows, safe_previous]
        after = partial[rows, safe_following]
        denominator = (following - previous).clamp_min(1).to(partial.dtype)
        ratio = ((timeline - previous).to(partial.dtype) / denominator)[..., None]
        filled = before + ratio * (after - before)

        # Extrapolate suffixes from the last two visible observations. This is
        # especially important for the 00001111 AV2 pattern: raw zero filling
        # otherwise creates a fictitious stop and a large boundary acceleration.
        last = torch.where(seen, timeline, -torch.ones_like(timeline)).amax(1)
        without_last = seen.clone()
        valid_last = last >= 0
        without_last[torch.arange(batch, device=partial.device)[valid_last], last[valid_last]] = False
        second_last = torch.where(
            without_last, timeline, -torch.ones_like(timeline)
        ).amax(1)
        has_velocity = second_last >= 0
        last_safe = last.clamp_min(0)
        second_safe = second_last.clamp_min(0)
        tail_velocity = (
            partial[torch.arange(batch, device=partial.device), last_safe]
            - partial[torch.arange(batch, device=partial.device), second_safe]
        ) / (last_safe - second_safe).clamp_min(1).to(partial.dtype)[:, None]
        tail_velocity = torch.where(
            has_velocity[:, None], tail_velocity, torch.zeros_like(tail_velocity)
        )
        suffix = timeline > last[:, None]
        suffix_value = partial[
            torch.arange(batch, device=partial.device), last_safe
        ][:, None] + (
            timeline - last[:, None]
        ).to(partial.dtype)[..., None] * tail_velocity[:, None]
        filled = torch.where(suffix[..., None], suffix_value, filled)
        filled = torch.where(seen[..., None], partial, filled)

        velocity = torch.cat(
            [torch.zeros_like(filled[:, :1]), filled[:, 1:] - filled[:, :-1]], dim=1
        )
        acceleration = torch.cat(
            [torch.zeros_like(velocity[:, :1]), velocity[:, 1:] - velocity[:, :-1]],
            dim=1,
        )
        return filled, velocity, acceleration

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        coverage: torch.Tensor,
        aligned_mean: torch.Tensor,
        memory_values: torch.Tensor,
        memory_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, modes = coverage.shape[:2]
        filled, velocity, acceleration = self._mask_aware_kinematics(partial, seen)
        query = torch.cat(
            [filled, velocity, acceleration, seen[..., None].to(partial.dtype)],
            dim=-1,
        ).reshape(batch, -1)
        memory_weights = memory_weights / memory_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        memory_center = (
            memory_values * memory_weights[:, :, None, None]
        ).sum(dim=1)
        delta = coverage - aligned_mean
        proxy_delta = memory_center[:, None, 8:] - aligned_mean[:, :, 8:]
        direction = delta[:, :, 8:]
        proxy_alpha_center = (
            (direction * proxy_delta).sum(dim=(-1, -2))
            / direction.square().sum(dim=(-1, -2)).clamp_min(1e-8)
        ).clamp(0, 1)
        record_delta = (
            memory_values[:, None, :, 8:] - aligned_mean[:, :, None, 8:]
        )
        record_alpha = (
            (direction[:, :, None] * record_delta).sum(dim=(-1, -2))
            / direction.square().sum(dim=(-1, -2))[:, :, None].clamp_min(1e-8)
        ).clamp(0, 1)
        if record_alpha.dim() == 2:
            record_alpha = record_alpha[:, None].expand(-1, modes, -1)
        proxy_alpha_records = (
            record_alpha * memory_weights[:, None]
        ).sum(dim=-1)

        def as_mode_feature(value: torch.Tensor) -> torch.Tensor:
            while value.dim() > 2:
                value = value.mean(dim=-1)
            if value.dim() == 1:
                value = value[:, None]
            if value.shape[1] == 1:
                return value.expand(-1, modes)
            if value.shape[1] != modes:
                return value.mean(dim=1, keepdim=True).expand(-1, modes)
            return value

        proxy_alpha_center = as_mode_feature(proxy_alpha_center)
        proxy_alpha_records = as_mode_feature(proxy_alpha_records)
        proxy_alpha = (
            (1.0 - self.retrieval_proxy_strength) * proxy_alpha_center
            + self.retrieval_proxy_strength * proxy_alpha_records
        ).clamp(0, 1)
        coverage_memory = torch.linalg.vector_norm(
            coverage - memory_center[:, None], dim=-1
        )
        mean_memory = torch.linalg.vector_norm(
            aligned_mean - memory_center[:, None], dim=-1
        )
        record_distance = torch.linalg.vector_norm(record_delta, dim=-1)
        record_weight = memory_weights[:, None]
        if record_distance.dim() >= 4:
            record_distance = record_distance.mean(dim=-1)
        if record_distance.dim() == 3:
            record_distance_mean = (record_distance * record_weight).sum(dim=-1)
            record_distance_min = record_distance.amin(dim=-1)
            record_distance_std = record_distance.std(dim=-1)
        else:
            record_distance_mean = record_distance
            record_distance_min = record_distance
            record_distance_std = torch.zeros_like(record_distance)
        record_distance_mean = as_mode_feature(record_distance_mean)
        record_distance_min = as_mode_feature(record_distance_min)
        record_distance_std = as_mode_feature(record_distance_std)
        geometry = torch.stack(
            [
                proxy_alpha_records,
                proxy_alpha_center,
                (proxy_alpha_records - proxy_alpha_center).abs(),
                proxy_alpha,
                record_distance_mean,
                record_distance_min,
                record_distance_std,
                torch.linalg.vector_norm(delta[:, :, 8:], dim=-1).mean(dim=-1),
            ],
            dim=-1,
        )
        endpoint_span = torch.linalg.vector_norm(delta[:, :, -1], dim=-1)
        path_span = torch.linalg.vector_norm(delta[:, :, 8:], dim=-1).mean(-1)

        def relative_set_feature(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            rank = value.argsort(dim=1).argsort(dim=1).to(value.dtype) / float(
                max(1, modes - 1)
            )
            zscore = (value - value.mean(dim=1, keepdim=True)) / value.std(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
            return rank, zscore

        endpoint_rank, endpoint_zscore = relative_set_feature(endpoint_span)
        path_rank, path_zscore = relative_set_feature(path_span)
        coverage_consensus = coverage[:, :, 8:].mean(dim=1, keepdim=True)
        aligned_consensus = aligned_mean[:, :, 8:].mean(dim=1, keepdim=True)
        set_geometry = torch.stack(
            [
                endpoint_rank,
                endpoint_zscore,
                path_rank,
                path_zscore,
                torch.linalg.vector_norm(
                    coverage[:, :, 8:] - coverage_consensus, dim=-1
                ).mean(dim=-1),
                torch.linalg.vector_norm(
                    aligned_mean[:, :, 8:] - aligned_consensus, dim=-1
                ).mean(dim=-1),
            ],
            dim=-1,
        )
        feature = torch.cat(
            [
                query[:, None].expand(-1, modes, -1),
                coverage.reshape(batch, modes, -1),
                aligned_mean.reshape(batch, modes, -1),
                delta.reshape(batch, modes, -1),
                memory_center.reshape(batch, -1)[:, None].expand(-1, modes, -1),
                geometry,
                set_geometry,
                proxy_alpha[..., None],
            ],
            dim=-1,
        )
        raw = self.network(feature).squeeze(-1)
        rank_raw = self.rank_network(feature).squeeze(-1)
        if self.direct_output:
            learned = torch.sigmoid(raw)
        else:
            correction = .5 * torch.tanh(raw)
            learned = (proxy_alpha + correction).clamp(0, 1)
        # Projection magnitude and hard-preservation ranking are separate
        # tasks: sharing one scalar forced a candidate to stay far from the
        # low-risk centre merely because it should rank above another mode.
        self.last_rank_scores = torch.sigmoid(rank_raw)
        if self.hard_mixture:
            # Compose responsibilities explicitly: the low-risk expert fills
            # the set, while only rank-selected coverage specialists remain.
            learned = torch.zeros_like(learned)
        if self.preserve_count > 0:
            keep = self.last_rank_scores.topk(
                min(self.preserve_count, modes), dim=1, largest=True
            ).indices
            learned = learned.scatter(
                1, keep, torch.ones_like(keep, dtype=learned.dtype)
            )
        if not self.training and self.hard_output_threshold >= 0:
            learned = (learned >= self.hard_output_threshold).to(learned.dtype)
        if self.retrieval_proxy_strength >= 1.0:
            return proxy_alpha_records.clamp(0, 1)
        return learned


class ConditionalModeDecoder(nn.Module):
    """Decode a calibrated multimodal set around a conditional low-risk centre."""

    def __init__(
        self, hidden_dim: int = 192, heads: int = 6, max_modes: int = 12
    ) -> None:
        super().__init__()
        self.max_modes = int(max_modes)
        # Keep legacy zero-filled query features for old decoder checkpoints.
        # New experiments can explicitly opt into mask-aware kinematics.
        self.mask_aware_query = False
        self.center_projection = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.query_projection = nn.Sequential(
            nn.Linear(8 * 7, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.object_embedding = nn.Embedding(10, hidden_dim)
        nn.init.zeros_(self.object_embedding.weight)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.center_encoder = nn.TransformerEncoder(layer, 3)
        self.mode_embedding = nn.Parameter(
            torch.randn(max_modes, hidden_dim) * .02
        )
        self.time_embedding = nn.Parameter(torch.randn(20, hidden_dim) * .02)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, 3)
        self.endpoint_intention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.endpoint_intention[-1].weight)
        nn.init.zeros_(self.endpoint_intention[-1].bias)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        center: torch.Tensor,
        modes: int,
        object_type: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 1 <= modes <= self.max_modes:
            raise ValueError("modes must lie in [1, max_modes]")
        velocity = torch.cat(
            [center[:, :1] * 0, center[:, 1:] - center[:, :-1]], dim=1
        )
        acceleration = torch.cat(
            [velocity[:, :1] * 0, velocity[:, 1:] - velocity[:, :-1]], dim=1
        )
        time = torch.linspace(
            0, 1, 20, device=center.device, dtype=center.dtype
        )[None, :, None].expand(center.shape[0], -1, -1)
        center_token = self.center_projection(torch.cat(
            [center, velocity, acceleration, time], dim=-1
        ))
        if self.mask_aware_query:
            query_filled, query_velocity, query_acceleration = (
                ModeAlignmentGate._mask_aware_kinematics(partial, seen)
            )
        else:
            query_filled = partial
            query_velocity = torch.cat(
                [partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]], dim=1
            )
            query_acceleration = torch.cat(
                [query_velocity[:, :1] * 0,
                 query_velocity[:, 1:] - query_velocity[:, :-1]], dim=1
            )
        query = self.query_projection(torch.cat(
            [query_filled, query_velocity, query_acceleration,
             seen[..., None].to(partial.dtype)], dim=-1,
        ).reshape(partial.shape[0], -1))
        if object_type is not None:
            query = query + self.object_embedding(object_type.long().clamp(0, 9))
        encoded = self.center_encoder(center_token + query[:, None])
        pooled = encoded.mean(dim=1)
        mode = self.mode_embedding[:modes][None].expand(center.shape[0], -1, -1)
        temporal = encoded[:, None].expand(-1, modes, -1, -1)
        context = torch.cat([
            temporal,
            pooled[:, None, None].expand(-1, modes, 20, -1),
            mode[:, :, None].expand(-1, -1, 20, -1)
            + self.time_embedding[None, None],
        ], dim=-1)
        output = self.output(self.fusion(context))
        endpoint = self.endpoint_intention(torch.cat([
            pooled[:, None].expand(-1, modes, -1),
            query[:, None].expand(-1, modes, -1),
            mode,
        ], dim=-1))
        waypoint_ramp = torch.linspace(
            0, 1, 20, device=center.device, dtype=center.dtype
        ).pow(1.5)[None, None, :, None]
        residual = output[..., :2] + waypoint_ramp * endpoint[:, :, None]
        return residual, torch.sigmoid(output[..., 2:3])


class ConditionalModeMixer(nn.Module):
    """Compress a coverage pool into a small query-conditioned mode set."""

    def __init__(
        self, hidden_dim: int = 192, heads: int = 6, max_heads: int = 4,
        max_candidates: int = 20,
    ) -> None:
        super().__init__()
        self.max_heads = int(max_heads)
        self.query_projection = nn.Sequential(
            nn.Linear(8 * 7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.object_embedding = nn.Embedding(10, hidden_dim)
        nn.init.zeros_(self.object_embedding.weight)
        self.candidate_projection = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, 2)
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 3, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.set_encoder = nn.TransformerEncoder(set_layer, 2)
        self.rank_embedding = nn.Parameter(
            torch.randn(max_candidates, hidden_dim) * .02
        )
        self.head_embedding = nn.Parameter(
            torch.randn(max_heads, hidden_dim) * .05
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.time_embedding = nn.Parameter(torch.randn(20, hidden_dim) * .02)
        self.refinement = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 7, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    @staticmethod
    def _motion_features(trajectory: torch.Tensor) -> torch.Tensor:
        velocity = torch.cat([
            trajectory[..., :1, :] * 0,
            trajectory[..., 1:, :] - trajectory[..., :-1, :],
        ], dim=-2)
        acceleration = torch.cat([
            velocity[..., :1, :] * 0,
            velocity[..., 1:, :] - velocity[..., :-1, :],
        ], dim=-2)
        time = torch.linspace(
            0, 1, trajectory.shape[-2],
            device=trajectory.device, dtype=trajectory.dtype,
        )
        shape = [1] * (trajectory.ndim - 2) + [trajectory.shape[-2], 1]
        time = time.reshape(shape).expand(*trajectory.shape[:-1], 1)
        return torch.cat([trajectory, velocity, acceleration, time], dim=-1)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        candidates: torch.Tensor,
        output_heads: int,
        temperature: float = .25,
        object_type: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        if not 1 <= output_heads <= self.max_heads:
            raise ValueError("output_heads must lie in [1, max_heads]")
        if temperature <= 0:
            raise ValueError("mixer temperature must be positive")
        batch, count = candidates.shape[:2]
        if count > self.rank_embedding.shape[0]:
            raise ValueError("candidate pool exceeds mixer capacity")
        query_velocity = torch.cat([
            partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]
        ], dim=1)
        query_acceleration = torch.cat([
            query_velocity[:, :1] * 0,
            query_velocity[:, 1:] - query_velocity[:, :-1],
        ], dim=1)
        query = self.query_projection(torch.cat([
            partial, query_velocity, query_acceleration,
            seen[..., None].to(partial.dtype),
        ], dim=-1).reshape(batch, -1))
        if object_type is not None:
            query = query + self.object_embedding(
                object_type.long().clamp(0, 9)
            )
        token = self.candidate_projection(self._motion_features(candidates))
        token = self.temporal_encoder(
            token.reshape(batch * count, 20, -1)
        ).mean(dim=1).reshape(batch, count, -1)
        token = self.set_encoder(
            token + query[:, None] + self.rank_embedding[:count][None]
        )
        head = query[:, None] + self.head_embedding[:output_heads][None]
        token_expanded = token[:, None].expand(-1, output_heads, -1, -1)
        head_expanded = head[:, :, None].expand(-1, -1, count, -1)
        query_expanded = query[:, None, None].expand(
            -1, output_heads, count, -1
        )
        logits = self.score(torch.cat([
            token_expanded, head_expanded, query_expanded
        ], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits / temperature, dim=-1)
        mixed = torch.einsum("bhc,bctd->bhtd", weights, candidates)
        attended = torch.einsum("bhc,bcd->bhd", weights, token)
        motion = self._motion_features(mixed)
        context = torch.cat([
            attended[:, :, None].expand(-1, -1, 20, -1),
            head[:, :, None].expand(-1, -1, 20, -1),
            motion,
        ], dim=-1)
        output = self.refinement(context)
        result = mixed + output[..., :2] * torch.sigmoid(output[..., 2:3])
        return (result, weights) if return_weights else result


class MemoryConditionalModeDecoder(nn.Module):
    """Decode query-specific modes by attending to retrieved complete records."""

    def __init__(
        self, hidden_dim: int = 192, heads: int = 6, max_modes: int = 12
    ) -> None:
        super().__init__()
        self.max_modes = int(max_modes)
        self.query_projection = nn.Sequential(
            nn.Linear(8 * 7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.object_embedding = nn.Embedding(10, hidden_dim)
        nn.init.zeros_(self.object_embedding.weight)
        self.center_projection = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.reference_projection = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.reference_temporal_encoder = nn.TransformerEncoder(
            temporal_layer, 2
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 3, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.reference_set_encoder = nn.TransformerEncoder(set_layer, 2)
        self.rank_embedding = nn.Parameter(
            torch.randn(20, hidden_dim) * .02
        )
        self.weight_projection = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mode_embedding = nn.Parameter(
            torch.randn(max_modes, hidden_dim) * .02
        )
        self.time_embedding = nn.Parameter(torch.randn(20, hidden_dim) * .02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=.1, batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def _motion_features(trajectory: torch.Tensor) -> torch.Tensor:
        velocity = torch.cat(
            [trajectory[..., :1, :] * 0,
             trajectory[..., 1:, :] - trajectory[..., :-1, :]], dim=-2,
        )
        acceleration = torch.cat(
            [velocity[..., :1, :] * 0,
             velocity[..., 1:, :] - velocity[..., :-1, :]], dim=-2,
        )
        time = torch.linspace(
            0, 1, trajectory.shape[-2],
            device=trajectory.device, dtype=trajectory.dtype,
        )
        shape = [1] * (trajectory.ndim - 2) + [trajectory.shape[-2], 1]
        time = time.reshape(shape).expand(*trajectory.shape[:-1], 1)
        return torch.cat([trajectory, velocity, acceleration, time], dim=-1)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        center: torch.Tensor,
        references: torch.Tensor,
        modes: int,
        memory_weights: Optional[torch.Tensor] = None,
        object_type: Optional[torch.Tensor] = None,
        retrieval_init_strength: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 1 <= modes <= self.max_modes:
            raise ValueError("modes must lie in [1, max_modes]")
        batch, records = references.shape[:2]
        query_velocity = torch.cat(
            [partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        query_acceleration = torch.cat(
            [query_velocity[:, :1] * 0,
             query_velocity[:, 1:] - query_velocity[:, :-1]], dim=1,
        )
        query = self.query_projection(torch.cat(
            [partial, query_velocity, query_acceleration,
             seen[..., None].to(partial.dtype)], dim=-1,
        ).reshape(batch, -1))
        if object_type is not None:
            query = query + self.object_embedding(
                object_type.long().clamp(0, 9)
            )

        reference_tokens = self.reference_projection(
            self._motion_features(references)
        )
        reference_tokens = self.reference_temporal_encoder(
            reference_tokens.reshape(batch * records, 20, -1)
        ).mean(dim=1).reshape(batch, records, -1)
        reference_tokens = (
            reference_tokens
            + query[:, None]
            + self.rank_embedding[:records][None]
        )
        if memory_weights is not None:
            reference_tokens = reference_tokens + self.weight_projection(
                memory_weights[:, :records, None].to(reference_tokens.dtype)
            )
        reference_tokens = self.reference_set_encoder(reference_tokens)
        mode = self.mode_embedding[:modes][None].expand(batch, -1, -1)
        attended, _ = self.cross_attention(
            mode + query[:, None], reference_tokens, reference_tokens
        )
        center_token = self.center_projection(
            self._motion_features(center)
        )
        context = torch.cat([
            center_token[:, None].expand(-1, modes, -1, -1),
            query[:, None, None].expand(-1, modes, 20, -1),
            attended[:, :, None].expand(-1, -1, 20, -1),
            mode[:, :, None] + self.time_embedding[None, None],
        ], dim=-1)
        output = self.decoder(context)
        # Start from rank-spread retrieved trajectories instead of forcing a
        # randomly initialized decoder to discover coverage from a zero
        # residual.  The learned branch refines these database hypotheses.
        reference_index = torch.linspace(
            0, records - 1, modes, device=references.device
        ).round().long()
        retrieval_residual = (
            references[:, reference_index] - center[:, None]
        )
        return (
            retrieval_init_strength * retrieval_residual + output[..., :2],
            torch.sigmoid(output[..., 2:3] + 5.0),
        )


class MemorySetCenterPredictor(nn.Module):
    """Predict a robust trajectory center from the complete retrieval set."""

    def __init__(self, hidden_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(8 * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(20 * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 40),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        observed = torch.cat(
            [partial, seen[..., None].to(partial.dtype)], dim=-1
        ).reshape(partial.shape[0], -1)
        query = self.query_encoder(observed)[:, None]
        candidates = self.candidate_encoder(
            references.reshape(references.shape[0], references.shape[1], -1)
        )
        attended, _ = self.cross_attention(query, candidates, candidates)
        pooled = candidates.mean(dim=1)
        return self.output(
            torch.cat([attended[:, 0], pooled], dim=-1)
        ).reshape(-1, 20, 2)


class EnhancedMemoryCenterPredictor(nn.Module):
    """High-capacity temporal/set residual for the low-risk centre."""

    def __init__(self, hidden_dim: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(8 * 7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.object_embedding = nn.Embedding(10, hidden_dim)
        nn.init.zeros_(self.object_embedding.weight)
        self.reference_projection = nn.Sequential(
            nn.Linear(7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, 2)
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.set_encoder = nn.TransformerEncoder(set_layer, 2)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=.1, batch_first=True
        )
        self.time_embedding = nn.Parameter(torch.randn(20, hidden_dim) * .02)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        velocity = torch.cat(
            [partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        acceleration = torch.cat(
            [velocity[:, :1] * 0, velocity[:, 1:] - velocity[:, :-1]], dim=1
        )
        query = self.query_encoder(torch.cat(
            [partial, velocity, acceleration,
             seen[..., None].to(partial.dtype)], dim=-1,
        ).reshape(partial.shape[0], -1))
        if object_type is not None:
            query = query + self.object_embedding(
                object_type.long().clamp(0, 9)
            )
        ref_velocity = torch.cat(
            [references[:, :, :1] * 0,
             references[:, :, 1:] - references[:, :, :-1]], dim=2,
        )
        ref_acceleration = torch.cat(
            [ref_velocity[:, :, :1] * 0,
             ref_velocity[:, :, 1:] - ref_velocity[:, :, :-1]], dim=2,
        )
        time = torch.linspace(
            0, 1, 20, device=references.device, dtype=references.dtype
        )[None, None, :, None].expand(
            references.shape[0], references.shape[1], -1, -1
        )
        tokens = self.reference_projection(torch.cat(
            [references, ref_velocity, ref_acceleration, time], dim=-1
        ))
        batch, count = references.shape[:2]
        tokens = self.temporal_encoder(
            tokens.reshape(batch * count, 20, -1)
        ).mean(dim=1).reshape(batch, count, -1)
        tokens = self.set_encoder(tokens + query[:, None])
        attended, _ = self.cross_attention(
            query[:, None], tokens, tokens
        )
        pooled_mean = tokens.mean(dim=1)
        pooled_max = tokens.amax(dim=1)
        context = torch.cat(
            [query, attended[:, 0], pooled_mean, pooled_max], dim=-1
        )
        temporal = context[:, None].expand(-1, 20, -1)
        return self.decoder(temporal + torch.cat(
            [self.time_embedding[None]] * 4, dim=-1
        ))


class RetrievalConditionedCandidateRanker(nn.Module):
    """Rank forecast candidates against the retrieved complete-record set."""

    def __init__(self, hidden_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(8 * 5, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(40, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.memory_encoder = nn.Sequential(
            nn.Linear(41, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.object_embedding = nn.Embedding(10, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        candidates: torch.Tensor,
        memory_values: torch.Tensor,
        memory_weights: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        velocity = torch.cat(
            [partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        observed = torch.cat(
            [partial, velocity, seen[..., None].to(partial.dtype)], dim=-1
        ).reshape(partial.shape[0], -1)
        query = self.query_encoder(observed)
        if object_type is not None:
            query = query + self.object_embedding(
                object_type.long().clamp(0, 9)
            )
        candidate_tokens = self.candidate_encoder(
            candidates.reshape(candidates.shape[0], candidates.shape[1], -1)
        ) + query[:, None]
        memory_input = torch.cat(
            [
                memory_values.reshape(
                    memory_values.shape[0], memory_values.shape[1], -1
                ),
                memory_weights[..., None],
            ],
            dim=-1,
        )
        memory_tokens = self.memory_encoder(memory_input) + query[:, None]
        attended, _ = self.cross_attention(
            candidate_tokens, memory_tokens, memory_tokens
        )
        pooled = (memory_weights[..., None] * memory_tokens).sum(dim=1)
        features = torch.cat(
            [
                candidate_tokens,
                attended,
                candidate_tokens - attended,
                candidate_tokens * attended,
                pooled[:, None].expand(-1, candidates.shape[1], -1),
            ],
            dim=-1,
        )
        return self.score(features).squeeze(-1)


class RetrievalProjectionGate(nn.Module):
    """Calibrate candidate amplitude from the retrieved-record ensemble."""

    def __init__(self, hidden_dim: int = 192) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(8 * 7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.object_embedding = nn.Embedding(10, hidden_dim)
        nn.init.zeros_(self.object_embedding.weight)
        self.feature_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        center: torch.Tensor,
        candidates: torch.Tensor,
        memory_values: torch.Tensor,
        memory_weights: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        velocity = torch.cat(
            [partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        acceleration = torch.cat(
            [velocity[:, :1] * 0, velocity[:, 1:] - velocity[:, :-1]], dim=1
        )
        query = self.query_encoder(torch.cat([
            partial, velocity, acceleration,
            seen[..., None].to(partial.dtype),
        ], dim=-1).reshape(partial.shape[0], -1))
        if object_type is not None:
            query = query + self.object_embedding(
                object_type.long().clamp(0, 9)
            )

        branch = (candidates[:, :, 8:] - center[:, None, 8:]).flatten(2)
        memory = (
            memory_values[:, :, 8:] - center[:, None, 8:]
        ).flatten(2)
        branch_norm = branch.square().sum(dim=-1).sqrt().clamp_min(1e-6)
        memory_norm = memory.square().sum(dim=-1).sqrt().clamp_min(1e-6)
        dot = torch.einsum("bcd,bkd->bck", branch, memory)
        projection = (
            dot / branch_norm.square()[:, :, None]
        ).clamp(0.0, 1.0)
        cosine = dot / (
            branch_norm[:, :, None] * memory_norm[:, None, :]
        )
        distance = torch.linalg.vector_norm(
            branch[:, :, None] - memory[:, None], dim=-1
        )
        weights = memory_weights / memory_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        weighted_projection = (projection * weights[:, None]).sum(dim=-1)
        weighted_cosine = (cosine * weights[:, None]).sum(dim=-1)
        weighted_distance = (distance * weights[:, None]).sum(dim=-1)
        projection_std = (
            weights[:, None]
            * (projection - weighted_projection[:, :, None]).square()
        ).sum(dim=-1).sqrt()
        cosine_std = (
            weights[:, None]
            * (cosine - weighted_cosine[:, :, None]).square()
        ).sum(dim=-1).sqrt()
        memory_norm_mean = (memory_norm * weights).sum(dim=-1)
        features = torch.stack([
            weighted_projection,
            projection_std,
            projection.amax(dim=-1),
            projection.amin(dim=-1),
            weighted_cosine,
            cosine_std,
            cosine.amax(dim=-1),
            cosine.amin(dim=-1),
            weighted_distance,
            distance.amin(dim=-1),
            branch_norm,
            memory_norm_mean[:, None].expand_as(branch_norm),
        ], dim=-1)
        encoded = self.feature_encoder(features)
        correction = self.output(torch.cat([
            encoded, query[:, None].expand_as(encoded)
        ], dim=-1)).squeeze(-1)
        base = weighted_projection.clamp(0.02, 0.98)
        base_logit = torch.log(base) - torch.log1p(-base)
        return torch.sigmoid(base_logit + correction)


class DatabaseCenterDecoder(nn.Module):
    """Decode one low-risk trajectory from the complete retrieved record set."""

    def __init__(self, hidden_dim: int = 256, heads: int = 8) -> None:
        super().__init__()
        self.reference_projection = nn.Sequential(
            nn.Linear(10, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.query_projection = nn.Sequential(
            nn.Linear(8 * 7, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.object_embedding = nn.Embedding(10, hidden_dim)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        candidate_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 3, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=heads,
            dim_feedforward=hidden_dim * 4, dropout=.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, 3)
        self.candidate_encoder = nn.TransformerEncoder(candidate_layer, 2)
        self.time_query = nn.Parameter(torch.randn(1, 20, hidden_dim) * .02)
        self.decoder = nn.TransformerDecoder(decoder_layer, 3)
        self.output = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, candidates = references.shape[:2]
        velocity = torch.cat(
            [references[:, :, :1] * 0,
             references[:, :, 1:] - references[:, :, :-1]], dim=2,
        )
        acceleration = torch.cat(
            [velocity[:, :, :1] * 0,
             velocity[:, :, 1:] - velocity[:, :, :-1]], dim=2,
        )
        query_position = torch.zeros_like(references)
        query_position[:, :, :8] = partial[:, None]
        query_mask = torch.zeros(
            batch, candidates, 20, 1,
            device=references.device, dtype=references.dtype,
        )
        query_mask[:, :, :8, 0] = seen[:, None].to(references.dtype)
        time = torch.linspace(
            0, 1, 20, device=references.device, dtype=references.dtype
        )[None, None, :, None].expand(batch, candidates, -1, -1)
        tokens = self.reference_projection(torch.cat(
            [references, velocity, acceleration,
             query_position, query_mask, time], dim=-1,
        )).reshape(batch * candidates, 20, -1)

        query_velocity = torch.cat(
            [partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        query_acceleration = torch.cat(
            [query_velocity[:, :1] * 0,
             query_velocity[:, 1:] - query_velocity[:, :-1]], dim=1,
        )
        query = self.query_projection(torch.cat(
            [partial, query_velocity, query_acceleration,
             seen[..., None].to(partial.dtype)], dim=-1,
        ).reshape(batch, -1))
        if object_type is not None:
            query = query + self.object_embedding(
                object_type.long().clamp(0, 9)
            )
        tokens = tokens + query[:, None, :].expand(
            -1, candidates, -1
        ).reshape(batch * candidates, 1, -1)
        tokens = self.temporal_encoder(tokens).reshape(
            batch, candidates, 20, -1
        )
        candidate_tokens = self.candidate_encoder(
            tokens.mean(dim=2) + query[:, None]
        )
        temporal_consensus = tokens.mean(dim=1)
        memory = torch.cat(
            [query[:, None], candidate_tokens, temporal_consensus], dim=1
        )
        decoded = self.decoder(
            self.time_query.expand(batch, -1, -1) + temporal_consensus,
            memory,
        )
        return references.mean(dim=1) + self.output(decoded)


class VariationalAnchorFlow(nn.Module):
    """Memory anchor + conditional residual latent + flow refinement."""

    def __init__(
        self,
        hidden_dim: int = 128,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        latent_dim: int = 32,
        variational_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.retrieval_key_encoder = RetrievalKeyEncoder()
        self.coverage_risk_gate = nn.Sequential(
            nn.Linear(8 * 7, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 12),
        )
        nn.init.zeros_(self.coverage_risk_gate[-1].weight)
        nn.init.zeros_(self.coverage_risk_gate[-1].bias)
        context_input = 8 * (2 + 1) + 20 * 2
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input, variational_dim),
            nn.GELU(),
            nn.Linear(variational_dim, variational_dim),
            nn.GELU(),
        )
        self.candidate_score_head = nn.Sequential(
            nn.Linear(variational_dim, max(variational_dim // 2, 16)),
            nn.GELU(),
            nn.Linear(max(variational_dim // 2, 16), 1),
        )
        nn.init.zeros_(self.candidate_score_head[-1].weight)
        nn.init.zeros_(self.candidate_score_head[-1].bias)
        self.candidate_selector_network = nn.Sequential(
            nn.Linear(context_input, variational_dim * 2),
            nn.GELU(),
            nn.Linear(variational_dim * 2, variational_dim),
            nn.GELU(),
            nn.Linear(variational_dim, 1),
        )
        nn.init.zeros_(self.candidate_selector_network[-1].weight)
        nn.init.zeros_(self.candidate_selector_network[-1].bias)
        self.mantra_refine = nn.Sequential(
            nn.Linear(8 * 3, 128),
            nn.GELU(),
            nn.Linear(128, 40),
        )
        nn.init.zeros_(self.mantra_refine[-1].weight)
        nn.init.zeros_(self.mantra_refine[-1].bias)
        self.mean_trajectory_network = nn.Sequential(
            nn.Linear(context_input, variational_dim * 2),
            nn.GELU(),
            nn.Linear(variational_dim * 2, variational_dim * 2),
            nn.GELU(),
            nn.Linear(variational_dim * 2, 40),
        )
        nn.init.zeros_(self.mean_trajectory_network[-1].weight)
        nn.init.zeros_(self.mean_trajectory_network[-1].bias)
        self.mean_motion_refiner = MotionCandidateRefiner(
            hidden_dim=max(variational_dim * 2, 128), heads=4
        )
        self.memory_set_center = MemorySetCenterPredictor(
            hidden_dim=max(variational_dim * 2, 128), heads=4
        )
        self.enhanced_memory_center = EnhancedMemoryCenterPredictor()
        self.memory_candidate_ranker = RetrievalConditionedCandidateRanker(
            hidden_dim=max(variational_dim * 2, 128), heads=4
        )
        self.memory_ranker_v2 = False
        self.retrieval_projection_gate = RetrievalProjectionGate()
        self.database_center_decoder = DatabaseCenterDecoder()
        self.mode_trajectory_network = nn.Sequential(
            nn.Linear(context_input, variational_dim * 2),
            nn.GELU(),
            nn.Linear(variational_dim * 2, variational_dim * 2),
            nn.GELU(),
            nn.Linear(variational_dim * 2, 40),
        )
        nn.init.zeros_(self.mode_trajectory_network[-1].weight)
        nn.init.zeros_(self.mode_trajectory_network[-1].bias)
        self.candidate_refiner = MotionCandidateRefiner(
            hidden_dim=max(variational_dim * 2, 128), heads=4
        )
        self.coverage_candidate_refiner = MotionCandidateRefiner(
            hidden_dim=max(variational_dim * 2, 128), heads=4
        )
        self.set_candidate_refiner = SetAwareCandidateRefiner()
        self.set_candidate_ranker = SetAwareCandidateRanker(
            hidden_dim=max(variational_dim * 2, 128), heads=4
        )
        self.conditional_mode_decoder = ConditionalModeDecoder()
        self.conditional_mode_decoder_v2 = ConditionalModeDecoder(
            hidden_dim=256, heads=8
        )
        self.conditional_mode_decoder_v2_enabled = False
        self.dual_pool_decoders = nn.ModuleList(
            [ConditionalModeDecoder() for _ in range(3)]
        )
        self.dual_pool_selector = DualResponsibilitySelector(experts=4)
        self.dual_pool_role_constrained = False
        self.dual_pool_fixed_mixture = False
        self.dual_pool_fixed_coverage_count = 6
        self.dual_pool_fixed_coverage_index = -1
        self.dual_pool_fixed_coverage_slots = ()
        self.dual_pool_fixed_retrieval_mode = 0
        self.dual_pool_fixed_selector_mode = False
        self.dual_pool_fixed_rank_by_memory = False
        self.dual_pool_fixed_learned_ranker = False
        self.dual_pool_ranker_selector_ensemble = False
        self.dual_pool_ranker_selector_weight = 1.0
        self.dual_pool_ranker_mixture = False
        self.dual_pool_ranker_mixture_temperature = 0.2
        self.dual_pool_adaptive_mixture = False
        self.dual_pool_adaptive_min_count = 3
        self.dual_pool_adaptive_max_count = 10
        self.dual_pool_adaptive_threshold = 0.0
        self.dual_pool_adaptive_gate = False
        self.dual_pool_adaptive_gate_threshold = 0.0
        self.dual_pool_adaptive_multiclass_gate = False
        self.dual_pool_adaptive_count_values = (0, 1, 3, 6, 9, 12)
        self.mode_alignment_gate = ModeAlignmentGate()
        self.conditional_mode_mixer = ConditionalModeMixer()
        self.memory_conditional_mode_decoder = MemoryConditionalModeDecoder()
        self.memory_conditional_mode_decoder_v2 = MemoryConditionalModeDecoder(
            hidden_dim=256, heads=8
        )
        self.memory_conditional_mode_decoder_v2_enabled = False
        # A frozen snapshot of the coverage decoder is kept beside the
        # trainable risk-calibration decoder.  It prevents mean/tail-risk
        # optimisation from collapsing all hypotheses onto the centre.
        self.coverage_mode_decoder = ConditionalModeDecoder()
        self.posterior_encoder = nn.Sequential(
            nn.Linear(variational_dim + 40, variational_dim),
            nn.GELU(),
        )
        self.posterior_mu = nn.Linear(variational_dim, latent_dim)
        self.posterior_logvar = nn.Linear(variational_dim, latent_dim)
        self.prior_mu = nn.Linear(variational_dim, latent_dim)
        self.prior_logvar = nn.Linear(variational_dim, latent_dim)
        self.residual_decoder = nn.Sequential(
            nn.Linear(variational_dim + latent_dim, variational_dim),
            nn.GELU(),
            nn.Linear(variational_dim, variational_dim),
            nn.GELU(),
            nn.Linear(variational_dim, 40),
        )
        nn.init.zeros_(self.residual_decoder[-1].weight)
        nn.init.zeros_(self.residual_decoder[-1].bias)
        self.flow = JointFlowModel(
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            dropout=dropout,
            latent_dim=latent_dim,
        )

    def reset_refinement_heads(self) -> None:
        """Reset train-only proposal heads without perturbing the flow trunk."""
        for head in (
            self.mean_trajectory_network,
            self.mode_trajectory_network,
            self.candidate_selector_network,
            self.candidate_score_head,
            self.candidate_refiner,
            self.coverage_candidate_refiner,
            self.set_candidate_refiner,
            self.set_candidate_ranker,
            self.conditional_mode_decoder,
            self.conditional_mode_mixer,
            self.mean_motion_refiner,
            self.memory_set_center,
            self.database_center_decoder,
            self.coverage_risk_gate,
        ):
            for module in head.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        for head in (
            self.mean_trajectory_network,
            self.mode_trajectory_network,
            self.candidate_selector_network,
            self.candidate_score_head,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.candidate_refiner.output.weight)
        nn.init.zeros_(self.candidate_refiner.output.bias)
        nn.init.zeros_(self.coverage_candidate_refiner.output.weight)
        nn.init.zeros_(self.coverage_candidate_refiner.output.bias)
        nn.init.zeros_(self.set_candidate_refiner.output.weight)
        nn.init.zeros_(self.set_candidate_refiner.output.bias)
        nn.init.zeros_(self.set_candidate_ranker.output.weight)
        nn.init.zeros_(self.set_candidate_ranker.output.bias)
        nn.init.zeros_(self.conditional_mode_decoder.output.weight)
        nn.init.zeros_(self.conditional_mode_decoder.output.bias)
        nn.init.zeros_(self.mean_motion_refiner.output.weight)
        nn.init.zeros_(self.mean_motion_refiner.output.bias)
        nn.init.zeros_(self.memory_set_center.output[-1].weight)
        nn.init.zeros_(self.memory_set_center.output[-1].bias)
        nn.init.zeros_(self.database_center_decoder.output.weight)
        nn.init.zeros_(self.database_center_decoder.output.bias)
        nn.init.zeros_(self.coverage_risk_gate[-1].weight)
        nn.init.zeros_(self.coverage_risk_gate[-1].bias)

    def encode_retrieval_key(self, key: torch.Tensor) -> torch.Tensor:
        return self.retrieval_key_encoder(key)

    def predict_coverage_risk_gate(
        self, partial: torch.Tensor, seen: torch.Tensor, modes: int = 12
    ) -> torch.Tensor:
        velocity = torch.cat(
            [partial[:, :1] * 0, partial[:, 1:] - partial[:, :-1]], dim=1
        )
        acceleration = torch.cat(
            [velocity[:, :1] * 0, velocity[:, 1:] - velocity[:, :-1]], dim=1
        )
        feature = torch.cat(
            [partial, velocity, acceleration, seen[..., None].to(partial.dtype)],
            dim=-1,
        ).reshape(partial.shape[0], -1)
        return torch.sigmoid(self.coverage_risk_gate(feature)[:, :modes])

    def predict_enhanced_memory_center(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.enhanced_memory_center(
            partial, seen, references, object_type=object_type
        )

    @staticmethod
    def free_mask(seen: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.ones(seen.shape[0], 20, 1, device=seen.device, dtype=dtype)
        mask[:, :8, 0] = (~seen).to(dtype)
        return mask

    def encode_condition(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        observed = torch.cat(
            [partial, seen[..., None].to(partial.dtype)], dim=-1
        ).reshape(partial.shape[0], -1)
        return self.context_encoder(
            torch.cat([observed, reference.reshape(reference.shape[0], -1)], dim=-1)
        )

    def score_references(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Predict train-supervised future risk for each retrieved candidate."""
        batch, candidates = references.shape[:2]
        partial_flat = partial[:, None].expand(-1, candidates, -1, -1).reshape(
            batch * candidates, 8, 2
        )
        seen_flat = seen[:, None].expand(-1, candidates, -1).reshape(
            batch * candidates, 8
        )
        reference_flat = references.reshape(batch * candidates, 20, 2)
        # Keep reranker supervision from perturbing the transport condition
        # encoder; this isolates candidate-order learning from the flow model.
        context = self.encode_condition(
            partial_flat, seen_flat, reference_flat
        ).detach()
        risk = F.softplus(self.candidate_score_head(context).squeeze(-1))
        return risk.reshape(batch, candidates)

    def select_reference_logits(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Score retrieved intentions from visible history only."""
        batch, candidates = references.shape[:2]
        partial_flat = partial[:, None].expand(-1, candidates, -1, -1).reshape(
            batch * candidates, 8, 2
        )
        seen_flat = seen[:, None].expand(-1, candidates, -1).reshape(
            batch * candidates, 8
        )
        observed = torch.cat(
            [partial_flat, seen_flat[..., None].to(partial.dtype)], dim=-1
        ).reshape(batch * candidates, -1)
        features = torch.cat(
            [observed, references.reshape(batch * candidates, -1)], dim=-1
        )
        return self.candidate_selector_network(features).reshape(batch, candidates)

    def select_set_reference_logits(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.set_candidate_ranker(
            partial, seen, references, object_type=object_type
        )

    def score_candidates_with_memory(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        candidates: torch.Tensor,
        memory_values: torch.Tensor,
        memory_weights: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.memory_ranker_v2:
            # The set-aware temporal ranker models candidate motion and
            # cross-candidate competition directly.  The candidates already
            # encode the retrieved records, so this complements the flattened
            # retrieval-conditioned scorer used by the legacy branch.
            return -self.set_candidate_ranker(
                partial, seen, candidates, object_type=object_type
            )
        return self.memory_candidate_ranker(
            partial,
            seen,
            candidates,
            memory_values,
            memory_weights,
            object_type=object_type,
        )

    def predict_retrieval_projection_gate(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        center: torch.Tensor,
        candidates: torch.Tensor,
        memory_values: torch.Tensor,
        memory_weights: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.retrieval_projection_gate(
            partial,
            seen,
            center,
            candidates,
            memory_values,
            memory_weights,
            object_type=object_type,
        )

    def predict_conditional_modes(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        center: torch.Tensor,
        modes: int,
        object_type: Optional[torch.Tensor] = None,
        center_copies: int = 1,
        teacher_modes: int = 0,
        split_decoder: bool = False,
        coverage_modes: int = 0,
        memory_values: Optional[torch.Tensor] = None,
        memory_weights: Optional[torch.Tensor] = None,
        memory_conditional: bool = False,
        memory_retrieval_init_strength: float = 1.0,
    ) -> torch.Tensor:
        decoder = (
            self.conditional_mode_decoder_v2
            if self.conditional_mode_decoder_v2_enabled
            else self.conditional_mode_decoder
        )
        if not 1 <= center_copies < modes:
            raise ValueError("center_copies must lie in [1, modes)")
        if not 0 <= teacher_modes <= modes - center_copies:
            raise ValueError(
                "teacher_modes must lie in [0, modes - center_copies]"
            )
        if memory_conditional:
            if memory_values is None:
                raise ValueError(
                    "memory-conditional decoding requires memory_values"
                )
            if split_decoder or teacher_modes:
                raise ValueError(
                    "memory-conditional decoding does not combine with "
                    "split or teacher modes"
                )
            decoded_modes = modes - center_copies
            memory_decoder = (
                self.memory_conditional_mode_decoder_v2
                if self.memory_conditional_mode_decoder_v2_enabled
                else self.memory_conditional_mode_decoder
            )
            residual, gate = memory_decoder(
                partial, seen, center, memory_values, decoded_modes,
                memory_weights=memory_weights,
                object_type=object_type,
                retrieval_init_strength=memory_retrieval_init_strength,
            )
            candidates = center[:, None] + residual * gate * self.free_mask(
                seen, residual.dtype
            )[:, None]
            return torch.cat([
                center[:, None].expand(-1, center_copies, -1, -1),
                candidates,
            ], dim=1)
        if split_decoder:
            if teacher_modes or not 0 < coverage_modes <= modes - center_copies:
                raise ValueError(
                    "split decoding requires teacher_modes=0 and a valid "
                    "positive coverage_modes"
                )
            risk_modes = modes - center_copies - coverage_modes
            if risk_modes:
                risk_residual, risk_gate = decoder(
                    partial, seen, center,
                    decoder.max_modes,
                    object_type=object_type,
                )
                risk_pool = center[:, None] + risk_residual * risk_gate \
                    * self.free_mask(seen, risk_residual.dtype)[:, None]
                risk_candidates = risk_pool[:, 1:1 + risk_modes]
            else:
                risk_candidates = center[:, None][:, :0]
            coverage_residual, coverage_gate = self.coverage_mode_decoder(
                partial, seen, center,
                self.coverage_mode_decoder.max_modes,
                object_type=object_type,
            )
            coverage_pool = center[:, None] + coverage_residual * coverage_gate \
                * self.free_mask(seen, coverage_residual.dtype)[:, None]
            # Mode zero was replaced by the explicit centre in the source
            # checkpoints.  Evenly spaced non-zero modes provide a stable,
            # deterministic initialization before directional specialization.
            coverage_index = torch.linspace(
                1, self.coverage_mode_decoder.max_modes - 1, coverage_modes,
                device=center.device,
            ).round().long()
            return torch.cat([
                center[:, None].expand(-1, center_copies, -1, -1),
                risk_candidates,
                coverage_pool[:, coverage_index],
            ], dim=1)
        if teacher_modes == 0:
            # Preserve the original mode-index semantics used by all v43--v48
            # checkpoints: decode all modes, then replace the leading slots by
            # exact centre copies.  This matters because learned mode embeddings
            # are not permutation invariant after training.
            residual, gate = decoder(
                partial, seen, center, modes, object_type=object_type
            )
            candidates = center[:, None] + residual * gate * self.free_mask(
                seen, residual.dtype
            )[:, None]
            return torch.cat([
                center[:, None].expand(-1, center_copies, -1, -1),
                candidates[:, center_copies:],
            ], dim=1)
        calibrated_modes = modes - center_copies - teacher_modes
        if calibrated_modes:
            residual, gate = decoder(
                partial, seen, center, calibrated_modes, object_type=object_type
            )
            calibrated = center[:, None] + residual * gate * self.free_mask(
                seen, residual.dtype
            )[:, None]
        else:
            calibrated = center[:, None][:, :0]
        pieces = [center[:, None].expand(-1, center_copies, -1, -1)]
        if teacher_modes:
            teacher_pool = self.predict_frozen_coverage_pool(
                partial, seen, center, object_type=object_type
            )
            # The ranker is trained only with training trajectories.  At
            # validation/test time it sees the incomplete query and frozen
            # candidates, never the target.
            teacher_logits = self.select_set_reference_logits(
                partial, seen, teacher_pool, object_type=object_type
            )
            teacher_index = teacher_logits.topk(
                teacher_modes, dim=1, largest=True, sorted=True
            ).indices
            rows = torch.arange(center.shape[0], device=center.device)
            pieces.append(teacher_pool[rows[:, None], teacher_index])
        if calibrated_modes:
            pieces.append(calibrated)
        return torch.cat(pieces, dim=1)

    def predict_dual_pool(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        center: torch.Tensor,
        modes: int,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode the frozen coverage branch and three complementary experts."""
        pools = []
        for decoder in [self.conditional_mode_decoder, *self.dual_pool_decoders]:
            residual, gate = decoder(
                partial, seen, center, modes, object_type=object_type
            )
            candidates = center[:, None] + residual * gate * self.free_mask(
                seen, residual.dtype
            )[:, None]
            pools.append(torch.cat([center[:, None], candidates[:, 1:]], dim=1))
        return torch.cat(pools, dim=1)

    def select_dual_pool(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        pool: torch.Tensor,
        output_modes: int,
        memory_values: torch.Tensor,
        memory_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Reserve one slot for repair and fill the rest by forecast ranking."""
        if self.dual_pool_fixed_mixture:
            modes_per_expert = pool.shape[1] // self.dual_pool_selector.experts
            if self.dual_pool_adaptive_gate:
                gate_logit = self.dual_pool_selector.forward_mixture_gate(
                    partial, seen, memory_values, memory_weights
                )
                if self.dual_pool_adaptive_multiclass_gate:
                    selected_count = torch.as_tensor(
                        self.dual_pool_adaptive_count_values,
                        device=pool.device,
                    )[gate_logit.argmax(dim=-1)]
                else:
                    use_max = (
                        gate_logit > self.dual_pool_adaptive_gate_threshold
                    )
                coverage = pool[:, :modes_per_expert]
                mean_pool = pool[:, -modes_per_expert:]
                rows = []
                for row in range(pool.shape[0]):
                    if self.dual_pool_adaptive_multiclass_gate:
                        count = int(selected_count[row])
                    else:
                        count = (
                            self.dual_pool_adaptive_max_count
                            if bool(use_max[row])
                            else self.dual_pool_adaptive_min_count
                        )
                    rows.append(
                        torch.cat(
                            [
                                coverage[row, :count],
                                mean_pool[row, : output_modes - count],
                            ],
                            dim=0,
                        )
                    )
                return torch.stack(rows, dim=0)
            if self.dual_pool_adaptive_mixture:
                coverage = pool[:, :modes_per_expert]
                mean_pool = pool[:, -modes_per_expert:]
                weights = memory_weights / memory_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                memory_center = (
                    memory_values * weights[:, :, None, None]
                ).sum(dim=1)
                coverage_score = torch.linalg.vector_norm(
                    coverage[:, :, 8:] - memory_center[:, None, 8:], dim=-1
                ).mean(dim=-1).min(dim=1).values
                mean_score = torch.linalg.vector_norm(
                    mean_pool[:, :, 8:] - memory_center[:, None, 8:], dim=-1
                ).mean(dim=-1).min(dim=1).values
                use_max = (
                    coverage_score + self.dual_pool_adaptive_threshold
                    < mean_score
                )
                rows = []
                for row in range(pool.shape[0]):
                    count = (
                        self.dual_pool_adaptive_max_count
                        if bool(use_max[row])
                        else self.dual_pool_adaptive_min_count
                    )
                    rows.append(
                        torch.cat(
                            [
                                coverage[row, :count],
                                mean_pool[row, : output_modes - count],
                            ],
                            dim=0,
                        )
                    )
                return torch.stack(rows, dim=0)
            coverage_count = max(
                0, min(int(self.dual_pool_fixed_coverage_count), output_modes)
            )
            mean_count = output_modes - coverage_count
            pieces = []
            if coverage_count:
                coverage = pool[:, :modes_per_expert]
                if self.dual_pool_fixed_selector_mode:
                    (
                        _imputation_logits,
                        _joint_logits,
                        minade_logits,
                        _minfde_logits,
                        _mean_logits,
                    ) = self.dual_pool_selector(
                        partial,
                        seen,
                        pool,
                        memory_values,
                        memory_weights,
                    )
                    index = minade_logits[:, :modes_per_expert].topk(
                        coverage_count,
                        dim=1,
                        largest=True,
                        sorted=True,
                    ).indices
                    rows = torch.arange(pool.shape[0], device=pool.device)[:, None]
                    coverage = coverage[rows, index]
                elif self.dual_pool_fixed_retrieval_mode:
                    weights = memory_weights / memory_weights.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-8)
                    memory_center = (
                        memory_values * weights[:, :, None, None]
                    ).sum(dim=1)
                    if self.dual_pool_fixed_retrieval_mode == 2:
                        index = memory_weights.argmax(dim=1)
                        rows = torch.arange(pool.shape[0], device=pool.device)
                        retrieval_mode = memory_values[rows, index]
                    else:
                        retrieval_mode = memory_center
                    coverage = retrieval_mode[:, None].expand(
                        -1, coverage_count, -1, -1
                    )
                elif self.dual_pool_fixed_learned_ranker:
                    score = self.score_candidates_with_memory(
                        partial,
                        seen,
                        coverage,
                        memory_values,
                        memory_weights,
                    )
                    if self.dual_pool_ranker_selector_ensemble:
                        selector_logits = self.dual_pool_selector(
                            partial,
                            seen,
                            pool,
                            memory_values,
                            memory_weights,
                        )[2][:, :modes_per_expert]
                        def normalized(value):
                            return (value - value.mean(dim=1, keepdim=True)) / (
                                value.std(dim=1, keepdim=True).clamp_min(1e-4)
                            )
                        score = normalized(score) - (
                            self.dual_pool_ranker_selector_weight
                            * normalized(selector_logits)
                        )
                    order = score.argsort(dim=1)
                    rows = torch.arange(
                        pool.shape[0], device=pool.device
                    )[:, None]
                    if self.dual_pool_ranker_mixture:
                        mixtures = []
                        for prefix in range(1, coverage_count + 1):
                            index = order[:, :prefix]
                            selected_score = score[rows, index]
                            weight = torch.softmax(
                                -selected_score
                                / self.dual_pool_ranker_mixture_temperature,
                                dim=1,
                            )
                            mixtures.append(
                                (
                                    coverage[rows, index]
                                    * weight[:, :, None, None]
                                ).sum(dim=1)
                            )
                        coverage = torch.stack(mixtures, dim=1)
                    else:
                        index = order[:, :coverage_count]
                        coverage = coverage[rows, index]
                elif self.dual_pool_fixed_rank_by_memory:
                    weights = memory_weights / memory_weights.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-8)
                    memory_center = (
                        memory_values * weights[:, :, None, None]
                    ).sum(dim=1)
                    score = torch.linalg.vector_norm(
                        coverage[:, :, 8:] - memory_center[:, None, 8:], dim=-1
                    ).mean(dim=-1)
                    index = score.topk(
                        coverage_count, dim=1, largest=False, sorted=True
                    ).indices
                    rows = torch.arange(pool.shape[0], device=pool.device)[:, None]
                    coverage = coverage[rows, index]
                elif self.dual_pool_fixed_coverage_slots:
                    slot_values = tuple(
                        int(value)
                        for value in self.dual_pool_fixed_coverage_slots
                    )[:coverage_count]
                    if len(slot_values) != coverage_count:
                        raise ValueError(
                            "fixed coverage slots must match coverage count"
                        )
                    index = torch.as_tensor(
                        slot_values,
                        device=pool.device,
                        dtype=torch.long,
                    )[None].expand(pool.shape[0], -1)
                    rows = torch.arange(
                        pool.shape[0], device=pool.device
                    )[:, None]
                    coverage = coverage[rows, index]
                elif self.dual_pool_fixed_coverage_index >= 0:
                    index = torch.full(
                        (pool.shape[0], coverage_count),
                        int(self.dual_pool_fixed_coverage_index),
                        device=pool.device,
                        dtype=torch.long,
                    )
                    rows = torch.arange(pool.shape[0], device=pool.device)[:, None]
                    coverage = coverage[rows, index]
                else:
                    coverage = coverage[:, :coverage_count]
                pieces.append(coverage)
            if mean_count:
                mean_pool = pool[:, -modes_per_expert:]
                pieces.append(mean_pool[:, :mean_count])
            return torch.cat(pieces, dim=1)
        (
            imputation_logits,
            joint_logits,
            minade_logits,
            minfde_logits,
            mean_logits,
        ) = self.dual_pool_selector(
            partial, seen, pool, memory_values, memory_weights
        )
        imputation_index = imputation_logits.argmax(dim=1, keepdim=True)
        if self.dual_pool_role_constrained:
            modes_per_expert = pool.shape[1] // 4
            coverage_slice = slice(0, modes_per_expert)
            mean_slice = slice(3 * modes_per_expert, 4 * modes_per_expert)
            joint_index = joint_logits[:, coverage_slice].argmax(
                dim=1, keepdim=True
            )
            minade_index = minade_logits[:, coverage_slice].argmax(
                dim=1, keepdim=True
            )
            minfde_index = minfde_logits[:, coverage_slice].argmax(
                dim=1, keepdim=True
            )
            mean_count = output_modes - 3
            mean_index = mean_logits[:, mean_slice].topk(
                mean_count, dim=1, largest=True, sorted=True
            ).indices + 3 * modes_per_expert
        else:
            joint_index = joint_logits.argmax(dim=1, keepdim=True)
            minade_index = minade_logits.argmax(dim=1, keepdim=True)
            minfde_index = minfde_logits.argmax(dim=1, keepdim=True)
            mean_count = output_modes - 4
            mean_index = mean_logits.topk(
                mean_count, dim=1, largest=True, sorted=True
            ).indices
        index = torch.cat(
            [joint_index, minade_index, minfde_index, mean_index], dim=1
        )
        rows = torch.arange(pool.shape[0], device=pool.device)[:, None]
        selected = pool[rows, index]
        repair = pool[rows, imputation_index].clone()
        # History and future are conditionally generated components.  Keep the
        # best repair history while attaching the top low-risk future so that
        # the repair responsibility cannot inflate mean forecasting error.
        repair[:, :, 8:] = pool[rows, mean_index[:, :1], 8:]
        if self.dual_pool_role_constrained:
            return selected
        return torch.cat([repair, selected], dim=1)

    def align_coverage_modes(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        coverage: torch.Tensor,
        low_risk: torch.Tensor,
        memory_values: torch.Tensor,
        memory_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match modes geometrically and apply a target-free learned gate."""
        coverage_endpoint = coverage[:, :, -1]
        low_risk_endpoint = low_risk[:, :, -1]
        match = torch.linalg.vector_norm(
            coverage_endpoint[:, :, None] - low_risk_endpoint[:, None], dim=-1
        ).argmin(dim=-1)
        rows = torch.arange(coverage.shape[0], device=coverage.device)[:, None]
        aligned_mean = low_risk[rows, match]
        if self.mode_alignment_gate.retrieval_specialists:
            specialist_count = min(
                max(1, self.mode_alignment_gate.preserve_count),
                coverage.shape[1],
                memory_values.shape[1],
            )
            memory_index = memory_weights.topk(
                specialist_count, dim=1, largest=True, sorted=True
            ).indices
            specialists = memory_values[rows[:, :specialist_count], memory_index]
            aligned = aligned_mean.clone()
            aligned[:, :specialist_count] = specialists
            alpha = torch.zeros(
                coverage.shape[:2], device=coverage.device, dtype=coverage.dtype
            )
            alpha[:, :specialist_count] = 1.0
            return aligned, alpha, aligned_mean
        alpha = self.mode_alignment_gate(
            partial,
            seen,
            coverage,
            aligned_mean,
            memory_values,
            memory_weights,
        )
        # Retain target-free gate inputs for validation diagnostics.  These
        # detached tensors are overwritten every batch and never enter loss.
        self.mode_alignment_gate.last_alpha = alpha.detach()
        self.mode_alignment_gate.last_coverage = coverage.detach()
        self.mode_alignment_gate.last_aligned_mean = aligned_mean.detach()
        aligned = coverage.clone()
        aligned[:, :, 8:] = (
            aligned_mean[:, :, 8:]
            + alpha[:, :, None, None]
            * (coverage[:, :, 8:] - aligned_mean[:, :, 8:])
        )
        return aligned, alpha, aligned_mean

    def mix_conditional_modes(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        candidates: torch.Tensor,
        output_heads: int,
        temperature: float = .25,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.conditional_mode_mixer(
            partial,
            seen,
            candidates,
            output_heads,
            temperature=temperature,
            object_type=object_type,
        )

    def predict_frozen_coverage_pool(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        center: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return all hypotheses from the frozen high-coverage teacher."""
        self.coverage_mode_decoder.eval()
        with torch.no_grad():
            residual, gate = self.coverage_mode_decoder(
                partial, seen, center, self.coverage_mode_decoder.max_modes,
                object_type=object_type,
            )
            return center[:, None] + residual * gate * self.free_mask(
                seen, residual.dtype
            )[:, None]

    def initialize_coverage_mode_decoder(self) -> None:
        """Snapshot the current multimodal decoder as a frozen coverage branch."""
        self.coverage_mode_decoder.load_state_dict(
            self.conditional_mode_decoder.state_dict(), strict=True
        )
        self.coverage_mode_decoder.eval()
        for parameter in self.coverage_mode_decoder.parameters():
            parameter.requires_grad = False

    def predict_mean_trajectory(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a low-risk conditional trajectory around memory consensus."""
        observed = torch.cat(
            [partial, seen[..., None].to(partial.dtype)], dim=-1
        ).reshape(partial.shape[0], -1)
        features = torch.cat(
            [observed, reference.reshape(reference.shape[0], -1)], dim=-1
        )
        residual = self.mean_trajectory_network(features).reshape(-1, 20, 2)
        return reference + residual * self.free_mask(seen, residual.dtype)

    def predict_motion_mean_trajectory(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        reference: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict a temporally structured conditional center trajectory."""
        residual, gate = self.mean_motion_refiner(
            partial, seen, reference, object_type=object_type
        )
        return reference + residual * gate * self.free_mask(
            seen, residual.dtype
        )

    def predict_memory_set_center(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend to all retrieved trajectories before predicting a center."""
        center = references.mean(dim=1)
        residual = self.memory_set_center(partial, seen, references)
        return center + residual * self.free_mask(seen, residual.dtype)

    def predict_set_aware_center(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict a robust center after temporal and cross-candidate attention."""
        refined = self.refine_candidate_set(
            partial, seen, references, object_type=object_type
        )
        return refined.mean(dim=1)

    def predict_database_center(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode a deterministic center from query and retrieved records."""
        center = self.database_center_decoder(
            partial, seen, references, object_type=object_type
        )
        return center

    def predict_mode_trajectory(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Refine likely retrieval modes without collapsing all candidates."""
        observed = torch.cat(
            [partial, seen[..., None].to(partial.dtype)], dim=-1
        ).reshape(partial.shape[0], -1)
        features = torch.cat(
            [observed, reference.reshape(reference.shape[0], -1)], dim=-1
        )
        residual = self.mode_trajectory_network(features).reshape(-1, 20, 2)
        return reference + residual * self.free_mask(seen, residual.dtype)

    def refine_candidate_trajectory(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Refine each retrieved mode without replacing its identity."""
        residual, gate = self.candidate_refiner(partial, seen, reference)
        return reference + residual * gate * self.free_mask(seen, residual.dtype)

    def refine_coverage_candidate_trajectory(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the separately trained low-oracle-risk coverage refiner."""
        residual, gate = self.coverage_candidate_refiner(partial, seen, reference)
        return reference + residual * gate * self.free_mask(seen, residual.dtype)

    def refine_candidate_set(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
        object_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Jointly refine a fixed candidate set while retaining candidate identity."""
        residual, gate = self.set_candidate_refiner(
            partial, seen, references, object_type=object_type
        )
        free = self.free_mask(seen, residual.dtype)[:, None]
        return references + residual * gate * free

    def training_residual(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        reference: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_condition(partial, seen, reference)
        posterior = self.posterior_encoder(
            torch.cat([context, target.reshape(target.shape[0], -1)], dim=-1)
        )
        mu_q = self.posterior_mu(posterior)
        logvar_q = self.posterior_logvar(posterior).clamp(-8, 8)
        mu_p = self.prior_mu(context)
        logvar_p = self.prior_logvar(context).clamp(-8, 8)
        latent = mu_q + torch.randn_like(mu_q) * torch.exp(0.5 * logvar_q)
        residual = self.residual_decoder(
            torch.cat([context, latent], dim=-1)
        ).reshape(-1, 20, 2)
        residual = residual * self.free_mask(seen, residual.dtype)
        kl = 0.5 * (
            logvar_p
            - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).square()) / logvar_p.exp()
            - 1
        ).sum(dim=-1)
        return residual, latent, kl

    def training_residual_pair(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        reference: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return posterior and prior residual samples for aligned training.

        The posterior sample supervises the variational residual decoder.  The
        independently sampled prior residual is available for the flow path,
        matching the latent distribution used at inference.
        """
        context = self.encode_condition(partial, seen, reference)
        posterior = self.posterior_encoder(
            torch.cat([context, target.reshape(target.shape[0], -1)], dim=-1)
        )
        mu_q = self.posterior_mu(posterior)
        logvar_q = self.posterior_logvar(posterior).clamp(-8, 8)
        mu_p = self.prior_mu(context)
        logvar_p = self.prior_logvar(context).clamp(-8, 8)
        latent_q = mu_q + torch.randn_like(mu_q) * torch.exp(0.5 * logvar_q)
        latent_p = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * logvar_p)
        free = self.free_mask(seen, context.dtype)
        residual_q = self.residual_decoder(
            torch.cat([context, latent_q], dim=-1)
        ).reshape(-1, 20, 2) * free
        residual_p = self.residual_decoder(
            torch.cat([context, latent_p], dim=-1)
        ).reshape(-1, 20, 2) * free
        residual_mean = self.residual_decoder(
            torch.cat([context, mu_p], dim=-1)
        ).reshape(-1, 20, 2) * free
        kl = 0.5 * (
            logvar_p
            - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).square()) / logvar_p.exp()
            - 1
        ).sum(dim=-1)
        return residual_q, latent_q, residual_p, latent_p, residual_mean, mu_p, kl

    @torch.no_grad()
    def sample_prior_residual(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        references: torch.Tensor,
        sampling: str = "sample",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, samples = references.shape[:2]
        partial_flat = partial[:, None].expand(-1, samples, -1, -1).reshape(
            batch * samples, 8, 2
        )
        seen_flat = seen[:, None].expand(-1, samples, -1).reshape(
            batch * samples, 8
        )
        reference_flat = references.reshape(batch * samples, 20, 2)
        context = self.encode_condition(partial_flat, seen_flat, reference_flat)
        mu = self.prior_mu(context)
        logvar = self.prior_logvar(context).clamp(-8, 8)
        if sampling == "sample":
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        elif sampling == "mean":
            latent = mu
        else:
            raise ValueError(f"unknown prior sampling mode: {sampling}")
        residual = self.residual_decoder(
            torch.cat([context, latent], dim=-1)
        ).reshape(batch * samples, 20, 2)
        residual = residual * self.free_mask(seen_flat, residual.dtype)
        return (
            residual.reshape(batch, samples, 20, 2),
            latent.reshape(batch, samples, self.latent_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        partial: torch.Tensor,
        seen: torch.Tensor,
        reference: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        return self.flow(state, time, partial, seen, reference, latent)


class JointCVAE(nn.Module):
    """Same-data, same-memory non-flow baseline."""

    def __init__(self, hidden_dim: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        context_input = 8 * (2 + 1) + 20 * 2
        self.context = nn.Sequential(
            nn.Linear(context_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + 40, hidden_dim),
            nn.GELU(),
        )
        self.posterior_mu = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 40),
        )

    def encode_context(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        memory_context: torch.Tensor,
    ) -> torch.Tensor:
        observed = torch.cat([partial, seen[..., None].float()], dim=-1).reshape(partial.shape[0], -1)
        return self.context(torch.cat([observed, memory_context.reshape(partial.shape[0], -1)], dim=-1))

    def forward(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        memory_context: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_context(partial, seen, memory_context)
        posterior = self.posterior(torch.cat([context, target.reshape(target.shape[0], -1)], dim=-1))
        mu_q = self.posterior_mu(posterior)
        logvar_q = self.posterior_logvar(posterior).clamp(-8, 8)
        mu_p = self.prior_mu(context)
        logvar_p = self.prior_logvar(context).clamp(-8, 8)
        z = mu_q + torch.randn_like(mu_q) * torch.exp(0.5 * logvar_q)
        prediction = self.decoder(torch.cat([context, z], dim=-1)).reshape(-1, 20, 2)
        kl = 0.5 * (
            logvar_p
            - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).square()) / logvar_p.exp()
            - 1
        ).sum(dim=-1)
        return prediction, kl

    @torch.no_grad()
    def sample(
        self,
        partial: torch.Tensor,
        seen: torch.Tensor,
        memory_context: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        context = self.encode_context(partial, seen, memory_context)
        mu = self.prior_mu(context)
        logvar = self.prior_logvar(context).clamp(-8, 8)
        batch, latent = mu.shape
        noise = torch.randn(batch, num_samples, latent, device=mu.device)
        z = mu[:, None, :] + noise * torch.exp(0.5 * logvar)[:, None, :]
        context_rep = context[:, None, :].expand(-1, num_samples, -1)
        decoded = self.decoder(
            torch.cat([context_rep, z], dim=-1).reshape(batch * num_samples, -1)
        )
        return decoded.reshape(batch, num_samples, 20, 2)


def clamp_observed(
    state: torch.Tensor,
    target: torch.Tensor,
    seen: torch.Tensor,
) -> torch.Tensor:
    result = state.clone()
    result[:, :8] = torch.where(seen[..., None], target[:, :8], result[:, :8])
    return result


def source_state(
    target: torch.Tensor,
    seen: torch.Tensor,
    anchors: Optional[torch.Tensor],
    source_noise: float,
) -> torch.Tensor:
    if anchors is None:
        source = torch.randn_like(target)
    else:
        source = anchors + source_noise * torch.randn_like(anchors)
    return clamp_observed(source, target, seen)


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    unknown: torch.Tensor,
    step_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    error = (prediction - target).square().sum(dim=-1)
    weights = unknown.float()
    if step_weights is not None:
        weights = weights * step_weights.to(device=weights.device, dtype=weights.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1)
