from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import JointDataBundle, load_bundle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def interpolate_history(partial: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    result = partial.clone()
    timeline = torch.arange(partial.shape[1], device=partial.device, dtype=partial.dtype)
    for row in range(partial.shape[0]):
        visible = torch.where(seen[row])[0]
        if visible.numel() == 0:
            continue
        if visible.numel() == 1:
            result[row] = partial[row, visible[0]]
            continue
        for dim in range(partial.shape[-1]):
            values = partial[row, visible, dim]
            filled = torch.empty_like(timeline)
            for pos in range(partial.shape[1]):
                if pos <= int(visible[0]):
                    filled[pos] = values[0]
                elif pos >= int(visible[-1]):
                    filled[pos] = values[-1]
                else:
                    right_index = torch.searchsorted(visible, torch.tensor(pos, device=partial.device))
                    left = visible[right_index - 1]
                    right = visible[right_index]
                    alpha = (pos - left).to(partial.dtype) / (right - left).to(partial.dtype)
                    filled[pos] = (1 - alpha) * partial[row, left, dim] + alpha * partial[row, right, dim]
            result[row, :, dim] = filled
    return result


def make_inputs(batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    filled = interpolate_history(batch["partial"], batch["seen"])
    seen_feature = batch["seen"].float().unsqueeze(-1)
    return torch.cat([filled, seen_feature], dim=-1), filled


def constant_velocity_prediction(batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    imputed_history = interpolate_history(batch["partial"], batch["seen"])
    future_len = batch["target"].shape[1] - imputed_history.shape[1]
    prediction = torch.empty(
        batch["partial"].shape[0],
        future_len,
        batch["partial"].shape[-1],
        device=batch["partial"].device,
        dtype=batch["partial"].dtype,
    )
    for row in range(batch["partial"].shape[0]):
        visible = torch.where(batch["seen"][row])[0]
        if visible.numel() >= 2:
            last = visible[-1]
            prev = visible[-2]
            span = (last - prev).to(batch["partial"].dtype).clamp_min(1)
            velocity = (batch["partial"][row, last] - batch["partial"][row, prev]) / span
            origin = batch["partial"][row, last]
            offset = torch.arange(
                imputed_history.shape[1],
                imputed_history.shape[1] + future_len,
                device=batch["partial"].device,
                dtype=batch["partial"].dtype,
            ) - last.to(batch["partial"].dtype)
            prediction[row] = origin + offset[:, None] * velocity
        elif visible.numel() == 1:
            prediction[row] = batch["partial"][row, visible[-1]]
        else:
            prediction[row].zero_()
    return prediction, imputed_history


class LSTMPredictor(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, model_input: torch.Tensor, pred_len: int = 12) -> torch.Tensor:
        _, state = self.encoder(model_input)
        decoder_input = model_input[:, -1:, :2]
        outputs = []
        hidden = state
        for _ in range(pred_len):
            decoded, hidden = self.decoder(decoder_input, hidden)
            step = self.head(decoded[:, -1])
            outputs.append(step)
            decoder_input = step[:, None]
        return torch.stack(outputs, dim=1)


class RNNPredictor(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        rnn_dropout = dropout if layers > 1 else 0.0
        self.encoder = nn.RNN(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=rnn_dropout,
        )
        self.decoder = nn.RNN(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=rnn_dropout,
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, model_input: torch.Tensor, pred_len: int = 12) -> torch.Tensor:
        _, state = self.encoder(model_input)
        decoder_input = model_input[:, -1:, :2]
        outputs = []
        hidden = state
        for _ in range(pred_len):
            decoded, hidden = self.decoder(decoder_input, hidden)
            step = self.head(decoded[:, -1])
            outputs.append(step)
            decoder_input = step[:, None]
        return torch.stack(outputs, dim=1)


class TransformerPredictor(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, 20, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.query = nn.Parameter(torch.randn(1, 12, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=max(1, layers // 2))
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, model_input: torch.Tensor, pred_len: int = 12) -> torch.Tensor:
        memory = self.input_proj(model_input) + self.position[:, : model_input.shape[1]]
        memory = self.encoder(memory)
        query = self.query[:, :pred_len].expand(model_input.shape[0], -1, -1)
        query = query + self.position[:, 8 : 8 + pred_len]
        decoded = self.decoder(query, memory)
        return self.head(decoded)


class TemporalConvPredictor(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        in_channels = 3
        for _ in range(max(1, layers)):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = hidden_dim
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 12 * 2),
        )

    def forward(self, model_input: torch.Tensor, pred_len: int = 12) -> torch.Tensor:
        encoded = self.encoder(model_input.transpose(1, 2)).flatten(1)
        return self.head(encoded).view(model_input.shape[0], 12, 2)[:, :pred_len]


class EquivariantMLPPredictor(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        depth = max(1, layers)
        modules = [nn.Linear(8 * 3 + 2, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        modules.append(nn.Linear(hidden_dim, 12 * 2))
        self.net = nn.Sequential(*modules)

    def forward(self, model_input: torch.Tensor, pred_len: int = 12) -> torch.Tensor:
        coords = model_input[:, :, :2]
        anchor = coords[:, -1:]
        centered = coords - anchor
        velocity = coords[:, -1] - coords[:, -2]
        features = torch.cat([centered, model_input[:, :, 2:]], dim=-1).flatten(1)
        delta = self.net(torch.cat([features, velocity], dim=-1)).view(model_input.shape[0], 12, 2)
        return anchor + delta[:, :pred_len]


class DestinationRefinePredictor(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.endpoint = nn.Linear(hidden_dim, 2)
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim + 12 * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 12 * 2),
        )

    def forward(self, model_input: torch.Tensor, pred_len: int = 12) -> torch.Tensor:
        _, (hidden, _) = self.encoder(model_input)
        context = hidden[-1]
        start = model_input[:, -1:, :2]
        endpoint = self.endpoint(context)[:, None]
        alpha = torch.linspace(
            1 / 12,
            1,
            12,
            device=model_input.device,
            dtype=model_input.dtype,
        )[None, :, None]
        coarse = start + alpha * (endpoint - start)
        residual = self.refine(torch.cat([context, coarse.flatten(1)], dim=-1)).view(model_input.shape[0], 12, 2)
        return (coarse + residual)[:, :pred_len]


class CVAEPredictor(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, latent_dim: int, dropout: float, dual: bool = False) -> None:
        super().__init__()
        self.dual = dual
        lstm_dropout = dropout if layers > 1 else 0.0
        self.motion_encoder = nn.LSTM(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        if dual:
            self.velocity_encoder = nn.LSTM(
                input_size=2,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        context_dim = hidden_dim * (2 if dual else 1)
        self.prior = nn.Linear(context_dim, latent_dim * 2)
        self.posterior = nn.Linear(context_dim + 12 * 2, latent_dim * 2)
        self.decoder = nn.Sequential(
            nn.Linear(context_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 12 * 2),
        )

    def encode_context(self, model_input: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.motion_encoder(model_input)
        context = hidden[-1]
        if not self.dual:
            return context
        coords = model_input[:, :, :2]
        velocity = torch.diff(coords, dim=1, prepend=coords[:, :1])
        _, (velocity_hidden, _) = self.velocity_encoder(velocity)
        return torch.cat([context, velocity_hidden[-1]], dim=-1)

    @staticmethod
    def split_stats(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(-6, 4)

    def decode(self, context: torch.Tensor, z: torch.Tensor, pred_len: int) -> torch.Tensor:
        return self.decoder(torch.cat([context, z], dim=-1)).view(context.shape[0], 12, 2)[:, :pred_len]

    def forward(self, model_input: torch.Tensor, pred_len: int = 12) -> torch.Tensor:
        context = self.encode_context(model_input)
        prior_mu, _ = self.split_stats(self.prior(context))
        return self.decode(context, prior_mu, pred_len)

    def training_loss(self, model_input: torch.Tensor, batch: Dict[str, torch.Tensor], args) -> torch.Tensor:
        context = self.encode_context(model_input)
        future = batch["target"][:, 8:]
        posterior_mu, posterior_logvar = self.split_stats(self.posterior(torch.cat([context, future.flatten(1)], dim=-1)))
        prior_mu, prior_logvar = self.split_stats(self.prior(context))
        eps = torch.randn_like(posterior_mu)
        z = posterior_mu + eps * torch.exp(0.5 * posterior_logvar)
        prediction = self.decode(context, z, future.shape[1])
        recon = future_loss(prediction, batch["target"], batch["future_valid"])
        kl = 0.5 * (
            prior_logvar
            - posterior_logvar
            + (torch.exp(posterior_logvar) + (posterior_mu - prior_mu).pow(2)) / torch.exp(prior_logvar)
            - 1
        ).sum(dim=-1).mean()
        return recon + args.kl_weight * kl


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.baseline == "rnn":
        return RNNPredictor(args.hidden_dim, args.layers, args.dropout)
    if args.baseline == "lstm":
        return LSTMPredictor(args.hidden_dim, args.layers, args.dropout)
    if args.baseline in {"transformer", "tutr"}:
        return TransformerPredictor(args.hidden_dim, args.layers, args.heads, args.dropout)
    if args.baseline == "graphtern":
        return TemporalConvPredictor(args.hidden_dim, args.layers, args.dropout)
    if args.baseline == "eqmotion":
        return EquivariantMLPPredictor(args.hidden_dim, args.layers, args.dropout)
    if args.baseline == "ppt":
        return DestinationRefinePredictor(args.hidden_dim, args.layers, args.dropout)
    if args.baseline == "cvae":
        return CVAEPredictor(args.hidden_dim, args.layers, args.latent_dim, args.dropout, dual=False)
    if args.baseline == "social_dualcvae":
        return CVAEPredictor(args.hidden_dim, args.layers, args.latent_dim, args.dropout, dual=True)
    raise ValueError(args.baseline)


def future_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    future_valid: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    delta = prediction - target[:, 8:]
    if loss_type == "ade":
        diff = torch.linalg.vector_norm(delta, dim=-1)
    elif loss_type == "smooth_l1":
        diff = torch.nn.functional.smooth_l1_loss(
            prediction, target[:, 8:], reduction="none"
        ).sum(dim=-1)
    else:
        diff = delta.pow(2).sum(dim=-1)
    return (diff * future_valid.float()).sum() / future_valid.float().sum().clamp_min(1)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, args, device) -> float:
    model.train()
    total = 0.0
    count = 0
    for raw_batch in tqdm(loader, desc=f"train {args.baseline}", leave=False):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        model_input, _ = make_inputs(batch)
        if hasattr(model, "training_loss"):
            loss = model.training_loss(model_input, batch, args)
        else:
            prediction = model(model_input, pred_len=batch["target"].shape[1] - 8)
            loss = future_loss(
                prediction, batch["target"], batch["future_valid"], args.loss_type
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(loss.item())
        count += 1
        if args.batch_sleep_ms > 0:
            time.sleep(args.batch_sleep_ms / 1000.0)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, args, device, coordinate_scale: float) -> Dict[str, float]:
    model.eval()
    sums = {
        "minADE": 0.0,
        "minFDE": 0.0,
        "meanADE": 0.0,
        "impute_minADE": 0.0,
        "JointADE": 0.0,
        "MR@2m": 0.0,
    }
    rows = 0
    impute_rows = 0
    p90_values = []
    for raw_batch in tqdm(loader, desc=f"eval {args.baseline}", leave=False):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        model_input, imputed_history = make_inputs(batch)
        futures = [
            model(model_input, pred_len=batch["target"].shape[1] - 8)
            for _ in range(args.num_samples)
        ]
        future = torch.stack(futures, dim=1)
        history = imputed_history[:, None].expand(-1, args.num_samples, -1, -1)
        samples = torch.cat([history, future], dim=2)
        future_error = torch.linalg.vector_norm(
            samples[:, :, 8:] - batch["target"][:, None, 8:],
            dim=-1,
        )
        future_valid = batch["future_valid"]
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
        missing = batch["impute_mask"].float()
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
        has_missing = missing_count > 0
        row_min_ade = ade.min(dim=1).values
        row_min_fde = fde.min(dim=1).values
        row_mean_ade = ade.mean(dim=1)
        row_impute_min_ade = imputation.min(dim=1).values
        row_min_joint = joint.min(dim=1).values
        sums["minADE"] += float(row_min_ade.sum().item())
        sums["minFDE"] += float(row_min_fde.sum().item())
        sums["meanADE"] += float(row_mean_ade.sum().item())
        sums["JointADE"] += float(row_min_joint.sum().item())
        sums["MR@2m"] += float((row_min_fde * coordinate_scale > 2.0).float().sum().item())
        p90_values.extend((row_mean_ade * coordinate_scale).detach().cpu().tolist())
        if has_missing.any():
            sums["impute_minADE"] += float(row_impute_min_ade[has_missing].sum().item())
            impute_rows += int(has_missing.sum().item())
        rows += batch["target"].shape[0]
        if args.batch_sleep_ms > 0:
            time.sleep(args.batch_sleep_ms / 1000.0)
    result = {
        key: sums[key] / max(rows, 1) * coordinate_scale
        for key in ("minADE", "minFDE", "meanADE", "JointADE")
    }
    result["MR@2m"] = sums["MR@2m"] / max(rows, 1) * 100.0
    result["P90-ADE"] = float(np.percentile(np.asarray(p90_values), 90))
    result["impute_minADE"] = sums["impute_minADE"] / max(impute_rows, 1) * coordinate_scale
    return result


@torch.no_grad()
def evaluate_constant_velocity(loader: DataLoader, args, device, coordinate_scale: float) -> Dict[str, float]:
    sums = {
        "minADE": 0.0,
        "minFDE": 0.0,
        "meanADE": 0.0,
        "impute_minADE": 0.0,
        "JointADE": 0.0,
        "MR@2m": 0.0,
    }
    rows = 0
    impute_rows = 0
    p90_values = []
    for raw_batch in tqdm(loader, desc=f"eval {args.baseline}", leave=False):
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        future, imputed_history = constant_velocity_prediction(batch)
        samples = torch.cat([imputed_history, future], dim=1)[:, None]
        future_error = torch.linalg.vector_norm(
            samples[:, :, 8:] - batch["target"][:, None, 8:],
            dim=-1,
        )
        future_valid = batch["future_valid"]
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
        missing = batch["impute_mask"].float()
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
        has_missing = missing_count > 0
        row_min_ade = ade.min(dim=1).values
        row_min_fde = fde.min(dim=1).values
        row_mean_ade = ade.mean(dim=1)
        row_impute_min_ade = imputation.min(dim=1).values
        row_min_joint = joint.min(dim=1).values
        sums["minADE"] += float(row_min_ade.sum().item())
        sums["minFDE"] += float(row_min_fde.sum().item())
        sums["meanADE"] += float(row_mean_ade.sum().item())
        sums["JointADE"] += float(row_min_joint.sum().item())
        sums["MR@2m"] += float((row_min_fde * coordinate_scale > 2.0).float().sum().item())
        p90_values.extend((row_mean_ade * coordinate_scale).detach().cpu().tolist())
        if has_missing.any():
            sums["impute_minADE"] += float(row_impute_min_ade[has_missing].sum().item())
            impute_rows += int(has_missing.sum().item())
        rows += batch["target"].shape[0]
        if args.batch_sleep_ms > 0:
            time.sleep(args.batch_sleep_ms / 1000.0)
    result = {
        key: sums[key] / max(rows, 1) * coordinate_scale
        for key in ("minADE", "minFDE", "meanADE", "JointADE")
    }
    result["MR@2m"] = sums["MR@2m"] / max(rows, 1) * 100.0
    result["P90-ADE"] = float(np.percentile(np.asarray(p90_values), 90))
    result["impute_minADE"] = sums["impute_minADE"] / max(impute_rows, 1) * coordinate_scale
    return result


def append_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_summary(output_dir: Path, args, best_epoch: int, best_val_score: float, metrics: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run": f"{args.dataset}_{args.difficulty}_{args.baseline}_seed{args.seed}",
        "dataset": args.dataset,
        "difficulty": args.difficulty,
        "baseline": args.baseline,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "metrics": metrics,
        "test_minADE": metrics["minADE"],
        "test_minFDE": metrics["minFDE"],
        "test_meanADE": metrics["meanADE"],
        "test_impute_minADE": metrics["impute_minADE"],
        "fold": args.fold,
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
        "selection_split": "validation",
        "test_used_for_selection": False,
    }
    if torch.cuda.is_available():
        summary["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        summary["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024 ** 2)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    append_csv(
        output_dir.parent / "all_results.csv",
        {
            "dataset": args.dataset,
            "difficulty": args.difficulty,
            "method": args.baseline,
            "seed": args.seed,
            **metrics,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--difficulty", required=True, choices=["Easy", "Hard", "Full"])
    parser.add_argument(
        "--baseline",
        required=True,
        choices=[
            "constant_velocity",
            "cvae",
            "eqmotion",
            "graphtern",
            "lstm",
            "ppt",
            "rnn",
            "social_dualcvae",
            "transformer",
            "tutr",
        ],
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--num-folds", type=int, default=1)
    parser.add_argument("--fold-seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.0)
    parser.add_argument("--batch-sleep-ms", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--loss-type", choices=["mse", "ade", "smooth_l1"], default="mse")
    parser.add_argument("--final-split", choices=["val", "test"], default="test")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--selection-metric", choices=["composite", "meanADE"], default="composite")
    parser.add_argument("--output-root", default="results/trajectory_prediction_baselines_v1")
    parser.add_argument(
        "--av2-root",
        default="data/raw/Argoverse2_Motion_Forecasting",
    )
    parser.add_argument("--av2-train-size", type=int, default=8192)
    parser.add_argument("--av2-val-size", type=int, default=1024)
    parser.add_argument("--av2-test-size", type=int, default=1024)
    parser.add_argument("--av2-selection-seed", type=int, default=2026)
    parser.add_argument("--av2-cache-root", default=None)
    parser.add_argument("--mask-mode", default="mixed")
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--mask-seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.cuda.reset_peak_memory_stats()

    if args.dataset.upper() in {"AV2", "ARGOVERSE2", "ARGOVERSE_2"}:
        bundle: JointDataBundle = load_bundle(
            REPO_ROOT,
            args.dataset,
            args.difficulty,
            av2_root=args.av2_root,
            av2_train_size=args.av2_train_size,
            av2_val_size=args.av2_val_size,
            av2_test_size=args.av2_test_size,
            av2_selection_seed=args.av2_selection_seed,
            av2_cache_root=args.av2_cache_root or None,
            mask_mode=args.mask_mode,
            mask_ratio=args.mask_ratio,
            mask_seed=args.mask_seed,
        )
    else:
        bundle = load_bundle(REPO_ROOT, args.dataset, args.difficulty)
    train_dataset = bundle.train
    val_dataset = bundle.val
    fold_meta = {"num_folds": 1, "fold": None}
    if args.num_folds > 1:
        if args.fold < 0 or args.fold >= args.num_folds:
            raise ValueError(f"--fold must be in [0, {args.num_folds}) when --num-folds > 1")
        if args.dataset.upper() in {"AV2", "ARGOVERSE2", "ARGOVERSE_2"}:
            raise ValueError("folded pedestrian runner is intended for ETH/UCY; AV2 uses its fixed split")
        all_indices = np.arange(len(bundle.train), dtype=np.int64)
        rng = np.random.default_rng(args.fold_seed)
        rng.shuffle(all_indices)
        folds = np.array_split(all_indices, args.num_folds)
        val_indices = folds[args.fold]
        train_indices = np.concatenate([folds[i] for i in range(args.num_folds) if i != args.fold])
        train_dataset = Subset(bundle.train, train_indices.tolist())
        val_dataset = Subset(bundle.train, val_indices.tolist())
        fold_meta = {
            "num_folds": args.num_folds,
            "fold": args.fold,
            "fold_seed": args.fold_seed,
            "train_count": int(len(train_indices)),
            "val_count": int(len(val_indices)),
            "train_index_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
            "val_index_sha256": hashlib.sha256(val_indices.tobytes()).hexdigest(),
        }
    loaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": DataLoader(bundle.test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }

    run_dir = (
        Path(args.output_root)
        / f"{args.dataset}_{args.difficulty}"
        / args.baseline
        / (f"fold{args.fold}of{args.num_folds}" if args.num_folds > 1
           else f"{args.dataset}_{args.difficulty}_{args.baseline}_seed{args.seed}")
    )
    if (run_dir / "summary.json").is_file():
        print(f"skip completed {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_audit.json").write_text(json.dumps({
        "args": vars(args),
        "fold": fold_meta,
        "dataset": args.dataset,
        "difficulty": args.difficulty,
        "test_count": int(len(bundle.test)),
        "coordinate_scale": float(bundle.coordinate_scale),
        "selection_split": "validation",
        "test_used_for_selection": False,
    }, indent=2, sort_keys=True) + "\n")

    if args.baseline == "constant_velocity":
        test = evaluate_constant_velocity(loaders["test"], args, device, bundle.coordinate_scale)
        write_summary(run_dir, args, 0, float("nan"), test)
        print(json.dumps(test, indent=2, sort_keys=True))
        return

    model = build_model(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_epoch = 0
    best_val = float("inf")
    wait = 0
    history_path = run_dir / "history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, args, device)
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        val = evaluate(model, loaders["val"], args, device, bundle.coordinate_scale)
        score = (
            val["meanADE"]
            if args.selection_metric == "meanADE"
            else (val["minADE"] + val["minFDE"]) / 2
        )
        append_csv(history_path, {"epoch": epoch, "train_loss": train_loss, **val, "score": score})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val={val} score={score:.6f}")
        if score < best_val:
            best_val = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if args.save_checkpoint:
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_val_score": best_val,
            },
            run_dir / "best.pt",
        )
    final_metrics = evaluate(model, loaders[args.final_split], args, device, bundle.coordinate_scale)
    if args.final_split == "test":
        write_summary(run_dir, args, best_epoch, best_val, final_metrics)
    else:
        (run_dir / "val_summary.json").write_text(json.dumps({
            "split": "val",
            "best_epoch": best_epoch,
            "best_val_score": best_val,
            **final_metrics,
        }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
