from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.memoflow import TrajectoryMemory
from src.train import sample_flow


class DeterministicVelocity(nn.Module):
    def forward(self, state, time, partial, seen, memory_context):
        condition = torch.cat(
            [
                partial,
                torch.zeros(
                    partial.shape[0],
                    12,
                    2,
                    dtype=partial.dtype,
                    device=partial.device,
                ),
            ],
            dim=1,
        )
        return 0.1 * condition + 0.05 * memory_context + time[:, None, None]


def arguments(use_memory: bool, chunk: int) -> SimpleNamespace:
    return SimpleNamespace(
        use_memory=use_memory,
        num_samples=6,
        memory_source_ratio=1.0,
        coverage_samples=0,
        diversity_mode="endpoint",
        diversity_relevance=0.2,
        source_noise=0.05,
        eval_source_noise=0.0,
        flow_strength=1.0,
        memory_flow_strength=0.5,
        flow_steps=5,
        sample_chunk_size=chunk,
    )


class FlowSampleChunkingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        batch_size = 3
        target = torch.randn(batch_size, 20, 2)
        seen = torch.tensor(
            [
                [True, False, True, True, False, True, True, True],
                [True, True, False, True, True, False, True, True],
                [False, True, True, True, False, True, True, True],
            ]
        )
        partial = target[:, :8].clone()
        partial[~seen] = 0
        self.batch = {
            "target": target,
            "partial": partial,
            "seen": seen,
            "key": torch.randn(batch_size, 6),
        }
        keys = torch.randn(12, 6)
        values = torch.randn(12, 20, 2)
        self.memory = TrajectoryMemory(keys, values, top_k=6)
        self.model = DeterministicVelocity().eval()

    def compare(self, use_memory: bool) -> None:
        memory = self.memory if use_memory else None
        torch.manual_seed(123)
        serial = sample_flow(
            self.model,
            memory,
            copy.deepcopy(self.batch),
            arguments(use_memory, 1),
        )
        torch.manual_seed(123)
        chunked = sample_flow(
            self.model,
            memory,
            copy.deepcopy(self.batch),
            arguments(use_memory, 4),
        )
        self.assertTrue(torch.allclose(serial, chunked, atol=1e-6, rtol=1e-6))

    def test_chunking_matches_serial_for_gaussian_sources(self) -> None:
        self.compare(use_memory=False)

    def test_chunking_matches_serial_for_memory_sources(self) -> None:
        self.compare(use_memory=True)


if __name__ == "__main__":
    unittest.main()
