from __future__ import annotations

import unittest

import torch

from models.memoflow import JointFlowModel, clamp_observed


class MemoFlowModelTest(unittest.TestCase):
    def test_flow_forward_shape_and_gradients(self) -> None:
        torch.manual_seed(7)
        model = JointFlowModel(
            hidden_dim=32,
            layers=1,
            heads=4,
            dropout=0.0,
        )
        state = torch.randn(2, 20, 2, requires_grad=True)
        partial = torch.randn(2, 8, 2)
        seen = torch.tensor(
            [
                [True, True, False, True, False, True, True, True],
                [True, False, True, True, True, False, True, True],
            ]
        )
        memory_context = torch.randn(2, 20, 2)
        velocity = model(
            state,
            torch.tensor([0.25, 0.75]),
            partial,
            seen,
            memory_context,
        )
        self.assertEqual(velocity.shape, (2, 20, 2))
        self.assertTrue(torch.isfinite(velocity).all())
        velocity.square().mean().backward()
        self.assertIsNotNone(state.grad)
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_observation_projection_preserves_visible_history(self) -> None:
        state = torch.randn(2, 20, 2)
        target = torch.randn(2, 20, 2)
        seen = torch.tensor(
            [
                [True, False, True, True, False, True, True, True],
                [False, True, True, True, True, False, True, True],
            ]
        )
        projected = clamp_observed(state, target, seen)
        self.assertTrue(torch.equal(projected[:, :8][seen], target[:, :8][seen]))
        self.assertTrue(torch.equal(projected[:, 8:], state[:, 8:]))


if __name__ == "__main__":
    unittest.main()
