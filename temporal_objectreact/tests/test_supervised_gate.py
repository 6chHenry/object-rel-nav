import unittest

import torch
import torch.nn.functional as F

from temporal_objectreact.temporal_aggregator import (
    ReliabilityGate,
    ReliabilityGatedGRUTemporalAggregator,
)
from temporal_objectreact.gnm_temporal import _make_aggregator
from temporal_objectreact.train_temporal import apply_temporal_corruption


class TemporalCorruptionTest(unittest.TestCase):
    def test_probability_one_corrupts_every_supervised_frame(self):
        goal = torch.ones(2, 18, 3, 4)

        corrupted, mask = apply_temporal_corruption(
            goal,
            num_frames=6,
            probability=1.0,
            mode="zero",
        )
        frames = corrupted.reshape(2, 6, 3, 3, 4)

        self.assertTrue(torch.all(~mask[:, 0]))
        self.assertTrue(torch.all(mask[:, 1:]))
        torch.testing.assert_close(frames[:, 0], torch.ones_like(frames[:, 0]))
        torch.testing.assert_close(frames[:, 1:], torch.zeros_like(frames[:, 1:]))

    def test_fixed_generator_reproduces_corruption_labels(self):
        goal = torch.ones(32, 18, 2, 2)
        first_generator = torch.Generator().manual_seed(7)
        second_generator = torch.Generator().manual_seed(7)

        first, first_mask = apply_temporal_corruption(
            goal,
            num_frames=6,
            probability=0.2,
            generator=first_generator,
        )
        second, second_mask = apply_temporal_corruption(
            goal,
            num_frames=6,
            probability=0.2,
            generator=second_generator,
        )

        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first_mask, second_mask)


class SupervisedReliabilityGateTest(unittest.TestCase):
    def test_factory_propagates_gated_history_update(self):
        aggregator = _make_aggregator(
            "reliability_gated_gru",
            dim=8,
            gate_history_update="gated",
        )

        self.assertEqual(aggregator.gate.history_update, "gated")

    def test_logits_are_returned_without_breaking_old_interface(self):
        torch.manual_seed(0)
        aggregator = ReliabilityGatedGRUTemporalAggregator(
            dim=16,
            gate_history_update="gated",
        )
        inputs = torch.randn(3, 6, 16)

        old_output, old_alpha = aggregator(inputs, return_alpha=True)
        output, alpha, logits = aggregator(
            inputs,
            return_alpha=True,
            return_logits=True,
        )

        torch.testing.assert_close(output, old_output)
        torch.testing.assert_close(alpha, old_alpha)
        self.assertEqual(tuple(logits.shape), (3, 6))
        torch.testing.assert_close(logits[:, 0], torch.zeros(3))

    def test_corruption_loss_reaches_gate_scorer(self):
        torch.manual_seed(1)
        gate = ReliabilityGate(dim=8, history_update="gated")
        inputs = torch.randn(4, 6, 8)
        targets = torch.tensor(
            [
                [0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        )

        _gated, _alpha, reliability_logits = gate(
            inputs,
            return_alpha=True,
            return_logits=True,
        )
        loss = F.binary_cross_entropy_with_logits(
            -reliability_logits[:, 1:],
            targets,
            pos_weight=torch.tensor(4.0),
        )
        loss.backward()

        gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in gate.scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)

    def test_gated_history_changes_later_gate_scores(self):
        torch.manual_seed(2)
        raw_gate = ReliabilityGate(dim=8, history_update="raw")
        gated_gate = ReliabilityGate(dim=8, history_update="gated")
        gated_gate.load_state_dict(raw_gate.state_dict())
        inputs = torch.randn(2, 6, 8)
        inputs[:, 1] *= 20.0

        _raw, _raw_alpha, raw_logits = raw_gate(
            inputs,
            return_alpha=True,
            return_logits=True,
        )
        _gated, _gated_alpha, gated_logits = gated_gate(
            inputs,
            return_alpha=True,
            return_logits=True,
        )

        torch.testing.assert_close(raw_logits[:, :2], gated_logits[:, :2])
        self.assertFalse(torch.allclose(raw_logits[:, 2:], gated_logits[:, 2:]))


if __name__ == "__main__":
    unittest.main()
