import unittest

import numpy as np
import torch

from libs.control.objectreact import ObjRelLearntController
from temporal_objectreact.objectreact_temporal_controller import (
    ObjRelTemporalLearntController,
)
from temporal_objectreact.temporal_aggregator import (
    ReliabilityGatedGRUTemporalAggregator,
)


class ReliabilityGateDiagnosticsTest(unittest.TestCase):
    def test_reliability_gate_returns_one_score_per_frame(self):
        torch.manual_seed(0)
        aggregator = ReliabilityGatedGRUTemporalAggregator(dim=16)
        inputs = torch.randn(3, 6, 16)

        output_without_diagnostics, alpha_without_diagnostics = aggregator(inputs)
        output, alpha = aggregator(inputs, return_alpha=True)

        self.assertEqual(tuple(output.shape), (3, 16))
        self.assertEqual(tuple(alpha.shape), (3, 6))
        self.assertIsNone(alpha_without_diagnostics)
        torch.testing.assert_close(output, output_without_diagnostics)
        self.assertTrue(torch.all(alpha >= 0.0))
        self.assertTrue(torch.all(alpha <= 1.0))
        torch.testing.assert_close(alpha[:, 0], torch.ones(3))

    def test_disabled_controller_has_no_extra_log_fields(self):
        controller = ObjRelLearntController.__new__(ObjRelLearntController)

        self.assertEqual(controller._controller_log_extras(), {})

    def test_temporal_controller_serializes_full_and_current_alpha(self):
        controller = ObjRelTemporalLearntController.__new__(
            ObjRelTemporalLearntController
        )
        controller.log_gate_diagnostics = True
        controller.model = type(
            "ModelStub",
            (),
            {
                "last_temporal_gate_alpha": torch.tensor(
                    [[1.0, 0.8, 0.6, 0.4, 0.2, 0.1]]
                )
            },
        )()

        extras = controller._controller_log_extras()

        np.testing.assert_allclose(
            extras["temporal_gate_alpha"],
            np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1], dtype=np.float32),
        )
        self.assertAlmostEqual(extras["temporal_gate_alpha_current"], 0.1)

    def test_temporal_controller_diagnostics_default_to_empty(self):
        controller = ObjRelTemporalLearntController.__new__(
            ObjRelTemporalLearntController
        )
        controller.log_gate_diagnostics = False

        self.assertEqual(controller._controller_log_extras(), {})


if __name__ == "__main__":
    unittest.main()
