import unittest

import numpy as np

from libs.control.objectreact import ObjRelLearntController


def make_controller(probability, seed=42, context="episode-a"):
    controller = ObjRelLearntController.__new__(ObjRelLearntController)
    controller.inject_costmap_noise = True
    controller.noise_prob = probability
    controller.inference_noise_seed = seed
    controller._noise_rng = None
    controller._reset_inference_noise(context)
    return controller


class InferenceCostmapNoiseTest(unittest.TestCase):
    def test_probability_zero_preserves_costmap(self):
        controller = make_controller(0.0)
        costmap = np.ones((8, 4, 4), dtype=np.float32)

        actual = controller._apply_inference_costmap_noise(costmap)

        np.testing.assert_array_equal(actual, costmap)
        self.assertFalse(controller._last_inference_noise_applied)

    def test_probability_one_zeros_costmap(self):
        controller = make_controller(1.0)
        costmap = np.ones((8, 4, 4), dtype=np.float32)

        actual = controller._apply_inference_costmap_noise(costmap)

        np.testing.assert_array_equal(actual, np.zeros_like(costmap))
        self.assertTrue(controller._last_inference_noise_applied)

    def test_same_episode_gets_same_noise_schedule(self):
        first = make_controller(0.2, seed=7, context="shared-episode")
        second = make_controller(0.2, seed=7, context="shared-episode")
        costmap = np.ones((8, 2, 2), dtype=np.float32)

        first_schedule = [
            not first._apply_inference_costmap_noise(costmap).any() for _ in range(100)
        ]
        second_schedule = [
            not second._apply_inference_costmap_noise(costmap).any() for _ in range(100)
        ]

        self.assertEqual(first_schedule, second_schedule)
        self.assertGreater(sum(first_schedule), 0)
        self.assertLess(sum(first_schedule), len(first_schedule))

    def test_episode_context_changes_noise_schedule(self):
        first = make_controller(0.2, seed=7, context="episode-a")
        second = make_controller(0.2, seed=7, context="episode-b")
        costmap = np.ones((8, 2, 2), dtype=np.float32)

        first_schedule = [
            not first._apply_inference_costmap_noise(costmap).any() for _ in range(100)
        ]
        second_schedule = [
            not second._apply_inference_costmap_noise(costmap).any() for _ in range(100)
        ]

        self.assertNotEqual(first_schedule, second_schedule)


if __name__ == "__main__":
    unittest.main()
