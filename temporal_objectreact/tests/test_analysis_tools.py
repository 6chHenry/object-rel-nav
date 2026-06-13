import tempfile
import unittest
from pathlib import Path

from temporal_objectreact.analysis.analyze_gate_diagnostics import parse_conditions
from temporal_objectreact.analysis.make_matched_rollouts import select_episode
from temporal_objectreact.analysis.result_utils import parse_metadata


def episode(root: Path, name: str, status: str, distance: float):
    episode_dir = root / name
    episode_dir.mkdir()
    video_path = episode_dir / "repeat.mp4"
    video_path.touch()
    return {
        "episode_dir": episode_dir,
        "video_path": video_path,
        "metadata": {
            "success_status": status,
            "final_distance": str(distance),
        },
    }


class AnalysisToolsTest(unittest.TestCase):
    def test_gate_conditions_can_select_supervised_model(self):
        self.assertEqual(
            parse_conditions(["train_noise02=02", "supervised=supervised"]),
            {"train_noise02": "02", "supervised": "supervised"},
        )

    def test_selects_largest_final_distance_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = {
                "episode_a_": episode(root, "plain-a", "exceeded_steps", 4.0),
                "episode_b_": episode(root, "plain-b", "exceeded_steps", 8.0),
            }
            reliability = {
                "episode_a_": episode(root, "rel-a", "success", 0.8),
                "episode_b_": episode(root, "rel-b", "success", 0.9),
            }

            episode_id, improvement = select_episode(plain, reliability)

            self.assertEqual(episode_id, "episode_b_")
            self.assertAlmostEqual(improvement, 7.1)

    def test_rejects_plain_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = {
                "episode_a_": episode(root, "plain-a", "success", 0.8),
            }
            reliability = {
                "episode_a_": episode(root, "rel-a", "success", 0.7),
            }

            with self.assertRaises(RuntimeError):
                select_episode(plain, reliability)

    def test_metadata_parser_preserves_values_after_equals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.txt"
            path.write_text(
                "success_status=success\nnote=left=right\n",
                encoding="utf-8",
            )

            metadata = parse_metadata(path)

            self.assertEqual(metadata["success_status"], "success")
            self.assertEqual(metadata["note"], "left=right")


if __name__ == "__main__":
    unittest.main()
