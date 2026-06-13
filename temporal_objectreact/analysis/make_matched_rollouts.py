#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from temporal_objectreact.analysis.result_utils import (
    TASKS,
    discover_latest_run,
    load_blacklists,
    load_controller_logs,
    load_episode_index,
)


ALPHA_KEY = "temporal_gate_alpha_current"
LABEL_KEY = "inference_costmap_noise_applied"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create matched Plain GRU vs Reliability Gate rollout videos."
    )
    parser.add_argument("--results-root", type=Path, default=Path("out/results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/analysis/qualitative"),
    )
    parser.add_argument(
        "--blacklist-config",
        type=Path,
        default=Path("configs/defaults.yaml"),
    )
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        metavar="TASK=EPISODE_ID",
        help="Override automatic selection for one task.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Only select and report matched episodes.",
    )
    return parser.parse_args()


def parse_episode_overrides(values: list[str]) -> dict[str, str]:
    overrides = {}
    valid_tasks = {task.name for task in TASKS}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected TASK=EPISODE_ID, got {value!r}")
        task, episode_id = value.split("=", 1)
        if task not in valid_tasks:
            raise ValueError(f"Unknown task {task!r}")
        if not episode_id.endswith("_"):
            episode_id += "_"
        overrides[task] = episode_id
    return overrides


def plain_experiment_name(task_name: str) -> str:
    return f"temporal_{task_name}_gru_train_noise02_eval_noise02"


def reliability_experiment_name(task_name: str) -> str:
    return f"temporal_{task_name}_reliability_gate_diag_train_noise02_eval_noise02"


def final_distance(episode: dict) -> float:
    return float(episode["metadata"].get("final_distance", "nan"))


def select_episode(
    plain_episodes: dict[str, dict],
    reliability_episodes: dict[str, dict],
    requested: str | None = None,
) -> tuple[str, float]:
    candidates = []
    for episode_id in sorted(set(plain_episodes) & set(reliability_episodes)):
        plain = plain_episodes[episode_id]
        reliability = reliability_episodes[episode_id]
        if reliability["metadata"].get("success_status") != "success":
            continue
        if plain["metadata"].get("success_status") == "success":
            continue
        if not plain["video_path"].is_file() or not reliability["video_path"].is_file():
            continue
        improvement = final_distance(plain) - final_distance(reliability)
        candidates.append((episode_id, improvement))
    if requested is not None:
        for episode_id, improvement in candidates:
            if episode_id == requested:
                return episode_id, improvement
        raise ValueError(
            f"Requested episode {requested!r} is not a valid matched "
            "Reliability-success/Plain-failure case."
        )
    if not candidates:
        raise RuntimeError("No valid Reliability-success/Plain-failure match found.")
    return sorted(candidates, key=lambda item: (-item[1], item[0]))[0]


def log_at(logs: list[dict | None], index: int) -> dict:
    if not logs:
        return {}
    log = logs[min(index, len(logs) - 1)]
    return log if isinstance(log, dict) else {}


def annotate_panel(
    frame: np.ndarray,
    method: str,
    status: str,
    step: int,
    log: dict,
    show_alpha: bool,
) -> np.ndarray:
    panel = cv2.copyMakeBorder(
        frame, 64, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28)
    )
    noise_injected = bool(log.get(LABEL_KEY, False))
    if noise_injected:
        cv2.rectangle(
            panel,
            (2, 66),
            (panel.shape[1] - 3, panel.shape[0] - 3),
            (0, 0, 255),
            5,
        )
    cv2.putText(
        panel,
        f"{method} | {status} | step {step}",
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    detail = "INJECTED NOISE" if noise_injected else "clean frame"
    color = (0, 80, 255) if noise_injected else (190, 190, 190)
    if show_alpha:
        alpha = log.get(ALPHA_KEY)
        alpha_text = "n/a" if alpha is None else f"{float(alpha):.3f}"
        detail += f" | current alpha={alpha_text}"
    cv2.putText(
        panel,
        detail,
        (14, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    return panel


def compose_frame(
    plain_frame: np.ndarray,
    reliability_frame: np.ndarray,
    plain_status: str,
    reliability_status: str,
    step: int,
    plain_log: dict,
    reliability_log: dict,
) -> np.ndarray:
    if plain_frame.shape[:2] != reliability_frame.shape[:2]:
        reliability_frame = cv2.resize(
            reliability_frame,
            (plain_frame.shape[1], plain_frame.shape[0]),
        )
    plain_panel = annotate_panel(
        plain_frame, "Plain GRU", plain_status, step, plain_log, False
    )
    reliability_panel = annotate_panel(
        reliability_frame,
        "Reliability-Gated GRU",
        reliability_status,
        step,
        reliability_log,
        True,
    )
    return np.concatenate([plain_panel, reliability_panel], axis=1)


def find_ffmpeg() -> str:
    candidate = Path("portable_envs/nav_env/bin/ffmpeg")
    if candidate.is_file():
        return str(candidate.resolve())
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    raise FileNotFoundError("ffmpeg was not found.")


def render_pair(
    task_name: str,
    episode_id: str,
    plain: dict,
    reliability: dict,
    output_path: Path,
    fps: float,
) -> dict[str, np.ndarray]:
    plain_logs = load_controller_logs(plain["results_path"])
    reliability_logs = load_controller_logs(reliability["results_path"])
    if not any(isinstance(log, dict) and ALPHA_KEY in log for log in reliability_logs):
        raise ValueError(
            f"Reliability logs lack {ALPHA_KEY!r}: {reliability['results_path']}"
        )

    plain_capture = cv2.VideoCapture(str(plain["video_path"]))
    reliability_capture = cv2.VideoCapture(str(reliability["video_path"]))
    plain_count = int(plain_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reliability_count = int(reliability_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if plain_count <= 0 or reliability_count <= 0:
        raise RuntimeError(f"Could not read input videos for {task_name}")
    total_frames = max(plain_count, reliability_count)

    ok_plain, last_plain = plain_capture.read()
    ok_reliability, last_reliability = reliability_capture.read()
    if not ok_plain or not ok_reliability:
        raise RuntimeError(f"Could not read first input frame for {task_name}")

    sample = compose_frame(
        last_plain,
        last_reliability,
        plain["metadata"].get("success_status", ""),
        reliability["metadata"].get("success_status", ""),
        0,
        log_at(plain_logs, 0),
        log_at(reliability_logs, 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact_indices = {
        "start": 0,
        "reliability_terminal": reliability_count - 1,
        "plain_terminal": plain_count - 1,
    }
    contact_frames = {}

    with tempfile.NamedTemporaryFile(
        suffix=".mp4", dir=output_path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (sample.shape[1], sample.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {temporary_path}")

    try:
        for index in range(total_frames):
            if index > 0:
                if index < plain_count:
                    ok, frame = plain_capture.read()
                    if ok:
                        last_plain = frame
                if index < reliability_count:
                    ok, frame = reliability_capture.read()
                    if ok:
                        last_reliability = frame
            combined = compose_frame(
                last_plain,
                last_reliability,
                plain["metadata"].get("success_status", ""),
                reliability["metadata"].get("success_status", ""),
                index,
                log_at(plain_logs, index),
                log_at(reliability_logs, index),
            )
            writer.write(combined)
            for label, contact_index in contact_indices.items():
                if index == contact_index:
                    contact_frames[label] = combined.copy()
    finally:
        writer.release()
        plain_capture.release()
        reliability_capture.release()

    try:
        subprocess.run(
            [
                find_ffmpeg(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temporary_path),
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )
    finally:
        temporary_path.unlink(missing_ok=True)
    return contact_frames


def save_contact_sheet(
    selected_frames: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
):
    columns = ("start", "reliability_terminal", "plain_terminal")
    titles = ("Start", "Reliability terminal", "Plain terminal")
    fig, axes = plt.subplots(len(TASKS), len(columns), figsize=(18, 13))
    for row, task in enumerate(TASKS):
        for column, (key, title) in enumerate(zip(columns, titles)):
            frame = selected_frames[task.name][key]
            axes[row, column].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[row, column].set_title(f"{task.name}: {title}")
            axes[row, column].axis("off")
    fig.tight_layout()
    fig.savefig(
        output_dir / "matched_rollouts_contact_sheet.png",
        dpi=160,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "matched_rollouts_contact_sheet.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def write_selection_report(records: list[dict], output_dir: Path):
    fieldnames = [
        "task",
        "episode_id",
        "distance_improvement",
        "plain_status",
        "plain_final_distance",
        "reliability_status",
        "reliability_final_distance",
        "plain_run",
        "reliability_run",
        "video",
    ]
    with (output_dir / "matched_rollouts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# Matched Rollout Selection",
        "",
        "| Task | Episode | Plain | Reliability | Final-distance improvement |",
        "|---|---|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['task']} | `{record['episode_id']}` | "
            f"{record['plain_status']} ({record['plain_final_distance']:.3f}) | "
            f"{record['reliability_status']} "
            f"({record['reliability_final_distance']:.3f}) | "
            f"{record['distance_improvement']:.3f} |"
        )
    (output_dir / "matched_rollouts.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blacklists = load_blacklists(args.blacklist_config)
    overrides = parse_episode_overrides(args.episode)
    records = []
    selected = {}

    for task in TASKS:
        plain_run = discover_latest_run(
            args.results_root, task, plain_experiment_name(task.name)
        )
        reliability_run = discover_latest_run(
            args.results_root,
            task,
            reliability_experiment_name(task.name),
            required_log_key=ALPHA_KEY,
        )
        plain_episodes = load_episode_index(
            plain_run, blacklist=blacklists[task.result_task]
        )
        reliability_episodes = load_episode_index(
            reliability_run, blacklist=blacklists[task.result_task]
        )
        episode_id, improvement = select_episode(
            plain_episodes,
            reliability_episodes,
            requested=overrides.get(task.name),
        )
        plain = plain_episodes[episode_id]
        reliability = reliability_episodes[episode_id]
        video_path = (
            args.output_dir / f"{task.name}_{episode_id.rstrip('_')}_matched.mp4"
        )
        record = {
            "task": task.name,
            "episode_id": episode_id,
            "distance_improvement": improvement,
            "plain_status": plain["metadata"].get("success_status", ""),
            "plain_final_distance": final_distance(plain),
            "reliability_status": reliability["metadata"].get("success_status", ""),
            "reliability_final_distance": final_distance(reliability),
            "plain_run": str(plain_run),
            "reliability_run": str(reliability_run),
            "video": str(video_path) if not args.skip_videos else "",
        }
        records.append(record)
        print(
            f"{task.name}: {episode_id} (final-distance improvement {improvement:.3f})"
        )
        if not args.skip_videos:
            selected[task.name] = render_pair(
                task.name,
                episode_id,
                plain,
                reliability,
                video_path,
                args.fps,
            )

    write_selection_report(records, args.output_dir)
    if not args.skip_videos:
        save_contact_sheet(selected, args.output_dir)
    print(f"Wrote qualitative outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
