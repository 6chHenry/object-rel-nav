from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


RUN_DIR_PATTERN = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}_.+")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    result_task: str


TASKS = (
    TaskSpec("imitate", "original"),
    TaskSpec("alt_goal", "alt_goal"),
    TaskSpec("shortcut", "via_alt_goal"),
    TaskSpec("reverse", "original_reverse"),
)


def parse_metadata(path: Path) -> dict[str, str]:
    metadata = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def episode_identifier(episode_dir: Path) -> str:
    return episode_dir.name.split("__", 1)[0] + "_"


def load_blacklists(config_path: Path) -> dict[str, set[str]]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)["episode_blacklists"]
    common = set(raw.get("all_tasks", []))
    return {
        task.result_task: common | set(raw.get(task.result_task, [])) for task in TASKS
    }


def discover_latest_run(
    results_root: Path,
    task: TaskSpec,
    exp_name: str,
    required_log_key: str | None = None,
) -> Path:
    base = results_root / task.result_task / exp_name / "val" / "hard"
    candidates = sorted(
        path
        for path in base.glob("*")
        if path.is_dir() and RUN_DIR_PATTERN.match(path.name)
    )
    if not candidates:
        raise FileNotFoundError(f"No result run found under {base}")
    if required_log_key is None:
        return candidates[-1]
    for candidate in reversed(candidates):
        if run_has_log_key(candidate, required_log_key):
            return candidate
    raise FileNotFoundError(
        f"No run under {base} contains controller log key {required_log_key!r}"
    )


def run_has_log_key(run_dir: Path, key: str) -> bool:
    for episode_dir in iter_episode_dirs(run_dir):
        results_path = episode_dir / "results_dict.npz"
        if not results_path.is_file():
            continue
        with np.load(results_path, allow_pickle=True) as data:
            for log in data["controller_logs"]:
                if isinstance(log, dict) and key in log:
                    return True
    return False


def iter_episode_dirs(run_dir: Path) -> Iterable[Path]:
    return sorted(
        path for path in run_dir.iterdir() if path.is_dir() and path.name != "summary"
    )


def load_episode_index(
    run_dir: Path,
    blacklist: set[str] | None = None,
) -> dict[str, dict]:
    blacklist = blacklist or set()
    episodes = {}
    for episode_dir in iter_episode_dirs(run_dir):
        identifier = episode_identifier(episode_dir)
        if identifier in blacklist:
            continue
        metadata_path = episode_dir / "metadata.txt"
        results_path = episode_dir / "results_dict.npz"
        video_path = episode_dir / "repeat.mp4"
        if not metadata_path.is_file() or not results_path.is_file():
            continue
        metadata = parse_metadata(metadata_path)
        episodes[identifier] = {
            "episode_dir": episode_dir,
            "metadata": metadata,
            "results_path": results_path,
            "video_path": video_path,
        }
    return episodes


def load_controller_logs(results_path: Path) -> list[dict | None]:
    with np.load(results_path, allow_pickle=True) as data:
        return list(data["controller_logs"])
