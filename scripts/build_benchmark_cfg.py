#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIGS = {
    "imitate": REPO_ROOT / "configs/benchmarks/object_react_gt_imitate.yaml",
    "alt_goal": REPO_ROOT / "configs/benchmarks/object_react_gt_alt_goal.yaml",
    "shortcut": REPO_ROOT / "configs/benchmarks/object_react_gt_shortcut.yaml",
    "reverse": REPO_ROOT / "configs/benchmarks/object_react_gt_reverse.yaml",
}

MODE_OVERRIDES = {
    "clean": {
        "use_costmap_ema": False,
        "inject_costmap_noise": False,
    },
    "noise": {
        "use_costmap_ema": False,
        "inject_costmap_noise": True,
    },
    "ema": {
        "use_costmap_ema": True,
        "inject_costmap_noise": False,
    },
    "ema_noise": {
        "use_costmap_ema": True,
        "inject_costmap_noise": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(BASE_CONFIGS))
    parser.add_argument(
        "--mode", default="clean", choices=sorted(MODE_OVERRIDES)
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--ema-lambda", dest="ema_lambda", type=float, default=0.7)
    parser.add_argument("--noise-prob", dest="noise_prob", type=float, default=0.2)
    args = parser.parse_args()

    with open(BASE_CONFIGS[args.task], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.update(MODE_OVERRIDES[args.mode])
    cfg["ema_lambda"] = args.ema_lambda
    cfg["noise_prob"] = args.noise_prob

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
