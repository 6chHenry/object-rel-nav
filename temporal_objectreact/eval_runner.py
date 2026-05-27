"""
Wrapper around ``main.py`` that lets callers override config keys without
writing a new YAML file every time.

Usage:
    python -m temporal_objectreact.eval_runner \\
        --config configs/object_react_temporal.yaml \\
        --set task_type=alt_goal reverse=False max_steps=200 \\
              controller.load_run=logs/temporal_gated_gru/latest.pth \\
              controller.temporal_aggregator=gated_gru

Each ``key=value`` token is parsed with ``yaml.safe_load`` so that strings,
bools, ints and floats round-trip cleanly.  Dotted keys (``controller.foo``)
descend into nested dicts and create intermediate ones if missing.

The merged config is written to a fresh temp file and passed to ``main.py``
via ``-c``; we avoid importing ``main`` directly because it triggers global
side effects (logger setup, seeding, habitat init) on import.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _set_nested(d: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _parse_value(raw: str):
    # YAML handles bools/ints/floats/null/strings uniformly.
    return yaml.safe_load(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--set", nargs="*", default=[],
                    help="key=value overrides; key may be dotted, e.g. controller.load_run=foo")
    ap.add_argument("--keep-temp", action="store_true",
                    help="don't delete the merged temp config (useful for debugging)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _REPO_ROOT / cfg_path
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Split overrides: top-level cfg vs. controller-yaml.
    # Keys like "controller.foo" where foo != "config_file" get rewritten
    # into the separate controller config file pointed to by
    # cfg["controller"]["config_file"].
    ctrl_overrides: dict = {}
    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"override must be key=value, got: {kv!r}")
        k, v = kv.split("=", 1)
        val = _parse_value(v)
        if k.startswith("controller.") and k != "controller.config_file":
            ctrl_overrides[k[len("controller."):]] = val
        else:
            _set_nested(cfg, k, val)

    tmp_paths = []
    try:
        if ctrl_overrides:
            ctrl_cfg_rel = cfg["controller"]["config_file"]
            ctrl_cfg_path = _REPO_ROOT / ctrl_cfg_rel
            with open(ctrl_cfg_path, "r") as f:
                ctrl_cfg = yaml.safe_load(f)
            for k, v in ctrl_overrides.items():
                _set_nested(ctrl_cfg, k, v)
            fd, ctrl_tmp = tempfile.mkstemp(
                prefix="eval_runner_ctrl_", suffix=".yaml", dir=str(_REPO_ROOT)
            )
            tmp_paths.append(ctrl_tmp)
            with os.fdopen(fd, "w") as f:
                yaml.safe_dump(ctrl_cfg, f, sort_keys=False)
            cfg["controller"]["config_file"] = str(
                Path(ctrl_tmp).relative_to(_REPO_ROOT)
            )

        fd, tmp_path = tempfile.mkstemp(
            prefix="eval_runner_", suffix=".yaml", dir=str(_REPO_ROOT)
        )
        tmp_paths.append(tmp_path)
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        rel_tmp = Path(tmp_path).relative_to(_REPO_ROOT)
        cmd = [sys.executable, "main.py", "-c", str(rel_tmp)]
        print(f"[eval_runner] {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=str(_REPO_ROOT))
        sys.exit(rc)
    finally:
        if not args.keep_temp:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.unlink(p)


if __name__ == "__main__":
    main()
