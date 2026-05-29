"""
Wrapper around ``main.py`` that lets callers override config keys without
writing a new YAML file every time.
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
    return yaml.safe_load(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = _REPO_ROOT / cfg_path
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

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
