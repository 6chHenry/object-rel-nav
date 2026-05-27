"""
Inference-time controller that uses ``GNMTemporal`` instead of the upstream
single-frame GNM.

This is a thin subclass of ``ObjRelLearntController`` from
``libs/control/objectreact.py``.  We only replace:

  1. the model class (``GNMTemporal``);
  2. how the per-step goal history is stacked before calling the model;
  3. checkpoint loading (we load *our* checkpoint, not the upstream HF one).

The rest of the pipeline (mapping, perception, planner) is reused
verbatim.  The class is registered on first import so that
``configs/object_react_temporal.yaml`` can refer to it by name.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from libs.control.objectreact import ObjRelLearntController  # noqa: E402
from temporal_objectreact.gnm_temporal import GNMTemporal  # noqa: E402


class ObjRelTemporalLearntController(ObjRelLearntController):
    """ObjectReact controller with a temporal aggregator over the costmap."""

    def __init__(self, config, **kwargs):
        # Load config dict ourselves so we can sniff our extra keys.
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        elif isinstance(config, dict):
            cfg = config
        else:
            raise ValueError(f"config must be a path or dict, got {type(config)}")

        # Required to keep the per-frame goal history that the temporal
        # aggregator consumes.
        if not cfg.get("goal_uses_context", False):
            print("[ObjRelTemporalLearntController] forcing goal_uses_context=True")
            cfg["goal_uses_context"] = True

        # Call parent init, which builds an ObjectReact GNM and downloads
        # the upstream HF checkpoint.  We then swap that model out for
        # ours.  This is wasteful (a one-time download) but keeps the
        # parent constructor's path setup untouched.
        super().__init__(cfg, **kwargs)

        # Build our model.  Keep all the kwargs the parent used.
        self.config["temporal_aggregator"] = cfg.get(
            "temporal_aggregator", "gated_gru"
        )
        self.config["temporal_ema_lambda"] = cfg.get("temporal_ema_lambda", 0.7)
        new_model = GNMTemporal(
            self.config["context_size"],
            self.config["len_traj_pred"],
            self.config["learn_angle"],
            self.config["obs_encoding_size"],
            self.config["goal_encoding_size"],
            temporal_aggregator=self.config["temporal_aggregator"],
            temporal_ema_lambda=self.config["temporal_ema_lambda"],
            noise_p=0.0,  # never inject noise at inference time
            goal_type=self.config.get("goal_type", "image_mask_enc"),
            obs_type=self.config.get("obs_type", "disabled"),
            dims=self.config.get("dims", 8),
            goal_uses_context=True,
            use_mask_grad=self.config.get("use_mask_grad", False),
            predict_dists=self.config.get("predict_dists", False),
        )
        # Load our finetuned weights if available.
        load_run = self.config.get("load_run", None)
        if load_run is None:
            print("[ObjRelTemporalLearntController] WARNING: no load_run set; "
                  "falling back to the upstream backbone with a randomly "
                  "initialised aggregator.")
            # Copy as much as possible from the parent model so we get the
            # ObjectReact pretrained encoder/head.
            new_model.load_state_dict(self.model.state_dict(), strict=False)
        else:
            ckpt_path = Path(load_run)
            if ckpt_path.is_dir():
                ckpt_path = ckpt_path / "latest.pth"
            print(f"[ObjRelTemporalLearntController] loading checkpoint {ckpt_path}")
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ck.get("model", ck)
            missing, unexpected = new_model.load_state_dict(state, strict=False)
            if missing:
                print(f"  missing keys: {len(missing)}")
            if unexpected:
                print(f"  unexpected keys: {len(unexpected)}")
        self.model = new_model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Override ready_goal so that the goal tensor carries K = context_size
    # + 1 stacked costmaps even though upstream defaults to single-frame.
    # ------------------------------------------------------------------
    def ready_goal(self, goal_data):
        from PIL import Image  # local to avoid hard import in unit tests
        import torch as _torch
        from libs.control.object_react.train.vint_train.training.train_utils import (
            get_goal_image,
        )

        # Encode the current frame's costmap exactly as the parent does.
        goal_enc, self.goal_mask_vis = self.encode_goal(goal_data)

        # Maintain a list of the last K encodings; pad by replication while
        # the history is shorter than K.
        self.maintain_history(
            _torch.as_tensor(goal_enc, dtype=_torch.float32), self.goal_history
        )
        goal_enc_stack = _torch.as_tensor(
            _torch.cat(self.goal_history), dtype=_torch.float32
        )

        # Prepend a 3-channel vis as upstream does (the model strips it).
        goal_vis = goal_enc_stack[:3, :, :]
        if self.config["dims"] < 3:
            goal_vis = goal_enc_stack[:1].repeat(3, 1, 1)
        goal_image = _torch.as_tensor(
            np.concatenate([goal_vis, goal_enc_stack], axis=0),
            dtype=_torch.float32,
        )[None, ...]

        goal_image, _ = get_goal_image(
            goal_image, self.goal_type, self.transform, self.device
        )
        return goal_image
