"""
GNMTemporal: ObjectReact's GNM controller with a temporal costmap aggregator.

The architectural change is localised: we keep the upstream ``GoalEncoder``
(which turns a single H/2 x W/2 multi-channel costmap into a 1024-d vector)
and the prediction head untouched.  The only difference is that the
``goal_uses_context=True`` branch no longer averages the per-frame
encodings; instead it feeds them through a GRU with an optional
confidence gate.

This lets us load the public ObjectReact pretrained weights for the
encoder + head (initialisation), and train only the new temporal module
when ``train_temporal_only=True``.  All other parameters are saved/loaded
through the same checkpoint format as upstream.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the upstream object_react package importable.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_TRAIN_PKG = _REPO_ROOT / "libs" / "control" / "object_react" / "train"
if str(_TRAIN_PKG) not in sys.path:
    sys.path.insert(0, str(_TRAIN_PKG))

from vint_train.models.gnm.gnm import GNM  # noqa: E402  (path injection above)

from .temporal_aggregator import (  # noqa: E402
    EMATemporalAggregator,
    GRUTemporalAggregator,
    ReliabilityGatedGRUTemporalAggregator,
    TemporalCostmapAggregator,
)


def _make_aggregator(kind: str, dim: int, ema_lambda: float = 0.7):
    """Factory for the temporal aggregator."""
    kind = kind.lower()
    if kind == "mean":
        return None  # signals "use upstream mean-pool" (no extra params)
    if kind == "ema":
        return EMATemporalAggregator(lam=ema_lambda)
    if kind == "gru":
        return GRUTemporalAggregator(dim=dim)
    if kind in {"gated_gru", "gru_gated", "tca"}:
        return TemporalCostmapAggregator(dim=dim, ema_lambda=ema_lambda, use_gate=True)
    if kind in {"reliability_gated_gru", "rel_gated_gru", "rgru"}:
        return ReliabilityGatedGRUTemporalAggregator(
            dim=dim,
            ema_lambda=ema_lambda,
        )
    if kind == "gru_no_gate":
        return TemporalCostmapAggregator(dim=dim, ema_lambda=ema_lambda, use_gate=False)
    raise ValueError(f"Unknown temporal_aggregator: {kind}")


class GNMTemporal(GNM):
    """GNM with a pluggable temporal aggregator over the goal embedding.

    The model expects ``goal_uses_context=True`` and a goal tensor that
    stacks ``context_size + 1`` consecutive WayObject Costmaps along the
    channel dimension, exactly as upstream does.  The only change is how
    those (K = context_size + 1) per-frame embeddings are combined.

    Extra kwargs:
        temporal_aggregator (str): "mean" (upstream behaviour), "ema",
            "gru", "gated_gru", "reliability_gated_gru", or "gru_no_gate".
        temporal_ema_lambda (float): EMA decay used by EMA / gate variants.
    """

    def __init__(self, *args, **kwargs):
        # Force the parent class into the context-aware branch.  Users
        # typically pass goal_uses_context=True via the config; we just
        # double-check here so that mistakes fail loudly instead of silently
        # falling back to single-frame behaviour.
        if not kwargs.get("goal_uses_context", False):
            raise ValueError(
                "GNMTemporal requires goal_uses_context=True so that the goal "
                "tensor carries the full context window."
            )

        self.temporal_kind = kwargs.pop("temporal_aggregator", "gated_gru")
        self.temporal_ema_lambda = kwargs.pop("temporal_ema_lambda", 0.7)
        # Noise augmentation: with prob p_noise drop ONE random frame in the
        # window during training, simulating a transient perception failure.
        self.noise_p = float(kwargs.pop("noise_p", 0.0))
        self.noise_mode = kwargs.pop("noise_mode", "zero")  # {"zero", "gauss"}

        super().__init__(*args, **kwargs)

        self.aggregator = _make_aggregator(
            self.temporal_kind,
            dim=self.goal_encoding_size,
            ema_lambda=self.temporal_ema_lambda,
        )

    # ------------------------------------------------------------------
    # We only override forward(); the constructor and all other parts of
    # the parent GNM are reused.  We rewrite the goal-encoding branch and
    # leave the obs branch and the prediction heads untouched.
    # ------------------------------------------------------------------
    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # ---- observation branch (copied verbatim from GNM.forward) -----
        if self.obs_type == "image":
            obs_encoding = self.obs_mobilenet(obs_img)
            obs_encoding = self.flatten(obs_encoding)
            obs_encoding = self.compress_observation(obs_encoding)
        elif self.obs_type == "image_mask_enc":
            obs_encoding = self.obs_mobilenet(obs_img)
        elif self.obs_type == "disabled":
            obs_encoding = None
        else:
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

        # ---- goal branch with temporal aggregation --------------------
        if self.goal_type == "image_mask_enc":
            # goal_img shape: (B, dims_per_frame * K, H/2, W/2) where K =
            # context_size + 1 and dims_per_frame already accounts for the
            # optional mask-gradient channel.  Split along channel dim, run
            # the (shared) encoder over every chunk in parallel, stack into
            # (B, K, D), then aggregate.
            K = self.context_size + 1
            per_frame = goal_img.shape[1] // K
            chunks = list(goal_img.split(per_frame, dim=1))
            # Keep only the first K chunks of equal width; any remainder is
            # an artefact of upstream prepending vis channels and is ignored
            # in our model (the upstream get_goal_image typically strips
            # those vis channels before forward, so this guard rarely fires).
            chunks = [c for c in chunks if c.shape[1] == per_frame][:K]
            assert len(chunks) == K, (
                f"Expected {K} chunks of width {per_frame}, got {len(chunks)} "
                f"from goal_img with {goal_img.shape[1]} channels."
            )

            # Optional input-level noise augmentation (training only).
            if self.training and self.noise_p > 0.0:
                B = goal_img.shape[0]
                drop_mask = torch.rand(B, device=goal_img.device) < self.noise_p
                if drop_mask.any():
                    # Pick one random frame per sample to corrupt.
                    drop_idx = torch.randint(0, K, (B,), device=goal_img.device)
                    for b in range(B):
                        if not drop_mask[b]:
                            continue
                        if self.noise_mode == "zero":
                            chunks[drop_idx[b].item()][b] = 0.0
                        elif self.noise_mode == "gauss":
                            chunks[drop_idx[b].item()][b] = torch.randn_like(
                                chunks[drop_idx[b].item()][b]
                            )
                        else:
                            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

            stacked = torch.stack(
                [self.goal_mobilenet(c) for c in chunks], dim=1
            )  # (B, K, D)

            if self.aggregator is None:
                goal_encoding = stacked.mean(dim=1)
            else:
                out = self.aggregator(stacked)
                goal_encoding = out[0] if isinstance(out, tuple) else out
        elif self.goal_type == "image":
            # The original upstream behaviour is not what we want to extend
            # (it stacks obs + goal images and runs a single conv stack).
            # We fall back to the parent implementation for completeness.
            obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
            goal_encoding = self.goal_mobilenet(obs_goal_input)
            goal_encoding = self.flatten(goal_encoding)
            goal_encoding = self.compress_goal(goal_encoding)
        elif self.goal_type == "disabled":
            goal_encoding = None
        else:
            raise ValueError(f"Unknown goal_type: {self.goal_type}")

        # ---- fusion + heads (copied verbatim from GNM.forward) ---------
        if obs_encoding is None:
            z = goal_encoding
        elif goal_encoding is None:
            z = obs_encoding
        else:
            z = torch.cat([obs_encoding, goal_encoding], dim=1)
        z = self.linear_layers(z)
        if self.kwargs.get("predict_dists", True):
            dist_pred = self.dist_predictor(z)
        else:
            dist_pred = None
        action_pred = self.action_predictor(z)

        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2].cpu(), dim=1).to(
            action_pred.device
        )
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)
        return dist_pred, action_pred

    # ------------------------------------------------------------------
    # Convenience: load pretrained ObjectReact weights into the encoder
    # and head while leaving the new aggregator at its random init.
    # ------------------------------------------------------------------
    def load_pretrained_backbone(self, checkpoint_path: str, strict: bool = False):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        own_keys = set(self.state_dict().keys())
        missing = [k for k in state if k not in own_keys]
        load_state = {k: v for k, v in state.items() if k in own_keys}
        result = self.load_state_dict(load_state, strict=False)
        print(
            f"[GNMTemporal] loaded {len(load_state)} keys "
            f"({len(missing)} keys in checkpoint had no match); "
            f"aggregator (kind={self.temporal_kind}) kept at random init."
        )
        return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    ctx = 4  # so K = 5
    dims = 8
    H, W = 120, 160  # after /2 inside the encoder
    model = GNMTemporal(
        context_size=ctx,
        len_traj_pred=10,
        learn_angle=True,
        obs_encoding_size=1024,
        goal_encoding_size=1024,
        goal_type="image_mask_enc",
        obs_type="disabled",
        dims=dims,
        goal_uses_context=True,
        use_mask_grad=False,
        predict_dists=False,
        temporal_aggregator="gated_gru",
    )
    B = 2
    goal_img = torch.randn(B, dims * (ctx + 1), H // 2, W // 2)
    obs_img = torch.zeros(B, 3, H, W)  # ignored when obs_type='disabled'
    dist, action = model(obs_img, goal_img)
    print("dist:", None if dist is None else dist.shape, "action:", action.shape)
