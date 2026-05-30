"""
Training entry point for the temporal-aggregation ObjectReact variant.

The script wraps the upstream ViNT_Dataset (which already supports
``goal_uses_context=True``) and trains our ``GNMTemporal`` controller with
the standard waypoint-regression loss.  We deliberately keep this script
small and self-contained instead of monkey-patching the upstream
``train.py``: only the model class and the in-model noise augmentation
differ from upstream, so a plain training loop is enough.

Usage
-----
    python -m temporal_objectreact.train_temporal --config configs/train/temporal_gru.yaml

The config file follows the same schema as the upstream object-react
training configs, with three extra top-level keys:

    model_type: gnm_temporal      # selects GNMTemporal instead of GNM
    temporal_aggregator: gated_gru  # mean | ema | gru | gated_gru | gru_no_gate
    noise_p: 0.2                  # prob of corrupting one frame per sample
    noise_mode: zero              # zero | gauss
    init_from: <path/to/latest.pth>  # optional warm-start
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader

# Wire up the upstream package.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_TRAIN_PKG = _REPO_ROOT / "libs" / "control" / "object_react" / "train"
sys.path.insert(0, str(_TRAIN_PKG))

from vint_train.data.vint_dataset import ViNT_Dataset  # noqa: E402
from vint_train.training.train_utils import get_goal_image  # noqa: E402

from temporal_objectreact.gnm_temporal import GNMTemporal  # noqa: E402


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_datasets(config, kwargs):
    """Construct train/test datasets exactly as the upstream loader does."""
    train_datasets = []
    test_loaders = {}
    if "context_type" not in config:
        config["context_type"] = "temporal"

    for name, dc in config["datasets"].items():
        dc.setdefault("negative_mining", True)
        dc.setdefault("goals_per_obs", 1)
        dc.setdefault("end_slack", 0)
        dc.setdefault("waypoint_spacing", 1)

        for split in ("train", "test"):
            if split not in dc:
                continue
            ds = ViNT_Dataset(
                data_folder=dc["data_folder"],
                data_split_folder=dc[split],
                dataset_name=name,
                image_size=config["image_size"],
                waypoint_spacing=dc["waypoint_spacing"],
                min_dist_cat=config["distance"]["min_dist_cat"],
                max_dist_cat=config["distance"]["max_dist_cat"],
                min_action_distance=config["action"]["min_dist_cat"],
                max_action_distance=config["action"]["max_dist_cat"],
                negative_mining=dc["negative_mining"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                context_type=config["context_type"],
                end_slack=dc["end_slack"],
                goals_per_obs=dc["goals_per_obs"],
                normalize=config["normalize"],
                **kwargs,
            )
            if split == "train":
                train_datasets.append(ds)
            else:
                test_loaders[f"{name}_{split}"] = DataLoader(
                    ds,
                    batch_size=config.get("eval_batch_size", config["batch_size"]),
                    shuffle=False,
                    num_workers=config.get(
                        "eval_num_workers", config.get("num_workers", 4)
                    ),
                    drop_last=False,
                    pin_memory=True,
                )

    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, test_loaders


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(config, kwargs):
    model = GNMTemporal(
        config["context_size"],
        config["len_traj_pred"],
        config["learn_angle"],
        config["obs_encoding_size"],
        config["goal_encoding_size"],
        temporal_aggregator=config.get("temporal_aggregator", "gated_gru"),
        temporal_ema_lambda=config.get("temporal_ema_lambda", 0.7),
        noise_p=config.get("noise_p", 0.0),
        noise_mode=config.get("noise_mode", "zero"),
        **kwargs,
    )
    init_from = config.get("init_from", None)
    if init_from:
        model.load_pretrained_backbone(init_from, strict=False)
    if config.get("freeze_backbone", False):
        for name, p in model.named_parameters():
            if not name.startswith("aggregator"):
                p.requires_grad_(False)
        print("[train_temporal] froze everything except aggregator.* parameters")
    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def waypoint_loss(action_pred, action_label, mask, learn_angle):
    """Same loss shape upstream uses: MSE on (dx, dy) and (cos, sin)."""
    # action_pred: (B, T, 4 or 2);  action_label: (B, T, 4 or 2 after calc_sin_cos)
    diff = (action_pred - action_label) ** 2  # (B, T, C)
    if mask is not None:
        diff = diff * mask.view(-1, 1, 1)
    return diff.mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, transform, *, train: bool,
              log_every: int = 50, max_iters: int | None = None):
    model.train(mode=train)
    total_loss, n = 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if max_iters is not None and i >= max_iters:
            break
        (
            obs_image,
            goal_image,
            action_label,
            _distance,
            _goal_pos,
            _dataset_idx,
            action_mask,
        ) = batch

        obs_image = obs_image.to(device, non_blocking=True)
        goal_image = goal_image.to(device, non_blocking=True)
        action_label = action_label.to(device, non_blocking=True)
        action_mask = action_mask.to(device, non_blocking=True)

        # Apply the same vis-stripping the upstream training loop does.
        goal_image, _viz = get_goal_image(goal_image, "image_mask_enc", transform, device)
        obs_image = transform(obs_image)

        with torch.set_grad_enabled(train):
            _dist_pred, action_pred = model(obs_image, goal_image)
            loss = waypoint_loss(
                action_pred, action_label, action_mask, learn_angle=True
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * obs_image.size(0)
        n += obs_image.size(0)
        if (i + 1) % log_every == 0:
            print(
                f"  step {i+1:5d} | loss {loss.item():.5f} | "
                f"avg {total_loss / n:.5f} | "
                f"{(i+1) / max(time.time() - t0, 1e-6):.2f} it/s"
            )

    avg = total_loss / max(n, 1)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None,
                    help="output dir; defaults to logs/<run_name>")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_iters", type=int, default=None,
                    help="cap iterations per epoch (useful for smoke tests)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.out or f"logs/{config['run_name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    seed = config.get("seed", 42)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    kwargs = {
        "predict_dists": config.get("predict_dists", True),
        "precomputed_filename": config.get("precomputed_filename", None),
        "pl_perturb_ratio": config.get("pl_perturb_ratio", 0.0),
        "pl_perturb_type": config.get("pl_perturb_type", "max_val"),
        "mask_crop_ratio": config.get("mask_crop_ratio", 1.0),
        "use_mask_grad": config.get("use_mask_grad", False),
        "goal_type": config.get("goal_type", "image_mask_enc"),
        "obs_type": config.get("obs_type", "disabled"),
        "dims": config.get("dims", 8),
        "goal_uses_context": True,  # mandatory for temporal aggregation
    }

    print("Loading datasets…")
    train_loader, test_loaders = build_datasets(config, kwargs)
    print(f"  train batches={len(train_loader)}, test loaders={list(test_loaders)}")

    print("Building model…")
    model = build_model(config, kwargs).to(args.device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train/1e6:.2f}M")

    # EMA aggregator + frozen backbone → no learnable params. Skip training and
    # dump the init weights so eval has a checkpoint to load.
    if n_train == 0:
        print("[train_temporal] no trainable params; saving init checkpoint and exiting.")
        torch.save({"model": model.state_dict(), "epoch": 0},
                   out_dir / "latest.pth")
        torch.save({"model": model.state_dict(), "epoch": 0},
                   out_dir / "epoch_001.pth")
        with open(out_dir / "history.json", "w") as f:
            json.dump([{"epoch": 0, "train_loss": None, "note": "no trainable params"}], f, indent=2)
        print(f"Done. checkpoint in {out_dir}")
        return

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["lr"]),
    )

    # We do the input normalisation that upstream applies in train_eval_loop.
    from torchvision import transforms as T
    transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    history = []
    epochs = int(config.get("epochs", 30))
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        train_loss = run_epoch(
            model, train_loader, optim, args.device, transform,
            train=True, max_iters=args.max_iters,
        )
        line = {"epoch": epoch + 1, "train_loss": train_loss}
        for name, dl in test_loaders.items():
            val_loss = run_epoch(
                model, dl, optim, args.device, transform,
                train=False, max_iters=args.max_iters,
            )
            line[f"val_{name}"] = val_loss
        history.append(line)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        torch.save({"model": model.state_dict(), "epoch": epoch},
                   out_dir / "latest.pth")
        if (epoch + 1) % config.get("save_every", 5) == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       out_dir / f"epoch_{epoch+1:03d}.pth")
        print("  ", json.dumps(line))

    print(f"\nDone. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
