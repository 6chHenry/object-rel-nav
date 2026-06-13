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

Supervised reliability-gate runs additionally use:

    gate_corruption_prob: 0.2
    gate_corruption_mode: zero
    gate_supervision_weight: 0.5
    gate_pos_weight: 4.0
    gate_history_update: gated
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
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
        gate_history_update=config.get("gate_history_update", "raw"),
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


def apply_temporal_corruption(
    goal_image: torch.Tensor,
    *,
    num_frames: int,
    probability: float,
    mode: str = "zero",
    corrupt_first_frame: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt frames independently and return the exact supervision mask."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"corruption probability must be in [0, 1], got {probability}")
    if goal_image.shape[1] % num_frames != 0:
        raise ValueError(
            f"{goal_image.shape[1]} goal channels cannot be split into "
            f"{num_frames} temporal frames"
        )

    batch_size = goal_image.shape[0]
    corruption_mask = (
        torch.rand(
            (batch_size, num_frames),
            device=goal_image.device,
            generator=generator,
        )
        < probability
    )
    if not corrupt_first_frame:
        corruption_mask[:, 0] = False

    frames = goal_image.reshape(
        batch_size,
        num_frames,
        goal_image.shape[1] // num_frames,
        *goal_image.shape[2:],
    ).clone()
    if mode == "zero":
        frames[corruption_mask] = 0.0
    elif mode == "gauss":
        noise = torch.randn(
            frames.shape,
            device=frames.device,
            dtype=frames.dtype,
            generator=generator,
        )
        frames[corruption_mask] = noise[corruption_mask]
    else:
        raise ValueError(f"Unknown gate corruption mode: {mode}")

    return frames.reshape_as(goal_image), corruption_mask


def gate_detection_metrics(
    labels: list[np.ndarray],
    scores: list[np.ndarray],
) -> tuple[float, float]:
    if not labels:
        return float("nan"), float("nan")
    all_labels = np.concatenate(labels)
    all_scores = np.concatenate(scores)
    if np.unique(all_labels).size < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(all_labels, all_scores)),
        float(average_precision_score(all_labels, all_scores)),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_epoch(
    model,
    loader,
    optimizer,
    device,
    transform,
    *,
    train: bool,
    log_every: int = 50,
    max_iters: int | None = None,
    gate_supervision_weight: float = 0.0,
    gate_corruption_prob: float = 0.0,
    gate_corruption_mode: str = "zero",
    gate_pos_weight: float = 4.0,
    gate_corruption_seed: int | None = None,
):
    model.train(mode=train)
    totals = {
        "waypoint_loss": 0.0,
        "gate_loss": 0.0,
        "total_loss": 0.0,
    }
    alpha_sums = {"clean": 0.0, "corrupt": 0.0}
    alpha_counts = {"clean": 0, "corrupt": 0}
    metric_labels: list[np.ndarray] = []
    metric_scores: list[np.ndarray] = []
    corrupted_frames = 0
    supervised_frames = 0
    n = 0
    t0 = time.time()
    corruption_generator = None
    if gate_corruption_seed is not None:
        corruption_generator = torch.Generator(device=device)
        corruption_generator.manual_seed(gate_corruption_seed)

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
        goal_image, _viz = get_goal_image(
            goal_image, "image_mask_enc", transform, device
        )
        obs_image = transform(obs_image)
        corruption_mask = None
        if gate_corruption_prob > 0.0:
            goal_image, corruption_mask = apply_temporal_corruption(
                goal_image,
                num_frames=model.context_size + 1,
                probability=gate_corruption_prob,
                mode=gate_corruption_mode,
                corrupt_first_frame=False,
                generator=corruption_generator,
            )

        with torch.set_grad_enabled(train):
            needs_gate_diagnostics = gate_supervision_weight > 0.0
            model_output = model(
                obs_image,
                goal_image,
                return_gate_diagnostics=needs_gate_diagnostics,
            )
            if needs_gate_diagnostics:
                _dist_pred, action_pred, gate_diagnostics = model_output
            else:
                _dist_pred, action_pred = model_output
                gate_diagnostics = None

            waypoint = waypoint_loss(
                action_pred, action_label, action_mask, learn_angle=True
            )
            gate_loss = waypoint.new_zeros(())
            if gate_supervision_weight > 0.0:
                if corruption_mask is None:
                    raise ValueError(
                        "gate supervision requires gate_corruption_prob > 0"
                    )
                if not gate_diagnostics:
                    raise ValueError(
                        "gate supervision requires a reliability-gated model"
                    )
                reliability_logits = gate_diagnostics["reliability_logits"]
                if reliability_logits is None:
                    raise ValueError("model did not return reliability gate logits")
                targets = corruption_mask[:, 1:].to(dtype=reliability_logits.dtype)
                corruption_logits = -reliability_logits[:, 1:]
                gate_loss = F.binary_cross_entropy_with_logits(
                    corruption_logits,
                    targets,
                    pos_weight=corruption_logits.new_tensor(gate_pos_weight),
                )
            loss = waypoint + gate_supervision_weight * gate_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = obs_image.size(0)
        totals["waypoint_loss"] += waypoint.item() * batch_size
        totals["gate_loss"] += gate_loss.item() * batch_size
        totals["total_loss"] += loss.item() * batch_size
        n += batch_size

        if gate_diagnostics and corruption_mask is not None:
            labels = corruption_mask[:, 1:]
            alpha = gate_diagnostics["alpha"][:, 1:]
            scores = torch.sigmoid(-gate_diagnostics["reliability_logits"][:, 1:])
            metric_labels.append(
                labels.detach().reshape(-1).cpu().numpy().astype(np.int8)
            )
            metric_scores.append(
                scores.detach().reshape(-1).cpu().numpy().astype(np.float32)
            )
            clean_mask = ~labels
            alpha_sums["clean"] += alpha[clean_mask].detach().sum().item()
            alpha_counts["clean"] += int(clean_mask.sum().item())
            alpha_sums["corrupt"] += alpha[labels].detach().sum().item()
            alpha_counts["corrupt"] += int(labels.sum().item())
            corrupted_frames += int(labels.sum().item())
            supervised_frames += labels.numel()

        if (i + 1) % log_every == 0:
            print(
                f"  step {i + 1:5d} | waypoint {waypoint.item():.5f} | "
                f"gate {gate_loss.item():.5f} | total {loss.item():.5f} | "
                f"avg {totals['total_loss'] / n:.5f} | "
                f"{(i + 1) / max(time.time() - t0, 1e-6):.2f} it/s"
            )

    roc_auc, average_precision = gate_detection_metrics(
        metric_labels,
        metric_scores,
    )
    metrics = {key: value / max(n, 1) for key, value in totals.items()}
    metrics.update(
        {
            "gate_roc_auc": roc_auc,
            "gate_average_precision": average_precision,
            "gate_alpha_clean": (
                alpha_sums["clean"] / alpha_counts["clean"]
                if alpha_counts["clean"]
                else float("nan")
            ),
            "gate_alpha_corrupt": (
                alpha_sums["corrupt"] / alpha_counts["corrupt"]
                if alpha_counts["corrupt"]
                else float("nan")
            ),
            "corruption_rate": (
                corrupted_frames / supervised_frames if supervised_frames else 0.0
            ),
        }
    )
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--out", default=None, help="output dir; defaults to logs/<run_name>"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--max_iters",
        type=int,
        default=None,
        help="cap iterations per epoch (useful for smoke tests)",
    )
    args = ap.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.out or f"logs/{config['run_name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    print(f"  trainable params: {n_train / 1e6:.2f}M")

    gate_supervision_weight = float(config.get("gate_supervision_weight", 0.0))
    gate_corruption_prob = float(config.get("gate_corruption_prob", 0.0))
    gate_corruption_mode = config.get("gate_corruption_mode", "zero")
    gate_pos_weight = float(config.get("gate_pos_weight", 4.0))
    gate_validation_seed = int(config.get("gate_validation_seed", 12345))
    if (
        gate_supervision_weight > 0.0
        and config.get("temporal_aggregator") != "reliability_gated_gru"
    ):
        raise ValueError(
            "gate_supervision_weight is only supported for "
            "temporal_aggregator=reliability_gated_gru"
        )
    if gate_supervision_weight > 0.0 and gate_corruption_prob <= 0.0:
        raise ValueError("gate_supervision_weight requires gate_corruption_prob > 0")

    # EMA aggregator + frozen backbone → no learnable params. Skip training and
    # dump the init weights so eval has a checkpoint to load.
    if n_train == 0:
        print(
            "[train_temporal] no trainable params; saving init checkpoint and exiting."
        )
        torch.save({"model": model.state_dict(), "epoch": 0}, out_dir / "latest.pth")
        torch.save({"model": model.state_dict(), "epoch": 0}, out_dir / "epoch_001.pth")
        with open(out_dir / "history.json", "w") as f:
            json.dump(
                [{"epoch": 0, "train_loss": None, "note": "no trainable params"}],
                f,
                indent=2,
            )
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
    best_val_total = float("inf")
    epochs = int(config.get("epochs", 30))
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        train_metrics = run_epoch(
            model,
            train_loader,
            optim,
            args.device,
            transform,
            train=True,
            max_iters=args.max_iters,
            gate_supervision_weight=gate_supervision_weight,
            gate_corruption_prob=gate_corruption_prob,
            gate_corruption_mode=gate_corruption_mode,
            gate_pos_weight=gate_pos_weight,
        )
        line = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["waypoint_loss"],
        }
        line.update({f"train_{key}": value for key, value in train_metrics.items()})
        val_total_losses = []
        for loader_index, (name, dl) in enumerate(test_loaders.items()):
            val_metrics = run_epoch(
                model,
                dl,
                optim,
                args.device,
                transform,
                train=False,
                max_iters=args.max_iters,
                gate_supervision_weight=gate_supervision_weight,
                gate_corruption_prob=gate_corruption_prob,
                gate_corruption_mode=gate_corruption_mode,
                gate_pos_weight=gate_pos_weight,
                gate_corruption_seed=gate_validation_seed + loader_index,
            )
            line[f"val_{name}"] = val_metrics["waypoint_loss"]
            line.update(
                {f"val_{name}_{key}": value for key, value in val_metrics.items()}
            )
            val_total_losses.append(val_metrics["total_loss"])
        history.append(line)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": line,
        }
        torch.save(checkpoint, out_dir / "latest.pth")
        mean_val_total = (
            float(np.mean(val_total_losses))
            if val_total_losses
            else train_metrics["total_loss"]
        )
        if mean_val_total < best_val_total:
            best_val_total = mean_val_total
            torch.save(checkpoint, out_dir / "best.pth")
            print(f"  saved best.pth (validation total loss={best_val_total:.5f})")
        if (epoch + 1) % config.get("save_every", 5) == 0:
            torch.save(checkpoint, out_dir / f"epoch_{epoch + 1:03d}.pth")
        print("  ", json.dumps(line))

    print(f"\nDone. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
