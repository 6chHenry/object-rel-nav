#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

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


DEFAULT_CONDITIONS = {
    "train_noise00": "00",
    "train_noise02": "02",
}
ALPHA_KEY = "temporal_gate_alpha"
CURRENT_ALPHA_KEY = "temporal_gate_alpha_current"
LABEL_KEY = "inference_costmap_noise_applied"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze whether Reliability Gate alpha detects injected noise."
    )
    parser.add_argument("--results-root", type=Path, default=Path("out/results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/analysis/gate_detection"),
    )
    parser.add_argument(
        "--blacklist-config",
        type=Path,
        default=Path("configs/defaults.yaml"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        metavar="NAME=TRAIN_TAG",
        help=(
            "condition and experiment train tag; repeat to compare models. "
            "Defaults to train_noise00=00 and train_noise02=02."
        ),
    )
    return parser.parse_args()


def parse_conditions(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_CONDITIONS)
    conditions = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"condition must have NAME=TRAIN_TAG form, got {value!r}")
        name, train_tag = value.split("=", 1)
        if not name or not train_tag:
            raise ValueError(f"condition must have NAME=TRAIN_TAG form, got {value!r}")
        conditions[name] = train_tag
    return conditions


def experiment_name(task_name: str, train_noise: str) -> str:
    return (
        f"temporal_{task_name}_reliability_gate_diag_"
        f"train_noise{train_noise}_eval_noise02"
    )


def collect_rows(
    args,
    conditions: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    blacklists = load_blacklists(args.blacklist_config)
    rows = []
    run_manifest: dict[str, dict[str, str]] = {}
    for condition, train_noise in conditions.items():
        run_manifest[condition] = {}
        for task in TASKS:
            exp_name = experiment_name(task.name, train_noise)
            run_dir = discover_latest_run(
                args.results_root,
                task,
                exp_name,
                required_log_key=ALPHA_KEY,
            )
            run_manifest[condition][task.name] = str(run_dir)
            episodes = load_episode_index(
                run_dir, blacklist=blacklists[task.result_task]
            )
            for episode_id, episode in episodes.items():
                metadata = episode["metadata"]
                logs = load_controller_logs(episode["results_path"])
                for step_index, log in enumerate(logs):
                    if not isinstance(log, dict) or log.get(ALPHA_KEY) is None:
                        continue
                    alpha = np.asarray(log[ALPHA_KEY], dtype=float).reshape(-1)
                    if alpha.size != 6:
                        raise ValueError(
                            f"Expected K=6 gate scores in "
                            f"{episode['results_path']} step {step_index}, "
                            f"got shape {alpha.shape}"
                        )
                    current_alpha = float(log[CURRENT_ALPHA_KEY])
                    if not np.isclose(current_alpha, alpha[-1], atol=1e-6):
                        raise ValueError(
                            f"Current alpha does not match window tail in "
                            f"{episode['results_path']} step {step_index}"
                        )
                    row = {
                        "condition": condition,
                        "train_noise_tag": train_noise,
                        "task": task.name,
                        "result_task": task.result_task,
                        "run_dir": str(run_dir),
                        "episode_id": episode_id,
                        "cluster_id": f"{task.name}::{episode_id}",
                        "step": step_index,
                        "noise_injected": int(bool(log.get(LABEL_KEY, False))),
                        "alpha_current": current_alpha,
                        "detection_score": 1.0 - current_alpha,
                        "success_status": metadata.get("success_status", ""),
                        "final_distance": float(metadata.get("final_distance", "nan")),
                    }
                    row.update(
                        {
                            f"alpha_window_{index}": value
                            for index, value in enumerate(alpha)
                        }
                    )
                    rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No diagnostic gate rows were found.")
    return frame, run_manifest


def metric_value(labels: np.ndarray, scores: np.ndarray, metric: str) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    if metric == "roc_auc":
        return float(roc_auc_score(labels, scores))
    if metric == "average_precision":
        return float(average_precision_score(labels, scores))
    raise ValueError(metric)


def cluster_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    clusters = {
        cluster_id: (
            group["noise_injected"].to_numpy(dtype=int),
            group["detection_score"].to_numpy(dtype=float),
        )
        for cluster_id, group in frame.groupby("cluster_id", sort=True)
    }
    cluster_ids = np.array(sorted(clusters), dtype=object)
    if cluster_ids.size == 0:
        return float("nan"), float("nan")
    values = []
    for _ in range(samples):
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        labels = np.concatenate([clusters[key][0] for key in selected])
        scores = np.concatenate([clusters[key][1] for key in selected])
        value = metric_value(labels, scores, metric)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(np.percentile(values, [2.5, 97.5]).astype(float))


def paired_delta_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    left_clusters = {
        key: (
            group["noise_injected"].to_numpy(dtype=int),
            group["detection_score"].to_numpy(dtype=float),
        )
        for key, group in left.groupby("cluster_id", sort=True)
    }
    right_clusters = {
        key: (
            group["noise_injected"].to_numpy(dtype=int),
            group["detection_score"].to_numpy(dtype=float),
        )
        for key, group in right.groupby("cluster_id", sort=True)
    }
    common = np.array(sorted(set(left_clusters) & set(right_clusters)), dtype=object)
    if common.size == 0:
        return float("nan"), float("nan"), 0
    values = []
    for _ in range(samples):
        selected = rng.choice(common, size=len(common), replace=True)
        left_labels = np.concatenate([left_clusters[key][0] for key in selected])
        left_scores = np.concatenate([left_clusters[key][1] for key in selected])
        right_labels = np.concatenate([right_clusters[key][0] for key in selected])
        right_scores = np.concatenate([right_clusters[key][1] for key in selected])
        left_value = metric_value(left_labels, left_scores, metric)
        right_value = metric_value(right_labels, right_scores, metric)
        if np.isfinite(left_value) and np.isfinite(right_value):
            values.append(right_value - left_value)
    if not values:
        return float("nan"), float("nan"), len(common)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high), len(common)


def summarize(
    frame: pd.DataFrame,
    args,
    conditions: dict[str, str],
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    records = []
    scopes = [("pooled", frame)] + [
        (task.name, frame[frame["task"] == task.name]) for task in TASKS
    ]
    for condition in conditions:
        for scope, scope_frame in scopes:
            group = scope_frame[scope_frame["condition"] == condition]
            labels = group["noise_injected"].to_numpy(dtype=int)
            scores = group["detection_score"].to_numpy(dtype=float)
            for metric in ("roc_auc", "average_precision"):
                value = metric_value(labels, scores, metric)
                low, high = cluster_bootstrap(
                    group, metric, args.bootstrap_samples, rng
                )
                records.append(
                    {
                        "condition": condition,
                        "scope": scope,
                        "metric": metric,
                        "value": value,
                        "ci_low": low,
                        "ci_high": high,
                        "n_steps": len(group),
                        "n_episodes": group["cluster_id"].nunique(),
                    }
                )
            for label, label_name in ((0, "clean"), (1, "injected")):
                values = group.loc[
                    group["noise_injected"] == label, "alpha_current"
                ].to_numpy(dtype=float)
                if len(values) == 0:
                    statistics = (("mean", np.nan), ("std", np.nan), ("median", np.nan))
                else:
                    statistics = (
                        ("mean", np.mean(values)),
                        (
                            "std",
                            np.std(values, ddof=1) if len(values) > 1 else np.nan,
                        ),
                        ("median", np.median(values)),
                    )
                for statistic, value in statistics:
                    records.append(
                        {
                            "condition": condition,
                            "scope": scope,
                            "metric": f"alpha_{label_name}_{statistic}",
                            "value": float(value),
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "n_steps": len(values),
                            "n_episodes": group["cluster_id"].nunique(),
                        }
                    )

    condition_names = list(conditions)
    if len(condition_names) < 2:
        return pd.DataFrame(records)
    left_name, right_name = condition_names[:2]
    for scope, scope_frame in scopes:
        left = scope_frame[scope_frame["condition"] == left_name]
        right = scope_frame[scope_frame["condition"] == right_name]
        common = sorted(set(left["cluster_id"]) & set(right["cluster_id"]))
        for metric in ("roc_auc", "average_precision"):
            left_common = left[left["cluster_id"].isin(common)]
            right_common = right[right["cluster_id"].isin(common)]
            value = metric_value(
                right_common["noise_injected"].to_numpy(dtype=int),
                right_common["detection_score"].to_numpy(dtype=float),
                metric,
            ) - metric_value(
                left_common["noise_injected"].to_numpy(dtype=int),
                left_common["detection_score"].to_numpy(dtype=float),
                metric,
            )
            low, high, n_common = paired_delta_bootstrap(
                left, right, metric, args.bootstrap_samples, rng
            )
            records.append(
                {
                    "condition": f"{right_name}_minus_{left_name}",
                    "scope": scope,
                    "metric": metric,
                    "value": value,
                    "ci_low": low,
                    "ci_high": high,
                    "n_steps": min(len(left_common), len(right_common)),
                    "n_episodes": n_common,
                }
            )
    return pd.DataFrame(records)


def plot_curves(
    frame: pd.DataFrame,
    output_dir: Path,
    conditions: dict[str, str],
):
    scopes = [("pooled", frame)] + [
        (task.name, frame[frame["task"] == task.name]) for task in TASKS
    ]
    palette = ["#4C78A8", "#E45756", "#72B7B2", "#F2CF5B"]
    colors = {
        condition: palette[index % len(palette)]
        for index, condition in enumerate(conditions)
    }
    fig, axes = plt.subplots(2, len(scopes), figsize=(18, 7))
    for column, (scope, scope_frame) in enumerate(scopes):
        for condition in conditions:
            group = scope_frame[scope_frame["condition"] == condition]
            labels = group["noise_injected"].to_numpy(dtype=int)
            scores = group["detection_score"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            axes[0, column].plot(fpr, tpr, color=colors[condition], label=condition)
            axes[1, column].plot(
                recall, precision, color=colors[condition], label=condition
            )
        axes[0, column].plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes[0, column].set_title(scope)
        axes[0, column].set_xlabel("False-positive rate")
        axes[0, column].set_ylabel("True-positive rate")
        axes[1, column].set_xlabel("Recall")
        axes[1, column].set_ylabel("Precision")
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "roc_pr_curves")


def plot_alpha_distribution(
    frame: pd.DataFrame,
    output_dir: Path,
    conditions: dict[str, str],
):
    labels = []
    values = []
    for condition in conditions:
        for injected, label in ((0, "clean"), (1, "injected")):
            labels.append(f"{condition}\n{label}")
            values.append(
                frame.loc[
                    (frame["condition"] == condition)
                    & (frame["noise_injected"] == injected),
                    "alpha_current",
                ].to_numpy(dtype=float)
            )
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.violinplot(values, showmeans=True, showextrema=True)
    axis.set_xticks(range(1, len(labels) + 1), labels)
    axis.set_ylabel("Current-frame reliability alpha")
    axis.set_ylim(0.0, 1.02)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir / "alpha_distributions")


def plot_representative_traces(
    frame: pd.DataFrame,
    output_dir: Path,
    conditions: dict[str, str],
):
    condition_names = list(conditions)
    focus_condition = condition_names[-1]
    fig, axes = plt.subplots(
        len(TASKS),
        len(condition_names),
        figsize=(7 * len(condition_names), 10),
        sharey=True,
        squeeze=False,
    )
    for row, task in enumerate(TASKS):
        task_frame = frame[frame["task"] == task.name]
        episode_scores = (
            task_frame.groupby(["condition", "episode_id"])
            .apply(
                lambda group: (
                    group.loc[group["noise_injected"] == 0, "alpha_current"].mean()
                    - group.loc[group["noise_injected"] == 1, "alpha_current"].mean()
                ),
                include_groups=False,
            )
            .dropna()
        )
        if focus_condition not in episode_scores.index.get_level_values(0):
            raise RuntimeError(
                f"No {focus_condition} episode with both clean and injected "
                f"steps for task {task.name}."
            )
        focus_scores = episode_scores.loc[focus_condition]
        episode_id = focus_scores.idxmax()
        for column, condition in enumerate(condition_names):
            group = task_frame[
                (task_frame["condition"] == condition)
                & (task_frame["episode_id"] == episode_id)
            ]
            axis = axes[row, column]
            axis.plot(group["step"], group["alpha_current"], color="#4C78A8")
            injected = group[group["noise_injected"] == 1]
            axis.scatter(
                injected["step"],
                injected["alpha_current"],
                color="#E45756",
                s=12,
                label="injected",
                zorder=3,
            )
            axis.set_title(f"{task.name}: {condition}\n{episode_id}")
            axis.set_xlabel("Step")
            axis.set_ylabel("Current alpha")
            axis.set_ylim(0.0, 1.02)
    axes[0, 0].legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "representative_gate_traces")


def save_figure(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, output_path: Path):
    metrics = summary[summary["metric"].isin(["roc_auc", "average_precision"])].copy()
    lines = [
        "# Reliability Gate Detection Results",
        "",
        "Detection score is `1 - alpha_current`; positive labels are steps "
        "where inference costmap noise was injected.",
        "",
        "| Condition | Scope | Metric | Value | 95% episode-bootstrap CI | Episodes | Steps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.scope} | {row.metric} | "
            f"{row.value:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] | "
            f"{row.n_episodes} | {row.n_steps} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    conditions = parse_conditions(args.condition)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, run_manifest = collect_rows(args, conditions)
    summary = summarize(frame, args, conditions)

    frame.to_csv(args.output_dir / "gate_scores_by_step.csv", index=False)
    summary.to_csv(args.output_dir / "gate_detection_summary.csv", index=False)
    (args.output_dir / "gate_detection_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, allow_nan=True),
        encoding="utf-8",
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    write_markdown(summary, args.output_dir / "gate_detection_report.md")
    plot_curves(frame, args.output_dir, conditions)
    plot_alpha_distribution(frame, args.output_dir, conditions)
    plot_representative_traces(frame, args.output_dir, conditions)

    pooled = summary[
        (summary["scope"] == "pooled")
        & summary["metric"].isin(["roc_auc", "average_precision"])
    ]
    print(pooled.to_string(index=False))
    print(f"Wrote gate analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
