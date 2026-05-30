# ObjectReact Evaluation Summary

Date: 2026-05-30

This summary separates two different uses of EMA:

- `temporal EMA` is a temporal costmap-feature aggregator, comparable with
  GRU-style temporal aggregation.
- `costmap EMA` is a pipeline-level smoothing / noise-robustness trick. It is
  useful, but it should not be directly compared against GRU as the EMA
  baseline.

The small-protocol temporal-grid numbers below are from `../temporal.md`; they
use `step_idx=10` and do not include Alt-Goal. The local temporal EMA, plain
GRU, gated-GRU, and reliability-GRU rows use the fuller `step_idx=3` protocol
from `scripts/evaluate_objecreact.py`.

Noise setting: the local temporal aggregation evaluations below do not inject
inference-time costmap noise. The `noise` and `costmap EMA + noise` rows in the
separate costmap exploration table are the only rows with inference-time noise.
For training, gated GRU and reliability gated GRU used `noise_p=0.2`, while
plain GRU used `noise_p=0.0`; temporal EMA is parameter-free.

## Temporal Aggregation Ablation

Temporary same-protocol table from `../temporal.md`. Values are
`Success / SPL / Soft SPL` percentages.

| Method | Imitate | Shortcut | Reverse |
|---|---:|---:|---:|
| ObjectReact / single frame | 40.0 / 40.0 / 56.6 | 44.4 / 44.4 / 56.8 | 44.4 / 44.4 / 50.0 |
| temporal EMA | 40.0 / 40.0 / 56.6 | 33.3 / 33.3 / 55.3 | 44.4 / 44.4 / 49.6 |
| plain GRU | 70.0 / 70.0 / 72.5 | 11.1 / 11.1 / 54.5 | 55.6 / 55.5 / 67.5 |
| gated GRU | 40.0 / 40.0 / 57.7 | 55.6 / 55.6 / 69.5 | 77.8 / 77.8 / 77.8 |

Local full-protocol temporal rows. These are comparable with each other.
They are blacklist-filtered metrics from `scripts/evaluate_objecreact.py`, not
raw `eval_runner` terminal summaries.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg Success | Avg SPL | Avg Soft SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal EMA | 75.76 | 69.57 | 57.69 | 63.33 | 66.59 | 66.58 | 75.95 |
| plain GRU | 75.76 | 65.22 | 61.54 | 66.67 | 67.30 | 67.30 | 75.73 |
| gated GRU | 72.73 | 47.83 | 61.54 | 63.33 | 61.36 | 61.35 | 72.50 |
| reliability gated GRU | 60.61 | 65.22 | 50.00 | 66.67 | 60.63 | 60.62 | 72.09 |

### Raw, No-Blacklist Temporal Metrics

The table below uses the same completed result directories as the
blacklist-filtered table above, but counts every raw episode directory in the
denominator. In practice this means 36 episodes per task, including episodes
that `configs/defaults.yaml` marks as invalid or problematic. This table is
useful for transparency, but it is not the primary comparison protocol used by
`scripts/evaluate_objecreact.py`.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg Success | Avg SPL | Avg Soft SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal EMA | 69.44 | 55.56 | 52.78 | 63.89 | 60.42 | 60.39 | 70.76 |
| plain GRU | 69.44 | 50.00 | 47.22 | 66.67 | 58.33 | 58.33 | 69.69 |
| gated GRU | 69.44 | 52.78 | 58.33 | 61.11 | 60.42 | 60.41 | 70.88 |
| reliability gated GRU | 55.56 | 55.56 | 47.22 | 66.67 | 56.25 | 56.24 | 67.62 |

Raw success counts:

| Method | Imitate | Alt-Goal | Shortcut | Reverse |
|---|---:|---:|---:|---:|
| temporal EMA | 25/36 | 20/36 | 19/36 | 23/36 |
| plain GRU | 25/36 | 18/36 | 17/36 | 24/36 |
| gated GRU | 25/36 | 19/36 | 21/36 | 22/36 |
| reliability gated GRU | 20/36 | 20/36 | 17/36 | 24/36 |

## Separate Costmap EMA Exploration

These rows are from `out/logs/ema_summary.md` and use the blacklist-filtered
evaluation protocol from `scripts/evaluate_objecreact.py`.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg Success | Avg SPL | Avg Soft SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean / ObjectReact | 72.73 | 65.22 | 46.15 | 73.33 | 64.36 | 64.35 | 75.17 |
| noise | 60.61 | 69.57 | 50.00 | 66.67 | 61.71 | 61.71 | 71.45 |
| costmap EMA | 75.76 | 60.87 | 57.69 | 73.33 | 66.91 | 66.91 | 77.70 |
| costmap EMA + noise | 72.73 | 56.52 | 38.46 | 73.33 | 60.26 | 60.24 | 74.15 |

### Raw, No-Blacklist Costmap Metrics

This table counts all 36 raw episode directories per task. The `noise` and
`costmap EMA + noise` rows still include their intended inference-time noise;
the phrase "no-blacklist" only means no episode filtering.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg Success | Avg SPL | Avg Soft SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean / ObjectReact | 69.44 | 55.56 | 44.44 | 69.44 | 59.72 | 59.70 | 71.38 |
| noise | 58.33 | 55.56 | 44.44 | 66.67 | 56.25 | 56.24 | 66.71 |
| costmap EMA | 72.22 | 63.89 | 50.00 | 69.44 | 63.89 | 63.87 | 74.56 |
| costmap EMA + noise | 66.67 | 50.00 | 38.89 | 72.22 | 56.94 | 56.92 | 70.28 |

## Notes

- In the temporary temporal-grid table, `gated GRU` beats `temporal EMA` on
  Shortcut and Reverse.
- In the local full-protocol temporal rows, `plain GRU` has the best average
  Success and SPL, while `temporal EMA` has the best average Soft SPL.
- `gated GRU` ties `plain GRU` on Shortcut, while `reliability gated GRU` ties
  `plain GRU` on Alt-Goal and Reverse.
- `reliability gated GRU` improves over `gated GRU` on `alt_goal` and `reverse`, but drops on `imitate` and `shortcut`.
- `costmap EMA` has the best average metrics in the separate costmap-smoothing table, but it is not the same baseline as `temporal EMA`.
- The full-protocol plain-GRU row uses `out/results/*/temporal_*_gru/...` complete 36-episode result folders. The one-episode duplicate `temporal_alt_goal_gru/20260530-13-17-11...` run is excluded.
- Without blacklist filtering, the local temporal ranking changes: gated GRU
  has the best Avg SPL and Avg Soft SPL, while temporal EMA ties it on Avg
  Success. Plain GRU drops mainly because its raw Alt-Goal and Shortcut success
  rates are lower.
- The blacklist-filtered table remains the primary table because it matches the
  repository's default evaluation script and removes episodes already marked as
  invalid/problematic.
- I did not find complete local result folders for `mean` aggregators; only config/script references are present.
