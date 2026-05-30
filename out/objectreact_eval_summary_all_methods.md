# ObjectReact Evaluation Summary

Date: 2026-05-30

This summary separates two different uses of EMA:

- `temporal EMA` is a temporal costmap-feature aggregator, comparable with
  GRU-style temporal aggregation.
- `costmap EMA` is a pipeline-level smoothing / noise-robustness trick. It is
  useful, but it should not be directly compared against GRU as the EMA
  baseline.

The temporary temporal EMA and plain GRU numbers below are from
`../temporal.md`; they use a smaller `step_idx=10` protocol and do not include
Alt-Goal. The local gated-GRU and reliability-GRU rows use the fuller
`step_idx=3` protocol from `scripts/evaluate_objecreact.py`.

## Temporal Aggregation Ablation

Temporary same-protocol table from `../temporal.md`. Values are
`Success / SPL / Soft SPL` percentages.

| Method | Imitate | Shortcut | Reverse |
|---|---:|---:|---:|
| ObjectReact / single frame | 40.0 / 40.0 / 56.6 | 44.4 / 44.4 / 56.8 | 44.4 / 44.4 / 50.0 |
| temporal EMA | 40.0 / 40.0 / 56.6 | 33.3 / 33.3 / 55.3 | 44.4 / 44.4 / 49.6 |
| plain GRU | 70.0 / 70.0 / 72.5 | 11.1 / 11.1 / 54.5 | 55.6 / 55.5 / 67.5 |
| gated GRU | 40.0 / 40.0 / 57.7 | 55.6 / 55.6 / 69.5 | 77.8 / 77.8 / 77.8 |

Local full-protocol learned-GRU rows. These should be compared with a rerun
temporal EMA/plain GRU before final ranking.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg Success | Avg SPL | Avg Soft SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| gated GRU | 72.73 | 47.83 | 61.54 | 63.33 | 61.36 | 61.35 | 72.50 |
| reliability gated GRU | 60.61 | 65.22 | 50.00 | 66.67 | 60.63 | 60.62 | 72.09 |

## Separate Costmap EMA Exploration

These rows are from `out/logs/ema_summary.md` and use the blacklist-filtered
evaluation protocol from `scripts/evaluate_objecreact.py`.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg Success | Avg SPL | Avg Soft SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean / ObjectReact | 72.73 | 65.22 | 46.15 | 73.33 | 64.36 | 64.35 | 75.17 |
| noise | 60.61 | 69.57 | 50.00 | 66.67 | 61.71 | 61.71 | 71.45 |
| costmap EMA | 75.76 | 60.87 | 57.69 | 73.33 | 66.91 | 66.91 | 77.70 |
| costmap EMA + noise | 72.73 | 56.52 | 38.46 | 73.33 | 60.26 | 60.24 | 74.15 |

## Notes

- In the temporary temporal-grid table, `gated GRU` beats `temporal EMA` on
  Shortcut and Reverse.
- In the local full-protocol learned-GRU rows, `gated GRU` is strongest on
  Shortcut, while `reliability gated GRU` is stronger on Alt-Goal and Reverse.
- `reliability gated GRU` improves over `gated GRU` on `alt_goal` and `reverse`, but drops on `imitate` and `shortcut`.
- `costmap EMA` has the best average metrics in the separate costmap-smoothing table, but it is not the same baseline as `temporal EMA`.
- I did not find `logs/temporal_gru/latest.pth` or complete local result folders for plain `gru`; the plain-GRU numbers shown here are copied from `../temporal.md`.
- I did not find complete local result folders for `mean` aggregators; only config/script references are present.
