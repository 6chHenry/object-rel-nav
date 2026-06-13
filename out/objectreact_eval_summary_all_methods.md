# ObjectReact Evaluation Summary

Date: 2026-06-12

Primary metrics are blacklist-filtered results from
`scripts/evaluate_objecreact.py`. The full protocol uses HM3D IIN validation
episodes, hard difficulty, `step_idx=3`, `end_idx=108`, and `max_steps=300`.

## Original and Parameter-Free Clean Baselines

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg SR | Avg SPL | Avg Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| ObjectReact, single frame | 72.73 | 65.22 | 46.15 | **73.33** | 64.36 | 64.35 | 75.17 |
| Temporal EMA, K=6 | **75.76** | **69.57** | **57.69** | 63.33 | **66.59** | **66.58** | **75.95** |

Temporal EMA is a parameter-free six-frame embedding baseline with
`lambda=0.7`. These references share the episode and metric protocol but use
their respective historical controller configurations, so the 2.23-point
difference is descriptive rather than a strict architectural ablation. The
older top-level `ema/noise` intervention runs are not used because those flags
were not connected to controller execution at that time.

## Train Noise 0.0, Clean Evaluation

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg SR | Avg SPL | Avg Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain GRU | 75.76 | **65.22** | **61.54** | **66.67** | **67.30** | **67.30** | **75.73** |
| cosine-gated GRU | **78.79** | 52.17 | 57.69 | 60.00 | 62.16 | 62.16 | 70.84 |
| reliability-gated GRU | 45.45 | 56.52 | 7.69 | 36.67 | 36.58 | 36.56 | 50.36 |

## Train Noise 0.0, Inference Noise 0.2

Observed injection rates are 19.44%-20.05%.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg SR | Avg SPL | Avg Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain GRU | **72.73** | **56.52** | 50.00 | 53.33 | **58.14** | **58.14** | **69.44** |
| cosine-gated GRU | 66.67 | 39.13 | **57.69** | **63.33** | 56.70 | 56.70 | 68.43 |
| reliability-gated GRU | 30.30 | 47.83 | 0.00 | 20.00 | 24.53 | 24.52 | 39.94 |

Clean-to-noisy Avg SPL deltas are -9.15, -5.46, and -12.05 respectively.

## Train Noise 0.2, Clean Evaluation

All learned variants use training `noise_p=0.2`.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg SR | Avg SPL | Avg Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain GRU | **75.76** | 52.17 | 57.69 | 56.67 | 60.57 | 60.57 | 72.42 |
| cosine-gated GRU | 72.73 | 47.83 | **61.54** | 63.33 | **61.36** | **61.35** | **72.50** |
| reliability-gated GRU | 60.61 | **65.22** | 50.00 | **66.67** | 60.62 | 60.62 | 72.09 |

The maximum clean Avg SPL difference is only 0.78 points.

## Train Noise 0.2, Inference Noise 0.2

All models use training `noise_p=0.2`. At inference, the current costmap is
zeroed independently with probability 0.2 before entering temporal history.
Seed 42 and the episode name define a shared corruption schedule across
methods. Observed injection rates are 19.41%-20.20%.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg SR | Avg SPL | Avg Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain GRU | **69.70** | 47.83 | **57.69** | 60.00 | 58.80 | 58.80 | 69.39 |
| cosine-gated GRU | 66.67 | 39.13 | 53.85 | 60.00 | 54.91 | 54.91 | 68.74 |
| reliability-gated GRU | 63.64 | **60.87** | 46.15 | **70.00** | **60.16** | **60.16** | **74.02** |

## Robustness Delta

| Method | Delta Avg SR | Delta Avg SPL | Delta Avg Soft-SPL |
|---|---:|---:|---:|
| plain GRU | -1.77 | -1.77 | -3.03 |
| cosine-gated GRU | -6.45 | -6.44 | -3.76 |
| reliability-gated GRU | **-0.46** | **-0.46** | **+1.93** |

## Main Findings

- The completed experiment is a full `2x2` design over training and inference
  noise probabilities `{0.0, 0.2}`.
- Reliability gating depends strongly on training corruption: noisy Avg SPL
  improves from 24.52 to 60.16 when training noise changes from 0.0 to 0.2.
- Plain GRU noisy Avg SPL changes only from 58.14 to 58.80.
- Cosine-gated GRU does not benefit from training corruption: noisy Avg SPL
  changes from 56.70 to 54.91.
- Under matched training noise, the three learned models have nearly tied
  clean averages but different task specializations.
- Reliability-gated GRU is the best learned model under verified 20%
  inference corruption. It leads plain GRU by 1.36 Avg SPL and cosine-gated
  GRU by 5.25 Avg SPL.
- The reliability model retains clean Avg SPL within 0.46 points and improves
  Avg Soft-SPL by 1.93 points.
- Cosine gating is not robust: it loses 6.44 Avg SPL, indicating that scalar
  feature similarity is an inadequate reliability signal.
- The reliability advantage is a single-seed trend, not yet a statistically
  significant result.

## Supervised Gate Fine-Tuning

A three-epoch fine-tune adds explicit per-frame corruption labels to the
noise-trained Reliability-Gated GRU. It changes pooled gate detection from
ROC-AUC/AP `0.312/0.136` to `1.000/1.000`. Mean alpha becomes `0.99951` on
clean frames and `0.000079` on injected frames.

| Task | Clean SPL | Eval Noise 0.2 SPL | Clean Soft-SPL | Noisy Soft-SPL |
|---|---:|---:|---:|---:|
| Imitate | 42.42 | 48.48 | 57.46 | 66.61 |
| Alt-Goal | 47.83 | 52.17 | 65.41 | 66.87 |
| Shortcut | 26.92 | 19.23 | 52.10 | 47.90 |
| Reverse | 30.00 | 33.33 | 45.00 | 51.06 |
| **Average** | **36.79** | **38.31** | **54.99** | **58.11** |

The detector succeeds, but navigation fails: the original noise-trained
Reliability-Gated GRU records 60.62 clean and 60.16 noisy Avg SPL. Therefore
corruption detection accuracy is not sufficient for robust control. This
fine-tune is retained as a negative mechanism experiment, not as an improved
navigation model.

## Data Caveats

- Effective filtered episode counts are 33/23/26/30 for
  Imitate/Alt-Goal/Shortcut/Reverse.
- Shortcut has two known missing-data failures.
- One non-blacklisted Shortcut episode has no `success_status` and remains in
  the denominator for all three methods.
- Supervised fine-tune results use the same blacklist-filtered protocol and
  `best.pth` checkpoint for both clean and noisy evaluation.
- Legacy `noise` and `costmap EMA + noise` tables are invalidated because the
  intervention flags were not connected to controller execution in those runs.
