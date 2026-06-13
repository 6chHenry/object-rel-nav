# ObjectReact Temporal Aggregation Project Report

Date: 2026-06-12

## Objective

This project tests whether short-term temporal aggregation improves the
robustness of ObjectReact's WayObject Costmap controller. The implemented
aggregators are:

- plain GRU;
- cosine-gated GRU;
- reliability-gated GRU.

The controlled study crosses training noise probabilities `{0.0, 0.2}` with
inference noise probabilities `{0.0, 0.2}` for all three learned GRU variants.

## Implementation Status

The temporal controller is integrated into the upstream ObjectReact pipeline
through `method=learnt_temporal`. Training, checkpoint loading, four-task
evaluation, and deterministic inference-time costmap corruption are complete.

The inference-noise implementation zeros the current WayObject Costmap before
it is appended to temporal history. For a fixed `inference_noise_seed` and
episode name, every controller receives the same Bernoulli corruption schedule.
The completed runs used:

```text
inject_costmap_noise: true
noise_prob: 0.2
inference_noise_seed: 42
```

Measured injection rates across the 24 noisy runs were 19.41%-20.20%.

## Training Setup

All learned variants use:

```text
init_from: ./model_weights/object_react_latest.pth
model_type: gnm_temporal
context_type: temporal
context_size: 5
effective window K: context_size + 1 = 6 frames
epochs: 10
batch_size: 128
lr: 7e-4
optimizer: Adam
seed: 0
training noise_p: 0.0 or 0.2
noise mode: zero one randomly selected frame
```

Checkpoints trained with `noise_p=0.0`:

| Method | Checkpoint |
|---|---|
| plain GRU | `logs/temporal_gru/latest.pth` |
| cosine-gated GRU | `logs/temporal_gated_gru_noise00/latest.pth` |
| reliability-gated GRU | `logs/temporal_reliability_gated_gru_noise00/latest.pth` |

Checkpoints trained with `noise_p=0.2`:

| Method | Checkpoint |
|---|---|
| plain GRU | `logs/temporal_gru_noise02/latest.pth` |
| cosine-gated GRU | `logs/temporal_gated_gru/latest.pth` |
| reliability-gated GRU | `logs/temporal_reliability_gated_gru/latest.pth` |

## Evaluation Protocol

```text
dataset: data/hm3d_iin_val
split: val
difficulty: hard
start_idx: 0
step_idx: 3
end_idx: 108
max_steps: 300
goal_source: gt_topometric
perception: ground truth
metric script: scripts/evaluate_objecreact.py
blacklist source: configs/defaults.yaml
```

The four tasks are Imitate, Alt-Goal, Shortcut, and Reverse. Results below are
blacklist-filtered. Effective denominators are 33, 23, 26, and 30 episodes,
respectively.

## Original and Parameter-Free Baselines

These clean baselines use the same episode range, tasks, blacklist, and metric
script. Original ObjectReact consumes only the current WayObject Costmap.
Temporal EMA applies a normalized exponential average over the same six-frame
window used by the learned temporal models, with `lambda=0.7` and no trainable
temporal parameters. The runs use their respective historical upstream and
temporal controller configurations, so their difference is contextual rather
than a strict one-variable ablation.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg SR | Avg SPL | Avg Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| ObjectReact, single frame | 72.73 | 65.22 | 46.15 | **73.33** | 64.36 | 64.35 | 75.17 |
| Temporal EMA, K=6 | **75.76** | **69.57** | **57.69** | 63.33 | **66.59** | **66.58** | **75.95** |

Temporal EMA records 2.23 points higher clean Avg SPL, but its performance is
task-dependent: it is higher on the first three tasks and 10.0 points lower on
Reverse. Because the controller configurations differ, this is not presented
as a causal EMA improvement.

The older runs named `noise`, `ema`, and `ema_noise` are not used for noisy
baseline claims because those top-level intervention flags were not yet
connected to controller execution. Only the clean ObjectReact result and the
later `learnt_temporal` Temporal EMA result are retained.

## Train Noise 0.0, Clean Evaluation

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg SR | Avg SPL | Avg Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain GRU | 75.76 | **65.22** | **61.54** | **66.67** | **67.30** | **67.30** | **75.73** |
| cosine-gated GRU | **78.79** | 52.17 | 57.69 | 60.00 | 62.16 | 62.16 | 70.84 |
| reliability-gated GRU | 45.45 | 56.52 | 7.69 | 36.67 | 36.58 | 36.56 | 50.36 |

Without corruption augmentation, plain GRU is strongest. The learned
reliability gate fails especially severely on Shortcut.

## Train Noise 0.0, Inference Noise 0.2

The 12 runs completed successfully. Observed injection rates were
19.44%-20.05%.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg SR | Avg SPL | Avg Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain GRU | **72.73** | **56.52** | 50.00 | 53.33 | **58.14** | **58.14** | **69.44** |
| cosine-gated GRU | 66.67 | 39.13 | **57.69** | **63.33** | 56.70 | 56.70 | 68.43 |
| reliability-gated GRU | 30.30 | 47.83 | 0.00 | 20.00 | 24.53 | 24.52 | 39.94 |

| Method | Clean Avg SPL | Noise Avg SPL | Delta SPL | Delta Soft-SPL |
|---|---:|---:|---:|---:|
| plain GRU | 67.30 | **58.14** | -9.15 | -6.29 |
| cosine-gated GRU | 62.16 | 56.70 | **-5.46** | **-2.41** |
| reliability-gated GRU | 36.56 | 24.52 | -12.05 | -10.42 |

Plain GRU has the best absolute noisy performance, while cosine gating has the
smallest clean-to-noisy degradation. Reliability gating remains unusable
without corruption augmentation.

## Train Noise 0.2, Clean Evaluation

All learned models use the same training configuration, including
`noise_p=0.2`.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg SR | Avg SPL | Avg Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain GRU | **75.76** | 52.17 | 57.69 | 56.67 | 60.57 | 60.57 | 72.42 |
| cosine-gated GRU | 72.73 | 47.83 | **61.54** | 63.33 | **61.36** | **61.35** | **72.50** |
| reliability-gated GRU | 60.61 | **65.22** | 50.00 | **66.67** | 60.62 | 60.62 | 72.09 |

The clean averages are effectively tied: the maximum Avg SPL difference is
0.78 points. Task behavior differs substantially, with plain GRU strongest on
Imitate, cosine-gated GRU on Shortcut, and reliability-gated GRU on Alt-Goal
and Reverse.

## Train Noise 0.2, Inference Noise 0.2

At every control step, the current WayObject Costmap is independently replaced
with a zero map with probability 0.2. The per-episode corruption sequence is
identical across methods.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg SR | Avg SPL | Avg Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain GRU | **69.70** | 47.83 | **57.69** | 60.00 | 58.80 | 58.80 | 69.39 |
| cosine-gated GRU | 66.67 | 39.13 | 53.85 | 60.00 | 54.91 | 54.91 | 68.74 |
| reliability-gated GRU | 63.64 | **60.87** | 46.15 | **70.00** | **60.16** | **60.16** | **74.02** |

### Robustness relative to clean evaluation

| Method | Clean Avg SPL | Noise Avg SPL | Delta SPL | Clean Avg Soft-SPL | Noise Avg Soft-SPL | Delta Soft-SPL |
|---|---:|---:|---:|---:|---:|---:|
| plain GRU | 60.57 | 58.80 | -1.77 | 72.42 | 69.39 | -3.03 |
| cosine-gated GRU | 61.35 | 54.91 | -6.44 | 72.50 | 68.74 | -3.76 |
| reliability-gated GRU | 60.62 | **60.16** | **-0.46** | 72.09 | **74.02** | **+1.93** |

The reliability-gated GRU is the strongest learned method under controlled
corruption. Its Avg SPL is 1.36 points above plain GRU and 5.25 points above
the cosine-gated GRU. More importantly, it retains its clean Avg SPL within
0.46 points, whereas the other models lose 1.77 and 6.44 points.

The task-level pattern is informative. Reliability gating is strongest on
Alt-Goal and Reverse, while plain GRU remains better on Imitate and Shortcut.
The simple cosine gate is consistently weak, suggesting that scalar embedding
similarity confuses legitimate visual change with corruption. The richer
reliability scorer can use per-channel current/history relations to make a more
useful trust decision.

## Complete 2x2 Summary

Avg SPL across the four tasks:

| Method | Train 0.0 / Eval 0.0 | Train 0.0 / Eval 0.2 | Train 0.2 / Eval 0.0 | Train 0.2 / Eval 0.2 |
|---|---:|---:|---:|---:|
| plain GRU | **67.30** | 58.14 | 60.57 | 58.80 |
| cosine-gated GRU | 62.16 | 56.70 | **61.35** | 54.91 |
| reliability-gated GRU | 36.56 | 24.52 | 60.62 | **60.16** |

Training corruption has a strong architecture-specific effect:

- reliability-gated GRU gains 35.64 noisy Avg SPL from training corruption;
- plain GRU gains only 0.66 noisy Avg SPL;
- cosine-gated GRU loses 1.79 noisy Avg SPL.

## Interpretation

The experiments support three claims:

1. Corruption augmentation is essential for the learned reliability gate, but
   is not a universal improvement for temporal models.
2. Under matched train/eval noise 0.2, reliability gating is the most robust
   learned variant and retains clean Avg SPL within 0.46 points.
3. Plain GRU is the strongest clean model without augmentation and remains
   comparatively insensitive to whether training corruption is used.

The 1.36-point Avg SPL advantage over plain GRU should be reported as a
promising trend rather than a statistically established margin because all
learned models use one training seed and the evaluation set is modest.

## Supervised Gate Fine-Tuning: Negative Result

To make the reliability score explicitly detect corrupted frames, the
noise-trained reliability checkpoint was fine-tuned for three epochs with
independent per-frame zero corruption and a weighted binary
cross-entropy gate loss. The fine-tune used:

```text
init_from: logs/temporal_reliability_gated_gru/latest.pth
epochs: 3
lr: 1e-4
gate_corruption_prob: 0.2
gate_supervision_weight: 0.5
gate_pos_weight: 4.0
gate_history_update: gated
checkpoint: logs/temporal_reliability_gated_gru_supervised_finetune/best.pth
```

The frame detector became essentially perfect:

| Gate metric | Original noise-trained model | Supervised fine-tune |
|---|---:|---:|
| Pooled ROC-AUC | 0.312 | **1.000** |
| Pooled average precision | 0.136 | **1.000** |
| Mean alpha, clean frame | 0.98803 | **0.99951** |
| Mean alpha, injected frame | 0.99840 | **0.000079** |

However, navigation performance collapsed:

| Task | Clean SPL | Inference-noise 0.2 SPL | Clean Soft-SPL | Noisy Soft-SPL |
|---|---:|---:|---:|---:|
| Imitate | 42.42 | 48.48 | 57.46 | 66.61 |
| Alt-Goal | 47.83 | 52.17 | 65.41 | 66.87 |
| Shortcut | 26.92 | 19.23 | 52.10 | 47.90 |
| Reverse | 30.00 | 33.33 | 45.00 | 51.06 |
| **Average** | **36.79** | **38.31** | **54.99** | **58.11** |

The original noise-trained reliability model reaches 60.62 clean and 60.16
noisy Avg SPL. Supervised fine-tuning therefore loses 23.83 clean points and
21.86 noisy points. Since the clean evaluation also collapses, the failure
cannot be explained only as overreacting to injected frames.

This experiment establishes an important negative result: correctly detecting
synthetic corruption is not sufficient for good navigation. The supervised
objective and gated-history intervention change the representation consumed by
the recurrent controller. A future design should decouple detection from
control, freeze or distill the original navigation policy, and use a softer
intervention than replacing features solely according to a near-binary gate.

## Invalidated Legacy Results

Earlier tables labeled `noise`, `costmap EMA`, and `costmap EMA + noise` are
not used in the current conclusions. At the time those runs were produced,
the top-level intervention flags were saved in YAML but were not consumed by
the inference controller. They therefore did not constitute an auditable
costmap intervention. The June 9 controlled GRU runs above are the first
verified inference-noise results; every logged control step records whether
corruption was applied.

This invalidation does not apply to the later parameter-free
`temporal_aggregator=ema` clean evaluation reported above, which runs through
the temporal controller with a six-frame embedding window.

## Result Directories

Train 0.2 / inference 0.2 timestamps:

| Method | Imitate | Alt-Goal | Shortcut | Reverse |
|---|---|---|---|---|
| plain GRU | `20260609-01-40-03` | `20260609-02-09-41` | `20260609-02-43-46` | `20260609-03-23-05` |
| cosine-gated GRU | `20260609-03-54-01` | `20260609-05-38-55` | `20260609-06-20-31` | `20260609-07-01-41` |
| reliability-gated GRU | `20260609-07-38-51` | `20260609-08-15-04` | `20260609-08-55-30` | `20260609-09-40-55` |

The incomplete cosine-gated Alt-Goal directory
`20260609-04-23-03_learnt_temporal_gt_topometric` is excluded.

Train 0.0 / inference 0.2 timestamps:

| Method | Imitate | Alt-Goal | Shortcut | Reverse |
|---|---|---|---|---|
| plain GRU | `20260611-07-40-22` | `20260611-08-16-20` | `20260611-08-55-51` | `20260611-09-39-42` |
| cosine-gated GRU | `20260611-10-17-48` | `20260611-10-54-59` | `20260611-11-35-36` | `20260611-12-16-50` |
| reliability-gated GRU | `20260611-12-53-42` | `20260611-13-41-57` | `20260611-14-26-58` | `20260611-15-17-53` |

Supervised reliability fine-tune timestamps:

| Eval noise | Imitate | Alt-Goal | Shortcut | Reverse |
|---|---|---|---|---|
| 0.0 | `20260613-10-41-08` | `20260613-11-24-02` | `20260613-12-06-32` | `20260613-12-55-08` |
| 0.2 | `20260613-05-56-27` | `20260613-06-38-24` | `20260613-07-21-33` | `20260613-08-08-07` |

## Reliability Checks and Limitations

- All 24 selected noisy runs loaded the intended checkpoint and contain 36
  episode directories: 12 trained with noise 0.0 and 12 trained with noise
  0.2.
- Shortcut has two known dataset failures, leaving 34 `results_dict.npz`
  files. These episodes are covered by the existing evaluation exclusions.
- One non-blacklisted Shortcut episode has no `success_status`; the evaluator
  retains it in the denominator for all three methods.
- Measured corruption rates are approximately 20% in every run.
- All learned models use training seed 0.
- The supervised fine-tune proves that gate detection and navigation quality
  can move in opposite directions: ROC-AUC/AP reach 1.0 while clean/noisy Avg
  SPL fall to 36.79/38.31.
- The evaluation still uses ground-truth perception plus synthetic zero-map
  corruption. FastSAM/LightGlue evaluation remains future work.

## Reproduction

```bash
cd /path/to/object-rel-nav
export CUDA_VISIBLE_DEVICES=0
export NOISE_PROB=0.2
export NOISE_SEED=42

bash temporal_objectreact/scripts/run_noise00_grus_eval_noise02.sh \
  2>&1 | tee out/noise00_train_eval_noise02_full.log
```
