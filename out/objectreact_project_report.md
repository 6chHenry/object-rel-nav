# ObjectReact Temporal Aggregation Project Report

Date: 2026-05-30

## Objective

This project evaluates whether temporal aggregation over ObjectReact costmap
features improves InstanceImageNav performance. The primary comparison is among
methods that aggregate a short temporal window of adjacent costmap features:
temporal EMA, plain GRU, gated GRU, and reliability gated GRU.

In addition, we keep the existing costmap EMA / noise experiments as a separate
engineering exploration. Costmap EMA is useful evidence for inference-time
smoothing, but it is not the same intervention as temporal EMA and should not
be used as the direct EMA baseline when comparing against GRU variants.

The main question is whether temporal modeling makes the controller more robust
across four evaluation conditions:

- Imitate / original goal following
- Alt-Goal
- Shortcut / via-alt-goal
- Reverse

## Methods

| Method | Description | Status |
|---|---|---|
| ObjectReact / single frame | Original ObjectReact behavior using the current frame only | Completed in colleague temporal-grid report |
| temporal EMA | Parameter-free EMA over adjacent temporal costmap features | Completed under the local full protocol |
| plain GRU | Learned temporal GRU aggregator without confidence gate | Completed under the local full protocol |
| gated GRU | GRU with cosine-confidence gate against a running EMA state | Completed |
| reliability gated GRU | GRU with a learned reliability gate | Completed |

The following methods are reported separately because they operate at a
different level of the pipeline:

| Method | Description | Status |
|---|---|---|
| clean / ObjectReact | Original ObjectReact evaluation without injected perturbation | Completed |
| noise | ObjectReact with costmap noise injection | Completed |
| costmap EMA | ObjectReact with exponential moving average smoothing over costmaps | Completed |
| costmap EMA + noise | Costmap EMA smoothing under injected costmap noise | Completed |

## Training Setup

The learned temporal models use the same ObjectReact backbone initialization:

```text
init_from: ./model_weights/object_react_latest.pth
model_type: gnm_temporal
context_type: temporal
context_size: 5
goal_type: image_mask_enc
obs_type: disabled
epochs: 10
batch_size: 128
lr: 7e-4
optimizer: adam
seed: 0
```

Training data:

```text
libs/control/object_react/train/vint_train/data/data_splits/training/bigger_bot_0.3-sh_0.4
```

Temporal model checkpoints:

| Model | Checkpoint | Training noise |
|---|---|---:|
| gated GRU | `logs/temporal_gated_gru/latest.pth` | `noise_p=0.2` |
| reliability gated GRU | `logs/temporal_reliability_gated_gru/latest.pth` | `noise_p=0.2` |
| plain GRU | `logs/temporal_gru/latest.pth` | `noise_p=0.0` |
| temporal EMA | no checkpoint required | parameter-free aggregator |

The plain GRU config exists at
`temporal_objectreact/configs/train/temporal_gru.yaml`, and the local checkpoint
exists at `logs/temporal_gru/latest.pth`.

Temporal EMA is parameter-free. It can be evaluated directly with
`controller.temporal_aggregator=ema` and no `controller.load_run`; the upstream
ObjectReact encoder/head weights are reused.

## Evaluation Setup

There are currently two evaluation protocols in the report.

The local full evaluation protocol used for temporal EMA, plain GRU, gated GRU,
and reliability gated GRU is:

```text
dataset: data/hm3d_iin_val
split: val
difficulty: hard
start_idx: 0
step_idx: 3
end_idx: 108
max_steps: 300
goal_source: gt_topometric
method: learnt_temporal for GRU variants
metric script: scripts/evaluate_objecreact.py
blacklist source: configs/defaults.yaml
```

The small-protocol temporal-grid numbers are taken from `../temporal.md`.
That report uses:

```text
step_idx: 10
end_idx: 108
max_steps: 300
goal_source: gt_topometric
perception: GT
tasks reported: Imitate, Shortcut, Reverse
Alt-Goal: not reported
episodes per task: about 9-11
```

Because the temporary temporal-grid numbers use a smaller episode subset and do
not include Alt-Goal, they should be treated as early small-protocol evidence.
Temporal EMA, plain GRU, gated GRU, and reliability gated GRU have now all been
evaluated under the local full protocol.

The four task settings are:

| Task | `task_type` | `reverse` |
|---|---|---|
| Imitate | `original` | `False` |
| Alt-Goal | `alt_goal` | `False` |
| Shortcut | `via_alt_goal` | `False` |
| Reverse | `original` | `True` |

For GRU evaluation, the commands use `configs/object_react_temporal.yaml` and
override `controller.load_run` plus `controller.temporal_aggregator`.

## Main Temporal Aggregation Results

The table below is the temporary apples-to-apples temporal aggregation table
from `../temporal.md`. These values compare methods that all operate over a
temporal costmap-feature window. Alt-Goal is omitted because it was not reported
in that run.

Values are `Success / SPL / Soft SPL` percentages.

| Method | Imitate | Shortcut | Reverse |
|---|---:|---:|---:|
| ObjectReact / single frame | 40.0 / 40.0 / 56.6 | 44.4 / 44.4 / 56.8 | 44.4 / 44.4 / 50.0 |
| temporal EMA | 40.0 / 40.0 / 56.6 | 33.3 / 33.3 / 55.3 | 44.4 / 44.4 / 49.6 |
| plain GRU | 70.0 / 70.0 / 72.5 | 11.1 / 11.1 / 54.5 | 55.6 / 55.5 / 67.5 |
| gated GRU | 40.0 / 40.0 / 57.7 | 55.6 / 55.6 / 69.5 | 77.8 / 77.8 / 77.8 |

Under this temporary temporal-grid protocol, gated GRU is stronger than temporal
EMA on the harder Shortcut and Reverse conditions. The key observation is that
fixed temporal smoothing does not reliably help, while a learned gate can
selectively suppress frames that disagree with recent history. Plain GRU also
shows that learnable temporal memory alone is not sufficient: it improves
Imitate but fails badly on Shortcut.

## Local Full-Protocol Temporal Results

The following local runs use the fuller `step_idx=3` protocol and include
Alt-Goal. These rows are comparable with each other.

All values in this table are blacklist-filtered metrics from
`scripts/evaluate_objecreact.py`, not the raw `eval_runner` terminal summaries.
For example, plain GRU Alt-Goal has raw success `18/36 = 50.00%`, but after
removing the configured 13 blacklisted episodes it is reported as
`15/23 = 65.22%`.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg Success | Avg SPL | Avg Soft SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal EMA | 75.76 | 69.57 | 57.69 | 63.33 | 66.59 | 66.58 | 75.95 |
| plain GRU | 75.76 | 65.22 | 61.54 | 66.67 | 67.30 | 67.30 | 75.73 |
| gated GRU | 72.73 | 47.83 | 61.54 | 63.33 | 61.36 | 61.35 | 72.50 |
| reliability gated GRU | 60.61 | 65.22 | 50.00 | 66.67 | 60.63 | 60.62 | 72.09 |

Under the current full protocol, plain GRU has the strongest average Success
and SPL among the completed temporal methods, while temporal EMA has the best
average Soft SPL. Gated GRU ties plain GRU on Shortcut but drops on Alt-Goal.
Reliability gated GRU ties plain GRU on Alt-Goal and Reverse, but loses on
Imitate and Shortcut.

## Separate Costmap EMA Exploration

The table below keeps the existing costmap EMA experiments, but reports them as
a separate exploration. These results are useful because they show that
inference-time costmap smoothing can improve robustness, but they should not be
used as the EMA baseline for the temporal GRU ablation.

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg Success | Avg SPL | Avg Soft SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean / ObjectReact | 72.73 | 65.22 | 46.15 | 73.33 | 64.36 | 64.35 | 75.17 |
| noise | 60.61 | 69.57 | 50.00 | 66.67 | 61.71 | 61.71 | 71.45 |
| costmap EMA | 75.76 | 60.87 | 57.69 | 73.33 | 66.91 | 66.91 | 77.70 |
| costmap EMA + noise | 72.73 | 56.52 | 38.46 | 73.33 | 60.26 | 60.24 | 74.15 |

Costmap EMA is a reasonable follow-up direction because it is simple,
training-free, and has the best average metrics in this separate result set.
The correct framing is that it is an additional pipeline-level smoothing
strategy, not a replacement for the temporal EMA baseline in the GRU ablation.

## Reliability Checks

The completed GRU results were checked against the following conditions:

- `logs/temporal_gated_gru/latest.pth` and
  `logs/temporal_reliability_gated_gru/latest.pth` both exist.
- Evaluation logs contain explicit checkpoint loading messages for the expected
  checkpoint paths.
- The gated-GRU checkpoint contains `aggregator.gate.w`,
  `aggregator.gate.b`, and GRU parameters.
- The reliability-GRU checkpoint contains `aggregator.gate.scorer.*` and GRU
  parameters.
- The selected evaluation directories each contain 36 episode folders and
  `results_summary.csv`.
- Re-running `scripts/evaluate_objecreact.py` on the selected result folders
  reproduces the table values.
- Incomplete duplicate runs were excluded from the table.
  This includes the one-episode duplicate directory
  `out/results/alt_goal/temporal_alt_goal_gru/val/hard/20260530-13-17-11_learnt_temporal_gt_topometric`.

One caveat remains: in the Shortcut task, one non-blacklisted episode has
metadata but no `success_status`. The existing evaluation script keeps it in
the denominator as `num_no_status=1/26`. This affects both gated GRU and
reliability gated GRU in the same task, so the comparison is still consistent,
but the report should mention this when presenting final numbers.

## Reproduction Commands

If retraining plain GRU is needed, use:

```bash
cd /inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/yushun/RAG_exploration/ObjectReact/object-rel-nav
source portable_envs/nav_env/bin/activate
export PYTHONPATH="$PWD:$PWD/libs/control/object_react/train"
export CUDA_VISIBLE_DEVICES=0

portable_envs/nav_env/bin/python -m temporal_objectreact.train_temporal \
  --config temporal_objectreact/configs/train/temporal_gru.yaml
```

Then evaluate:

```bash
export EGL_PLATFORM=surfaceless
export PYTHON="$PWD/portable_envs/nav_env/bin/python"
export LOAD_RUN="logs/temporal_gru/latest.pth"
export AGGREGATOR="gru"
export EXP_NAME="gru"

bash temporal_objectreact/scripts/21_eval_all_tasks.sh temporal
```

After evaluation finishes, regenerate the summary with:

```bash
portable_envs/nav_env/bin/python scripts/evaluate_objecreact.py ./out/results/
```
