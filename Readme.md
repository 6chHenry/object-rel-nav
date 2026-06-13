# Temporal Object-Relative Navigation

Temporal Object-Relative Navigation is a project built on top of
[ObjectReact](https://object-react.github.io/), [TANGO](https://podgorki.github.io/TANGO/),
and [RoboHop](https://oravus.github.io/RoboHop/). The repository extends the
original object-relative navigation pipeline with temporal costmap aggregation,
additional benchmark scripts, and robustness-oriented evaluation utilities for
InstanceImageNav in HM3D scenes.

This project was jointly developed by [@6chHenry](https://github.com/6chHenry),
[@dra-ya](https://github.com/dra-ya), and
[@Zi-hang-Zhou](https://github.com/Zi-hang-Zhou).

## Overview

The original ObjectReact controller predicts navigation actions from the
current object-relative costmap feature. In this repository, we add temporal
reasoning over a short window of recent costmap encodings. The motivation is to
make the controller less sensitive to transient perception failures, noisy
costmap observations, and task variants where single-frame decisions are
ambiguous.

The main additions are:

- A temporal ObjectReact controller with a pluggable costmap-feature aggregator.
- EMA, plain GRU, gated GRU, and reliability-gated GRU aggregation variants.
- Noise augmentation during temporal training to simulate unreliable frames.
- Evaluation wrappers for the four ObjectReact-style tasks: Imitate, Alt-Goal,
  Shortcut, and Reverse.
- Scripts for ablation studies, inferred-perception/noise evaluation, and demo
  rollout recording.
- Summary reports under `out/` for the completed local experiments.

## Repository Layout

```text
.
|-- main.py                                  # Main navigation entry point
|-- configs/
|   |-- object_react.yaml                    # Baseline ObjectReact config
|   |-- object_react_temporal.yaml           # Temporal ObjectReact config
|   |-- tango.yaml                           # TANGO config
|   |-- benchmarks/                          # Task-specific benchmark configs
|   `-- controller/                          # Controller-level configs
|-- libs/                                    # Controllers, mapping, perception, logging
|-- scripts/                                 # Baseline eval/map/benchmark utilities
|-- temporal_objectreact/
|   |-- temporal_aggregator.py               # EMA / GRU / gated temporal modules
|   |-- gnm_temporal.py                      # GNM with temporal costmap aggregation
|   |-- objectreact_temporal_controller.py   # Inference-time temporal controller
|   |-- train_temporal.py                    # Temporal training entry point
|   |-- eval_runner.py                       # Config override wrapper for eval
|   |-- configs/train/                       # Training configs for temporal variants
|   `-- scripts/                             # Training, eval, ablation, demo scripts
`-- out/                                     # Project reports and experiment summaries
```

## Implemented Features

### Temporal Costmap Aggregation

The temporal controller replaces the single-frame goal/costmap embedding path
with a sliding window of `K = context_size + 1` recent encodings. The aggregated
representation is then passed to the original ObjectReact action head.

Supported aggregators:

| Aggregator | Description |
|---|---|
| `mean` | Mean aggregation, matching the upstream multi-frame baseline behavior. |
| `ema` | Parameter-free exponential moving average over temporal costmap features. |
| `gru` | One-layer GRU over the temporal feature sequence. |
| `gated_gru` | GRU preceded by a cosine-confidence gate against a running EMA state. |
| `reliability_gated_gru` | GRU preceded by a learned reliability gate over current/history relations. |
| `gru_no_gate` | GRU path with the confidence gate disabled, useful for ablations. |

### Robustness-Oriented Training

The temporal training path supports input-level noise augmentation. During
training, one frame in the temporal window can be zeroed or perturbed with
Gaussian noise. This simulates temporary perception failure and encourages the
aggregator to use temporal context rather than over-trusting a single frame.

### Evaluation Extensions

The repository includes evaluation drivers for:

- Baseline ObjectReact evaluation.
- Temporal ObjectReact evaluation with checkpoint and aggregator overrides.
- Four benchmark tasks: Imitate, Alt-Goal, Shortcut, and Reverse.
- Aggregator ablations across EMA, GRU, gated GRU, and reliability-gated GRU.
- Inferred-perception/noise robustness experiments.
- Visualization/demo recording with `save_vis=True`.

## Environment Setup

The upstream navigation stack depends on Habitat, PyTorch, OpenCV, and several
vision/modeling libraries. The recommended environment follows the original
ObjectReact setup and uses Python 3.9.

```bash
conda create -n nav
conda activate nav

conda install python=3.9 mamba -c conda-forge

mamba install \
  pip numpy matplotlib pytorch torchvision pytorch-cuda=11.8 \
  opencv=4.6 cmake=3.14.0 habitat-sim withbullet numba=0.57 \
  pyyaml ipykernel networkx h5py natsort open-clip-torch \
  transformers einops scikit-learn kornia pgmpy python-igraph pyvis \
  -c pytorch -c nvidia -c aihabitat -c conda-forge

mamba install -c conda-forge ultralytics
mamba install -c conda-forge tyro faiss-gpu scikit-image ipykernel \
  spatialmath-python gdown utm seaborn wandb kaggle yacs
```

Clone this repository and initialize the ObjectReact submodule:

```bash
git clone https://github.com/6chHenry/object-rel-nav.git
cd object-rel-nav
git submodule update --init --recursive
```

Install Habitat-Lab v0.2.4:

```bash
cd libs
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.4
pip install -e habitat-lab
cd ../..
```

For all commands below, set `PYTHONPATH` so the temporal modules and upstream
ObjectReact training package can be imported:

```bash
export PYTHONPATH="$PWD:$PWD/libs/control/object_react/train${PYTHONPATH:+:$PYTHONPATH}"
```

## Data Preparation

Create a `data/` directory and prepare the following datasets:

- HM3D v0.2 validation scenes.
- Official InstanceImageNav HM3D v3 dataset.
- ObjectReact HM3D evaluation trajectories.
- `maps_via_alt_goal` for the Shortcut task.
- Optional ObjectReact training data if temporal models will be trained.

The expected layout is:

```text
data/
|-- hm3d_v0.2/
|   `-- val/
|-- instance_imagenav_hm3d_v3/
|-- hm3d_iin_val/
`-- hm3d_generated/
    `-- stretch_maps/.../maps_via_alt_goal
```

### Download Evaluation Data

HM3D requires a Matterport account and agreement approval. Download the HM3D
v0.2 validation archives from the official Matterport Habitat dataset page and
extract them into `data/hm3d_v0.2/val/`.

Download the official InstanceImageNav challenge dataset:

```bash
mkdir -p data
cd data
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip
unzip instance_imagenav_hm3d_v3.zip
cd ..
```

Download ObjectReact evaluation trajectories and maps:

```bash
cd data
huggingface-cli download oravus/objectreact_hm3d_iin \
  --repo-type dataset \
  --local-dir ./ \
  --include "evaluation/**"

unzip -q "evaluation/*.zip"
rm -rf evaluation
cd ..
```

This repository also provides helper scripts under
`temporal_objectreact/scripts/download/` for common data preparation steps.

### Optional Training Data

Temporal training uses the upstream ObjectReact training split:

```text
libs/control/object_react/train/vint_train/data/data_splits/training/bigger_bot_0.3-sh_0.4
```

The helper script can download it when the required command-line tools are
available:

```bash
bash temporal_objectreact/scripts/download/04_get_training_data.sh
```

## Model Weights

Create `model_weights/` and download the controller weights:

```bash
mkdir -p model_weights
```

- ObjectReact: downloaded automatically by the provided helper script.
- TANGO: requires Depth Anything metric-depth and base ViT checkpoints.
- PixNav: uses the original authors' checkpoint.

For ObjectReact:

```bash
bash temporal_objectreact/scripts/00_download_pretrained.sh
```

This writes the upstream ObjectReact checkpoint to:

```text
model_weights/object_react_latest.pth
```

## Quick Sanity Check

The temporal model has a lightweight import/forward-shape check that does not
require Habitat scenes or pretrained weights:

```bash
PYTHONPATH=. python -m temporal_objectreact.gnm_temporal
```

Expected output includes an action tensor shape similar to:

```text
action: torch.Size([2, 10, 4])
```

You can also run a formatting check:

```bash
ruff format --check .
```

## Running Navigation

Run the default navigation configuration:

```bash
python main.py
```

Run the baseline ObjectReact controller:

```bash
python main.py -c configs/object_react.yaml
```

Run the temporal ObjectReact controller:

```bash
python main.py -c configs/object_react_temporal.yaml
```

Outputs are written under `out/`, including logs and visualizations when
enabled in the config.

## Training Temporal Models

Download the upstream ObjectReact checkpoint first:

```bash
bash temporal_objectreact/scripts/00_download_pretrained.sh
```

Train one temporal variant:

```bash
bash temporal_objectreact/scripts/10_train_temporal.sh gated_gru
bash temporal_objectreact/scripts/10_train_temporal.sh reliability_gated_gru
bash temporal_objectreact/scripts/10_train_temporal.sh gru
bash temporal_objectreact/scripts/10_train_temporal.sh ema
```

For a short smoke run, pass a maximum-iteration cap as the second argument:

```bash
bash temporal_objectreact/scripts/10_train_temporal.sh gated_gru 50
```

Checkpoints are saved to:

```text
logs/temporal_<variant>/latest.pth
```

The main training configs are:

```text
temporal_objectreact/configs/train/temporal_gated_gru.yaml
temporal_objectreact/configs/train/temporal_reliability_gated_gru.yaml
temporal_objectreact/configs/train/temporal_gru.yaml
temporal_objectreact/configs/train/temporal_ema.yaml
```

## Evaluation and Reproduction

### Evaluate One Task

Evaluate a trained temporal model on one task:

```bash
LOAD_RUN=logs/temporal_gated_gru/latest.pth \
AGGREGATOR=gated_gru \
bash temporal_objectreact/scripts/20_eval_one.sh temporal imitate
```

Evaluate the baseline controller:

```bash
bash temporal_objectreact/scripts/20_eval_one.sh baseline imitate
```

Supported task names:

```text
imitate
alt_goal
shortcut
reverse
```

### Evaluate All Tasks

Run all four tasks for a temporal model:

```bash
LOAD_RUN=logs/temporal_gated_gru/latest.pth \
AGGREGATOR=gated_gru \
bash temporal_objectreact/scripts/21_eval_all_tasks.sh temporal
```

Run all four tasks for the baseline:

```bash
bash temporal_objectreact/scripts/21_eval_all_tasks.sh baseline
```

The script aggregates metrics with:

```bash
python scripts/evaluate_objecreact.py ./out/results/
```

### Aggregator Ablation

Run the temporal aggregation ablation:

```bash
bash temporal_objectreact/scripts/30_ablation.sh
```

The script expects checkpoints at:

```text
logs/temporal_gru/latest.pth
logs/temporal_gated_gru/latest.pth
logs/temporal_reliability_gated_gru/latest.pth
```

The `ema` aggregator is parameter-free and does not require a trained
aggregator checkpoint.

### Inferred-Perception / Noise Evaluation

Evaluate robustness under noisier perception settings:

```bash
LOAD_RUN=logs/temporal_gated_gru/latest.pth \
AGGREGATOR=gated_gru \
bash temporal_objectreact/scripts/40_perception_noise_eval.sh temporal
```

### Demo Rollout Recording

Record one episode per task with visualization enabled:

```bash
LOAD_RUN=logs/temporal_gated_gru/latest.pth \
AGGREGATOR=gated_gru \
bash temporal_objectreact/scripts/50_demo_record.sh temporal 0
```

## Reported Results

The current project reports are stored in:

```text
out/objectreact_project_report.md
out/objectreact_eval_summary_all_methods.md
```

The completed local full-protocol learned-GRU runs include four task settings:

| Method | Imitate | Alt-Goal | Shortcut | Reverse | Avg Success | Avg SPL | Avg Soft SPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| gated GRU | 72.73 | 47.83 | 61.54 | 63.33 | 61.36 | 61.35 | 72.50 |
| reliability gated GRU | 60.61 | 65.22 | 50.00 | 66.67 | 60.63 | 60.62 | 72.09 |

These numbers should be interpreted as local project results rather than final
benchmark claims. The report notes that some temporary EMA/plain-GRU baselines
were collected under a smaller protocol, so a strict final ranking should rerun
all aggregators under the same evaluation configuration.

### Presentation Rollout Videos

Four matched Plain GRU versus Reliability-Gated GRU rollout videos are
available in [`presentation_videos/`](presentation_videos/README.md). They
cover Imitate, Alt-Goal, Shortcut, and Reverse under controlled 20% inference
corruption. The accompanying README documents the selected episodes, panel
annotations, aggregate context, and the limitations of these deliberately
selected qualitative examples.

## Formatting and Code Quality

The repository is formatted with Ruff:

```bash
ruff format .
ruff format --check .
```

The codebase includes upstream and vendored research components. For linting
work, it is usually more useful to separate local project code from vendored
directories such as `libs/depth/depth_anything` and `libs/matcher/LightGlue`.

## Acknowledgements

This project builds on the public ObjectReact, TANGO, RoboHop, Habitat, Depth
Anything, and LightGlue ecosystems. We thank the original authors for releasing
their code, models, and benchmark resources.

## Citation

If you use this repository, please cite the upstream projects it builds on.

ObjectReact:

```bibtex
@inproceedings{garg2025objectreact,
  title={ObjectReact: Learning Object-Relative Control for Visual Navigation},
  author={Garg, Sourav and Craggs, Dustin and Bhat, Vineeth and Mares, Lachlan and Podgorski, Stefan and Krishna, Madhava and Dayoub, Feras and Reid, Ian},
  booktitle={Conference on Robot Learning},
  year={2025},
  organization={PMLR}
}
```

TANGO:

```bibtex
@inproceedings{podgorski2025tango,
  title={TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals},
  author={Podgorski, Stefan and Garg, Sourav and Hosseinzadeh, Mehdi and Mares, Lachlan and Dayoub, Feras and Reid, Ian},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={2399--2406},
  year={2025},
  organization={IEEE}
}
```

RoboHop:

```bibtex
@inproceedings{garg2024robohop,
  title={Robohop: Segment-based topological map representation for open-world visual navigation},
  author={Garg, Sourav and Rana, Krishan and Hosseinzadeh, Mehdi and Mares, Lachlan and S{\"u}nderhauf, Niko and Dayoub, Feras and Reid, Ian},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4090--4097},
  year={2024},
  organization={IEEE}
}
```
