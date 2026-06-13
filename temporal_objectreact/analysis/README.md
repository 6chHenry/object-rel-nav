# Gate and Qualitative Analysis

Run the eight diagnostic evaluations on a GPU node:

```bash
cd /path/to/object-rel-nav
CUDA_VISIBLE_DEVICES=0 \
  bash temporal_objectreact/scripts/run_reliability_gate_diagnostics.sh
```

The experiment compares Reliability-Gated GRU checkpoints trained with
`noise_p=0.0` and `noise_p=0.2`. Both are evaluated on all four tasks with
inference noise probability `0.2` and seed `42`. Set `DRY_RUN=1` to print the
commands without launching Habitat.

After all runs finish, generate the gate-detection statistics:

```bash
portable_envs/nav_env/bin/python \
  temporal_objectreact/analysis/analyze_gate_diagnostics.py
```

Outputs are written to `out/analysis/gate_detection/`, including per-step
scores, pooled/per-task ROC-AUC and average precision with episode-cluster
bootstrap confidence intervals, comparisons between training conditions, and
PNG/PDF figures.

Generate matched qualitative rollouts:

```bash
portable_envs/nav_env/bin/python \
  temporal_objectreact/analysis/make_matched_rollouts.py
```

For each task, the script automatically selects a non-blacklisted episode where
Reliability-Gated GRU succeeds and Plain GRU fails, prioritizing the largest
final-distance improvement. Outputs in `out/analysis/qualitative/` include four
side-by-side videos, a PNG/PDF contact sheet, and the selection manifest.
