# Temporal Costmap Aggregation for ObjectReact

Drop-in extension to [ObjectReact](https://object-react.github.io/) that
replaces the single-frame WayObject Costmap policy with a temporal
aggregator over a sliding window of `K = context_size + 1` recent costmap
encodings. The aggregator is a one-layer GRU mixed with a running EMA
through a cosine-similarity confidence gate; the gate learns *when* to
discount the current frame.

## Code layout

```
temporal_objectreact/
├── temporal_aggregator.py          # EMA / GRU / TemporalCostmapAggregator
├── gnm_temporal.py                 # GNMTemporal: subclass of upstream GNM
├── objectreact_temporal_controller.py  # inference-time controller
├── train_temporal.py               # standalone training entry point
├── eval_runner.py                  # main.py wrapper that supports key=value overrides
├── configs/train/                  # training YAMLs (gru, gated_gru, ema)
└── scripts/                        # bash drivers for train + eval + ablation + demo
```

Wired into the upstream pipeline by:

* `libs/experiments/model_loader.py` — `method == 'learnt_temporal'` branch.
* `libs/experiments/task_setup.py` — normalises `args.method` so downstream
  checks (`args.method == 'learnt'`) keep working.

The new top-level evaluation config is `configs/object_react_temporal.yaml`
and the matching controller config is
`configs/controller/object_react_temporal_controller.yaml`.

## End-to-end usage

1. **Download upstream pretrained weights** (~18 MB):
   ```
   bash temporal_objectreact/scripts/00_download_pretrained.sh
   ```
2. **Train** one variant (writes to `logs/temporal_<variant>/`):
   ```
   bash temporal_objectreact/scripts/10_train_temporal.sh gated_gru
   bash temporal_objectreact/scripts/10_train_temporal.sh gru
   bash temporal_objectreact/scripts/10_train_temporal.sh ema
   ```
   Use the second positional arg to cap iters for a smoke test:
   `bash temporal_objectreact/scripts/10_train_temporal.sh gated_gru 50`.
3. **Evaluate** one (variant, task):
   ```
   LOAD_RUN=logs/temporal_gated_gru/latest.pth AGGREGATOR=gated_gru \
   bash temporal_objectreact/scripts/20_eval_one.sh temporal imitate
   ```
4. **Run all four tasks**:
   ```
   LOAD_RUN=logs/temporal_gated_gru/latest.pth AGGREGATOR=gated_gru \
   bash temporal_objectreact/scripts/21_eval_all_tasks.sh temporal
   ```
   For the baseline number, run `... 21_eval_all_tasks.sh baseline`.
5. **Ablation across aggregators** (mean / ema / gru / gated_gru):
   ```
   bash temporal_objectreact/scripts/30_ablation.sh
   ```
6. **Inferred-perception evaluation**:
   ```
   LOAD_RUN=logs/temporal_gated_gru/latest.pth AGGREGATOR=gated_gru \
   bash temporal_objectreact/scripts/40_perception_noise_eval.sh temporal
   ```
7. **Record demo rollouts** (one episode per task, `save_vis=True`):
   ```
   LOAD_RUN=logs/temporal_gated_gru/latest.pth AGGREGATOR=gated_gru \
   bash temporal_objectreact/scripts/50_demo_record.sh temporal 0
   ```

All scripts forward overrides to `main.py` through `eval_runner.py`, which
accepts `key=value` (and dotted keys, e.g. `controller.load_run=foo.pth`).

## Aggregator zoo

| name (`temporal_aggregator`) | params | comment |
|------------------------------|--------|---------|
| `mean`                       | 0      | upstream behaviour              |
| `ema`                        | 0      | $\lambda$ from `temporal_ema_lambda` |
| `gru`                        | ~6.3 M | one-layer GRU                   |
| `gated_gru` (default)        | +2     | GRU mixed with running EMA via $\sigma(w\cos+b)$ |
| `gru_no_gate`                | ~6.3 M | gated mix disabled              |

## Smoke test

Verifies imports + forward shape (no habitat, no pretrained weights):

```
PYTHONPATH=. python -m temporal_objectreact.gnm_temporal
```

You should see `action: torch.Size([2, 10, 4])`.
