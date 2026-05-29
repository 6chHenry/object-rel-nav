#!/usr/bin/env bash
# Run the temporal-aggregator ablation: mean (= upstream), ema, gru, gated_gru.
# For each aggregator we expect a matching checkpoint at logs/temporal_<name>/latest.pth
# (mean uses the upstream ObjectReact checkpoint as-is).
#
# Outputs land in out/results/<exp_name>/... and are summarised by evaluate_objecreact.py.
set -euo pipefail

cd "$(dirname "$0")/../.."

AGGREGATORS=("mean" "ema" "gru" "gated_gru")

for AGG in "${AGGREGATORS[@]}"; do
  if [[ "$AGG" == "mean" ]]; then
    CKPT="./model_weights/object_react_latest.pth"
  else
    CKPT="./logs/temporal_${AGG}/latest.pth"
  fi
  if [[ ! -f "$CKPT" ]]; then
    echo "[30_ablation] skipping $AGG: missing checkpoint $CKPT"
    continue
  fi
  echo
  echo "================ ablation: $AGG ($CKPT) ================"
  LOAD_RUN="$CKPT" AGGREGATOR="$AGG" EXP_NAME="abl_${AGG}" \
    bash temporal_objectreact/scripts/21_eval_all_tasks.sh temporal
done

echo
echo "[30_ablation] all aggregators done; summary:"
python scripts/evaluate_objecreact.py ./out/results/
