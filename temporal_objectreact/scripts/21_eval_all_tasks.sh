#!/usr/bin/env bash
# Run all 4 InstanceImageNav tasks for one variant in sequence, then summarise.
#
# Usage:
#   LOAD_RUN=logs/temporal_gated_gru/latest.pth \
#   AGGREGATOR=gated_gru \
#   bash temporal_objectreact/scripts/21_eval_all_tasks.sh temporal
#
#   bash temporal_objectreact/scripts/21_eval_all_tasks.sh baseline
set -euo pipefail

cd "$(dirname "$0")/../.."

VARIANT="${1:?usage: $0 <baseline|temporal>}"

for TASK in imitate alt_goal shortcut reverse; do
  echo
  echo "================ $VARIANT / $TASK ================"
  bash temporal_objectreact/scripts/20_eval_one.sh "$VARIANT" "$TASK"
done

echo
echo "[21_eval_all_tasks] aggregating numbers via evaluate_objecreact.py"
python scripts/evaluate_objecreact.py ./out/results/
