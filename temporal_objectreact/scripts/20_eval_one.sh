#!/usr/bin/env bash
# Run main.py end-to-end navigation evaluation for one (method, task) combo.
#
# Usage:
#   bash temporal_objectreact/scripts/20_eval_one.sh <baseline|temporal> <task>
#
# tasks:
#   imitate     -> task_type=original  reverse=False
#   alt_goal    -> task_type=alt_goal  reverse=False
#   shortcut    -> task_type=via_alt_goal reverse=False
#   reverse     -> task_type=original  reverse=True
#
# Env knobs:
#   LOAD_RUN     path to checkpoint, overrides controller.load_run
#   AGGREGATOR   gated_gru / gru / ema / mean  (temporal only)
#   EXP_NAME     suffix appended to exp_name
#   MAX_STEPS    override max_steps
set -euo pipefail

cd "$(dirname "$0")/../.."

VARIANT="${1:?usage: $0 <baseline|temporal> <task>}"
TASK="${2:?usage: $0 <baseline|temporal> <task>}"

case "$VARIANT" in
  baseline) CFG="configs/object_react.yaml" ;;
  temporal) CFG="configs/object_react_temporal.yaml" ;;
  *) echo "unknown variant: $VARIANT (baseline|temporal)" >&2; exit 1 ;;
esac

case "$TASK" in
  imitate)  TASK_TYPE="original";     REVERSE="False" ;;
  alt_goal) TASK_TYPE="alt_goal";     REVERSE="False" ;;
  shortcut) TASK_TYPE="via_alt_goal"; REVERSE="False" ;;
  reverse)  TASK_TYPE="original";     REVERSE="True"  ;;
  *) echo "unknown task: $TASK" >&2; exit 1 ;;
esac

OVERRIDES=("task_type=$TASK_TYPE" "reverse=$REVERSE")
[[ -n "${LOAD_RUN:-}"   ]] && OVERRIDES+=("controller.load_run=$LOAD_RUN")
if [[ -n "${AGGREGATOR:-}" && "$VARIANT" == "temporal" ]]; then
  OVERRIDES+=("controller.temporal_aggregator=$AGGREGATOR")
fi
[[ -n "${MAX_STEPS:-}"  ]] && OVERRIDES+=("max_steps=$MAX_STEPS")
TAG="${VARIANT}_${TASK}${EXP_NAME:+_$EXP_NAME}"
OVERRIDES+=("exp_name=$TAG")

mkdir -p out
echo "[20_eval_one] $VARIANT / $TASK   overrides=${OVERRIDES[*]}"
python -m temporal_objectreact.eval_runner -c "$CFG" --set "${OVERRIDES[@]}" \
  2>&1 | tee -a "out/eval_${TAG}.log"
