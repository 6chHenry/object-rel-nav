#!/usr/bin/env bash
# Run baseline + 3 temporal variants × 4 tasks sequentially.
#
# Env knobs (with sane defaults):
#   STEP_IDX     : episode subsampling for the 0..108 range (default 10)
#   END_IDX      : last episode idx (default 108)
#   MAX_STEPS    : episode step budget (default 300)
#   PYTHON       : python interpreter (default nav_eval's python)
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHON="${PYTHON:-/home/zihang/miniconda3/envs/nav_eval/bin/python}"
STEP_IDX="${STEP_IDX:-10}"
END_IDX="${END_IDX:-108}"
MAX_STEPS="${MAX_STEPS:-300}"

mkdir -p out

run_one () {
  local VARIANT="$1" TASK="$2"
  local TAG_SUFFIX="$3"          # checkpoint tag (gated_gru / gru / ema / baseline)
  local LOAD_RUN="$4" AGG="$5"

  local TAG="${TAG_SUFFIX}_${TASK}"
  echo
  echo "================ $TAG (step_idx=$STEP_IDX, end_idx=$END_IDX, max_steps=$MAX_STEPS) ================"

  local CFG
  case "$VARIANT" in
    baseline) CFG="configs/object_react.yaml" ;;
    temporal) CFG="configs/object_react_temporal.yaml" ;;
  esac

  local TASK_TYPE REVERSE
  case "$TASK" in
    imitate)  TASK_TYPE="original";     REVERSE="False" ;;
    alt_goal) TASK_TYPE="alt_goal";     REVERSE="False" ;;
    shortcut) TASK_TYPE="via_alt_goal"; REVERSE="False" ;;
    reverse)  TASK_TYPE="original";     REVERSE="True"  ;;
  esac

  local OVERRIDES=(
    "task_type=$TASK_TYPE" "reverse=$REVERSE"
    "max_steps=$MAX_STEPS" "step_idx=$STEP_IDX" "end_idx=$END_IDX"
    "exp_name=$TAG"
  )
  if [[ "$VARIANT" == "temporal" ]]; then
    OVERRIDES+=( "controller.load_run=$LOAD_RUN" "controller.temporal_aggregator=$AGG" )
  fi

  "$PYTHON" -m temporal_objectreact.eval_runner -c "$CFG" --set "${OVERRIDES[@]}" \
    2>&1 | tee -a "out/eval_${TAG}.log" || echo "[30_eval_grid] $TAG returned non-zero, continuing"
}

TASKS=(imitate alt_goal shortcut reverse)

for TASK in "${TASKS[@]}"; do
  run_one baseline "$TASK" baseline "" ""
done
for TASK in "${TASKS[@]}"; do
  run_one temporal "$TASK" gated_gru "logs/temporal_gated_gru/latest.pth" gated_gru
done
for TASK in "${TASKS[@]}"; do
  run_one temporal "$TASK" gru "logs/temporal_gru/latest.pth" gru
done
for TASK in "${TASKS[@]}"; do
  run_one temporal "$TASK" ema "logs/temporal_ema/latest.pth" ema
done

echo
echo "[30_eval_grid] all 16 runs done; aggregating with evaluate_objecreact.py"
"$PYTHON" scripts/evaluate_objecreact.py ./out/results/ 2>&1 | tee out/grid_summary.log
echo "[30_eval_grid] DONE"
