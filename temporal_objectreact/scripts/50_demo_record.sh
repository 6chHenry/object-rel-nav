#!/usr/bin/env bash
# Rerun one episode of each task with save_vis=True for the demo video.
set -euo pipefail

cd "$(dirname "$0")/../.."

VARIANT="${1:?usage: $0 <baseline|temporal> [episode_idx]}"
EPISODE_IDX="${2:-0}"

case "$VARIANT" in
  baseline) CFG="configs/object_react.yaml" ;;
  temporal) CFG="configs/object_react_temporal.yaml" ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

mkdir -p out
for TASK in imitate alt_goal shortcut reverse; do
  case "$TASK" in
    imitate)  TT="original";     RV="False" ;;
    alt_goal) TT="alt_goal";     RV="False" ;;
    shortcut) TT="via_alt_goal"; RV="False" ;;
    reverse)  TT="original";     RV="True"  ;;
  esac
  TAG="demo_${VARIANT}_${TASK}"

  OVERRIDES=(
    "task_type=$TT" "reverse=$RV"
    "save_vis=true" "plot=false"
    "start_idx=$EPISODE_IDX" "step_idx=1" "end_idx=$((EPISODE_IDX + 1))"
    "exp_name=$TAG"
  )
  [[ -n "${LOAD_RUN:-}"   ]] && OVERRIDES+=("controller.load_run=$LOAD_RUN")
  if [[ -n "${AGGREGATOR:-}" && "$VARIANT" == "temporal" ]]; then
    OVERRIDES+=("controller.temporal_aggregator=$AGGREGATOR")
  fi

  python -m temporal_objectreact.eval_runner -c "$CFG" --set "${OVERRIDES[@]}" \
    2>&1 | tee "out/${TAG}.log"
done
echo "[50_demo_record] visualisations in out/results/${VARIANT}_<task>/..."
