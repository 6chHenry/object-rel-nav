#!/usr/bin/env bash
# Evaluate baseline vs temporal under inferred (= noisier) perception.
# Flips goal_source -> topological and edge_weight_str -> e3d_max, which the
# upstream README documents as the inferred-perception setting.
set -euo pipefail

cd "$(dirname "$0")/../.."

VARIANT="${1:?usage: $0 <baseline|temporal>}"
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
  TAG="${VARIANT}_${TASK}_inferred"

  OVERRIDES=(
    "task_type=$TT" "reverse=$RV"
    "goal_source=topological"
    "goal_gen.edge_weight_str=e3d_max"
    "exp_name=$TAG"
  )
  [[ -n "${LOAD_RUN:-}"   ]] && OVERRIDES+=("controller.load_run=$LOAD_RUN")
  if [[ -n "${AGGREGATOR:-}" && "$VARIANT" == "temporal" ]]; then
    OVERRIDES+=("controller.temporal_aggregator=$AGGREGATOR")
  fi

  echo
  echo "================ inferred perception: $VARIANT / $TASK ================"
  python -m temporal_objectreact.eval_runner -c "$CFG" --set "${OVERRIDES[@]}" \
    2>&1 | tee -a "out/eval_${TAG}.log"
done

python scripts/evaluate_objecreact.py ./out/results/
