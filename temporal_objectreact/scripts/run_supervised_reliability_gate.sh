#!/usr/bin/env bash
# Train or evaluate the reliability gate with explicit corruption supervision.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash temporal_objectreact/scripts/run_supervised_reliability_gate.sh finetune
#   CUDA_VISIBLE_DEVICES=0 bash temporal_objectreact/scripts/run_supervised_reliability_gate.sh eval-finetune
#   bash temporal_objectreact/scripts/run_supervised_reliability_gate.sh analyze-finetune
#   CUDA_VISIBLE_DEVICES=0 bash temporal_objectreact/scripts/run_supervised_reliability_gate.sh full
#   CUDA_VISIBLE_DEVICES=0 bash temporal_objectreact/scripts/run_supervised_reliability_gate.sh eval-full
#   bash temporal_objectreact/scripts/run_supervised_reliability_gate.sh analyze-full
#
# Training knobs:
#   OUTPUT=/custom/output/path
#   TRAIN_MAX_ITERS=2
#
# Eval knobs:
#   NOISE_PROBS="0.0 0.2"  default; use "0.2" for gate diagnostics only
#   NOISE_SEED=42
#   MAX_STEPS=300
#   DRY_RUN=1
set -euo pipefail

cd "$(dirname "$0")/../.."

PHASE="${1:?usage: $0 <finetune|eval-finetune|analyze-finetune|full|eval-full|analyze-full>}"
PY="${PYTHON:-$PWD/portable_envs/nav_env/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/libs/control/object_react/train${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/objectreact-mpl-${USER:-root}}"
mkdir -p "$MPLCONFIGDIR" out

LIBS=(
  "$PWD/portable_envs/nav_env/lib"
  "/usr/lib/x86_64-linux-gnu"
  "/lib/x86_64-linux-gnu"
  "/usr/local/nvidia/lib"
  "/usr/local/nvidia/lib64"
)
export LD_LIBRARY_PATH="$(IFS=:; echo "${LIBS[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

train_stage() {
  local stage="$1"
  local config="temporal_objectreact/configs/train/temporal_reliability_gated_gru_supervised.yaml"
  local output="logs/temporal_reliability_gated_gru_supervised"
  if [[ "$stage" == "finetune" ]]; then
    config="temporal_objectreact/configs/train/temporal_reliability_gated_gru_supervised_finetune.yaml"
    output="logs/temporal_reliability_gated_gru_supervised_finetune"
  fi
  output="${OUTPUT:-$output}"
  mkdir -p "$output"
  local extra=()
  if [[ -n "${TRAIN_MAX_ITERS:-}" ]]; then
    extra+=(--max_iters "$TRAIN_MAX_ITERS")
  fi

  echo "[supervised-gate] training stage=$stage config=$config output=$output"
  "$PY" -m temporal_objectreact.train_temporal \
    --config "$config" \
    --out "$output" \
    --device "$DEVICE" \
    "${extra[@]}" \
    2>&1 | tee "$output/train.log"
}

task_overrides() {
  case "$1" in
    imitate) echo "task_type=original reverse=False" ;;
    alt_goal) echo "task_type=alt_goal reverse=False" ;;
    shortcut) echo "task_type=via_alt_goal reverse=False" ;;
    reverse) echo "task_type=original reverse=True" ;;
    *) echo "unknown task: $1" >&2; exit 1 ;;
  esac
}

noise_tag() {
  case "$1" in
    0|0.0) echo "00" ;;
    0.2) echo "02" ;;
    *) echo "$1" | tr -d '.' ;;
  esac
}

eval_stage() {
  local stage="$1"
  local train_tag="supervised"
  local checkpoint="logs/temporal_reliability_gated_gru_supervised/best.pth"
  if [[ "$stage" == "finetune" ]]; then
    train_tag="supervised_finetune"
    checkpoint="logs/temporal_reliability_gated_gru_supervised_finetune/best.pth"
  fi
  checkpoint="${CHECKPOINT:-$checkpoint}"
  if [[ ! -s "$checkpoint" ]]; then
    echo "ERROR: missing checkpoint: $checkpoint" >&2
    exit 1
  fi

  local tasks=(imitate alt_goal shortcut reverse)
  for noise_prob in ${NOISE_PROBS:-0.0 0.2}; do
    local eval_noise_tag
    eval_noise_tag="$(noise_tag "$noise_prob")"
    for task in "${tasks[@]}"; do
      read -r task_type_override reverse_override <<<"$(task_overrides "$task")"
      local tag="temporal_${task}_reliability_gate_diag_train_noise${train_tag}_eval_noise${eval_noise_tag}"
      local overrides=(
        "$task_type_override"
        "$reverse_override"
        "controller.load_run=$checkpoint"
        "controller.temporal_aggregator=reliability_gated_gru"
        "controller.gate_history_update=gated"
        "inject_costmap_noise=True"
        "noise_prob=$noise_prob"
        "inference_noise_seed=${NOISE_SEED:-42}"
        "log_gate_diagnostics=True"
        "use_costmap_ema=False"
        "exp_name=$tag"
      )
      if [[ -n "${MAX_STEPS:-}" ]]; then
        overrides+=("max_steps=$MAX_STEPS")
      fi

      local cmd=(
        "$PY" -m temporal_objectreact.eval_runner
        -c configs/object_react_temporal.yaml
        --set "${overrides[@]}"
      )
      echo
      echo "[supervised-gate] stage=$stage task=$task noise=$noise_prob"
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '[supervised-gate] DRY_RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
      else
        "${cmd[@]}" 2>&1 | tee -a "out/eval_${tag}.log"
      fi
    done
  done

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$PY" scripts/evaluate_objecreact.py ./out/results/
  fi
}

analyze_stage() {
  local stage="$1"
  local condition_name="supervised"
  local train_tag="supervised"
  local output_dir="out/analysis/gate_detection_supervised"
  if [[ "$stage" == "finetune" ]]; then
    condition_name="supervised_finetune"
    train_tag="supervised_finetune"
    output_dir="out/analysis/gate_detection_supervised_finetune"
  fi

  "$PY" temporal_objectreact/analysis/analyze_gate_diagnostics.py \
    --condition train_noise02=02 \
    --condition "$condition_name=$train_tag" \
    --output-dir "$output_dir"
}

case "$PHASE" in
  finetune) train_stage finetune ;;
  full) train_stage full ;;
  eval-finetune) eval_stage finetune ;;
  eval-full) eval_stage full ;;
  analyze-finetune) analyze_stage finetune ;;
  analyze-full) analyze_stage full ;;
  *) echo "unknown phase: $PHASE" >&2; exit 1 ;;
esac
