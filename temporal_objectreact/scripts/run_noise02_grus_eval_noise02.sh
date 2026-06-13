#!/usr/bin/env bash
# Evaluate the three GRU variants trained with noise_p=0.2 under inference-time
# costmap noise. This reuses existing checkpoints and does not train.
#
# Usage:
#   tmux new -s gru_eval_noise02
#   cd /path/to/object-rel-nav
#   CUDA_VISIBLE_DEVICES=0 bash temporal_objectreact/scripts/run_noise02_grus_eval_noise02.sh
#
# Optional env:
#   NOISE_PROB=0.2
#   NOISE_SEED=42
#   MAX_STEPS=300
#   DRY_RUN=1      # print commands without running eval
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="${PYTHON:-$PWD/portable_envs/nav_env/bin/python}"
CFG="configs/object_react_temporal.yaml"
NOISE_PROB="${NOISE_PROB:-0.2}"
NOISE_SEED="${NOISE_SEED:-42}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: missing python: $PY" >&2
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: missing config: $CFG" >&2
  exit 1
fi

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

tasks=(imitate alt_goal shortcut reverse)
methods=(gru gated_gru reliability_gated_gru)

checkpoint_for() {
  case "$1" in
    gru) echo "logs/temporal_gru_noise02/latest.pth" ;;
    gated_gru) echo "logs/temporal_gated_gru/latest.pth" ;;
    reliability_gated_gru) echo "logs/temporal_reliability_gated_gru/latest.pth" ;;
    *) echo "unknown method: $1" >&2; exit 1 ;;
  esac
}

exp_suffix_for() {
  case "$1" in
    gru) echo "gru_train_noise02_eval_noise02" ;;
    gated_gru) echo "gated_gru_train_noise02_eval_noise02" ;;
    reliability_gated_gru) echo "reliability_gated_gru_train_noise02_eval_noise02" ;;
    *) echo "unknown method: $1" >&2; exit 1 ;;
  esac
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

for method in "${methods[@]}"; do
  ckpt="$(checkpoint_for "$method")"
  exp_suffix="$(exp_suffix_for "$method")"
  if [[ ! -s "$ckpt" ]]; then
    echo "ERROR: missing checkpoint for $method: $ckpt" >&2
    exit 1
  fi

  for task in "${tasks[@]}"; do
    read -r task_type_override reverse_override <<<"$(task_overrides "$task")"
    tag="temporal_${task}_${exp_suffix}"

    overrides=(
      "$task_type_override"
      "$reverse_override"
      "controller.load_run=$ckpt"
      "controller.temporal_aggregator=$method"
      "inject_costmap_noise=True"
      "noise_prob=$NOISE_PROB"
      "inference_noise_seed=$NOISE_SEED"
      "use_costmap_ema=False"
      "exp_name=$tag"
    )
    if [[ -n "${MAX_STEPS:-}" ]]; then
      overrides+=("max_steps=$MAX_STEPS")
    fi

    echo
    echo "================ $method / $task / eval noise=$NOISE_PROB ================"
    echo "[eval-noise02] checkpoint=$ckpt"
    echo "[eval-noise02] overrides=${overrides[*]}"

    cmd=("$PY" -m temporal_objectreact.eval_runner -c "$CFG" --set "${overrides[@]}")
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf '[eval-noise02] DRY_RUN:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}" 2>&1 | tee -a "out/eval_${tag}.log"
    fi
  done
done

echo
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[eval-noise02] dry run finished; no evaluations were launched"
else
  echo "[eval-noise02] aggregating numbers via evaluate_objecreact.py"
  "$PY" scripts/evaluate_objecreact.py ./out/results/
fi
