#!/usr/bin/env bash
# Run the gate-detection experiment:
#   Reliability-Gated GRU trained with noise_p in {0.0, 0.2}
#   x four navigation tasks
#   x inference costmap noise_p=0.2, seed=42.
#
# This launches eight full simulator evaluations and must be run on a GPU node.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 \
#     bash temporal_objectreact/scripts/run_reliability_gate_diagnostics.sh
#
# Optional env:
#   PYTHON=/path/to/python
#   NOISE_PROB=0.2
#   NOISE_SEED=42
#   MAX_STEPS=300
#   DRY_RUN=1
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
export PATH="$(dirname "$PY"):$PATH"
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
train_noises=(00 02)

checkpoint_for() {
  case "$1" in
    00) echo "logs/temporal_reliability_gated_gru_noise00/latest.pth" ;;
    02) echo "logs/temporal_reliability_gated_gru/latest.pth" ;;
    *) echo "unknown training noise: $1" >&2; exit 1 ;;
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

for train_noise in "${train_noises[@]}"; do
  ckpt="$(checkpoint_for "$train_noise")"
  if [[ ! -s "$ckpt" ]]; then
    echo "ERROR: missing checkpoint: $ckpt" >&2
    exit 1
  fi

  for task in "${tasks[@]}"; do
    read -r task_type_override reverse_override <<<"$(task_overrides "$task")"
    tag="temporal_${task}_reliability_gate_diag_train_noise${train_noise}_eval_noise02"
    overrides=(
      "$task_type_override"
      "$reverse_override"
      "controller.load_run=$ckpt"
      "controller.temporal_aggregator=reliability_gated_gru"
      "inject_costmap_noise=True"
      "noise_prob=$NOISE_PROB"
      "inference_noise_seed=$NOISE_SEED"
      "log_gate_diagnostics=True"
      "use_costmap_ema=False"
      "exp_name=$tag"
    )
    if [[ -n "${MAX_STEPS:-}" ]]; then
      overrides+=("max_steps=$MAX_STEPS")
    fi

    cmd=("$PY" -m temporal_objectreact.eval_runner -c "$CFG" --set "${overrides[@]}")
    echo
    echo "================ train noise=$train_noise / $task ================"
    echo "[gate-diagnostics] checkpoint=$ckpt"
    echo "[gate-diagnostics] overrides=${overrides[*]}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf '[gate-diagnostics] DRY_RUN:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}" 2>&1 | tee -a "out/eval_${tag}.log"
    fi
  done
done

echo
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[gate-diagnostics] dry run finished; no evaluations were launched"
else
  echo "[gate-diagnostics] all eight evaluations finished"
  echo "[gate-diagnostics] analyze with:"
  echo "  $PY temporal_objectreact/analysis/analyze_gate_diagnostics.py"
fi
