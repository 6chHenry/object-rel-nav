#!/usr/bin/env bash
# Train gated GRU and reliability-gated GRU with noise_p=0.0, then evaluate both.
#
# Usage:
#   tmux new -s gated_noise00
#   cd /path/to/object-rel-nav
#   CUDA_VISIBLE_DEVICES=0 bash temporal_objectreact/scripts/run_gated_grus_noise00_train_eval.sh
#
# Optional env:
#   DEVICE=cuda:0
#   MAX_ITERS=20      # smoke test only
#   FORCE=1           # allow overwriting existing noise00 checkpoints
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="$PWD/portable_envs/nav_env/bin/python"
DEVICE="${DEVICE:-cuda:0}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: missing portable python: $PY" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/libs/control/object_react/train${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/objectreact-mpl-${USER:-root}}"
export PATH="$(dirname "$PY"):$PATH"
mkdir -p "$MPLCONFIGDIR"

LIBS=(
  "$PWD/portable_envs/nav_env/lib"
  "/usr/lib/x86_64-linux-gnu"
  "/lib/x86_64-linux-gnu"
  "/usr/local/nvidia/lib"
  "/usr/local/nvidia/lib64"
)
export LD_LIBRARY_PATH="$(IFS=:; echo "${LIBS[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

TRAIN_EXTRA=()
if [[ -n "${MAX_ITERS:-}" ]]; then
  TRAIN_EXTRA+=(--max_iters "$MAX_ITERS")
fi

run_one() {
  local label="$1"
  local aggregator="$2"
  local cfg="$3"
  local out_dir="$4"
  local exp_name="$5"

  if [[ ! -f "$cfg" ]]; then
    echo "ERROR: missing config: $cfg" >&2
    exit 1
  fi
  if [[ -e "$out_dir/latest.pth" && "${FORCE:-0}" != "1" ]]; then
    echo "ERROR: checkpoint already exists: $out_dir/latest.pth" >&2
    echo "Set FORCE=1 to rerun and overwrite this experiment output." >&2
    exit 1
  fi

  mkdir -p "$out_dir"
  echo
  echo "================ train $label ================"
  echo "[noise00] python=$PY"
  echo "[noise00] device=$DEVICE cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "[noise00] cfg=$cfg out=$out_dir aggregator=$aggregator"
  "$PY" -m temporal_objectreact.train_temporal \
    --config "$cfg" \
    --out "$out_dir" \
    --device "$DEVICE" \
    "${TRAIN_EXTRA[@]}" \
    2>&1 | tee "$out_dir/train.log"

  if [[ ! -s "$out_dir/latest.pth" ]]; then
    echo "ERROR: training finished but checkpoint is missing: $out_dir/latest.pth" >&2
    exit 1
  fi

  echo
  echo "================ eval $label ================"
  PYTHON="$PY" \
  LOAD_RUN="$out_dir/latest.pth" \
  AGGREGATOR="$aggregator" \
  EXP_NAME="$exp_name" \
  bash temporal_objectreact/scripts/21_eval_all_tasks.sh temporal
}

run_one \
  "gated_gru_noise00" \
  "gated_gru" \
  "temporal_objectreact/configs/train/temporal_gated_gru_noise00.yaml" \
  "logs/temporal_gated_gru_noise00" \
  "gated_gru_noise00"

run_one \
  "reliability_gated_gru_noise00" \
  "reliability_gated_gru" \
  "temporal_objectreact/configs/train/temporal_reliability_gated_gru_noise00.yaml" \
  "logs/temporal_reliability_gated_gru_noise00" \
  "reliability_gated_gru_noise00"

echo
echo "[noise00] all training and evaluation jobs finished"
