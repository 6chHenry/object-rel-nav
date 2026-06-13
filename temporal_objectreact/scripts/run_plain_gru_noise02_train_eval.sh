#!/usr/bin/env bash
# Train plain GRU with noise_p=0.2, then run the standard four-task eval.
#
# Usage:
#   tmux new -s gru_noise02
#   cd /path/to/object-rel-nav
#   bash temporal_objectreact/scripts/run_plain_gru_noise02_train_eval.sh
#
# Optional env:
#   CUDA_VISIBLE_DEVICES=1
#   DEVICE=cuda:0
#   MAX_ITERS=20      # smoke test only
#   FORCE=1           # allow overwriting an existing output dir
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="$PWD/portable_envs/nav_env/bin/python"
CFG="temporal_objectreact/configs/train/temporal_gru_noise02.yaml"
OUT_DIR="${OUT_DIR:-logs/temporal_gru_noise02}"
EXP_NAME="${EXP_NAME:-gru_noise02}"
DEVICE="${DEVICE:-cuda:0}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: missing portable python: $PY" >&2
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: missing config: $CFG" >&2
  exit 1
fi
if [[ -e "$OUT_DIR/latest.pth" && "${FORCE:-0}" != "1" ]]; then
  echo "ERROR: checkpoint already exists: $OUT_DIR/latest.pth" >&2
  echo "Set FORCE=1 to rerun and overwrite this experiment output." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PWD/libs/control/object_react/train${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/objectreact-mpl-${USER:-root}}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

LIBS=(
  "$PWD/portable_envs/nav_env/lib"
  "/usr/lib/x86_64-linux-gnu"
  "/lib/x86_64-linux-gnu"
  "/usr/local/nvidia/lib"
  "/usr/local/nvidia/lib64"
)
export LD_LIBRARY_PATH="$(IFS=:; echo "${LIBS[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[gru-noise02] python=$PY"
echo "[gru-noise02] device=$DEVICE cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[gru-noise02] train cfg=$CFG out=$OUT_DIR"
TRAIN_EXTRA=()
if [[ -n "${MAX_ITERS:-}" ]]; then
  TRAIN_EXTRA+=(--max_iters "$MAX_ITERS")
fi
"$PY" -m temporal_objectreact.train_temporal \
  --config "$CFG" \
  --out "$OUT_DIR" \
  --device "$DEVICE" \
  "${TRAIN_EXTRA[@]}" \
  2>&1 | tee "$OUT_DIR/train.log"

if [[ ! -s "$OUT_DIR/latest.pth" ]]; then
  echo "ERROR: training finished but checkpoint is missing: $OUT_DIR/latest.pth" >&2
  exit 1
fi

echo "[gru-noise02] eval checkpoint=$OUT_DIR/latest.pth exp_name=$EXP_NAME"
PYTHON="$PY" \
LOAD_RUN="$OUT_DIR/latest.pth" \
AGGREGATOR="gru" \
EXP_NAME="$EXP_NAME" \
bash temporal_objectreact/scripts/21_eval_all_tasks.sh temporal
