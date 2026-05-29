#!/usr/bin/env bash
# Chain train: gru -> ema (gated_gru is launched separately).
# Waits for the currently running gated_gru job to finish before starting,
# then runs gru and ema sequentially on the same GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."

GATED_PID_FILE=logs/temporal_gated_gru/train.pid
if [[ -f "$GATED_PID_FILE" ]]; then
  PID="$(cat "$GATED_PID_FILE")"
  echo "[chain] waiting for gated_gru run (PID $PID) to finish..."
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
  echo "[chain] gated_gru finished, starting gru"
fi

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-nav}"
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda activate "$CONDA_ENV"
  else
    echo "[chain] conda env '$CONDA_ENV' not found; using current shell python"
  fi
else
  echo "[chain] conda init script '$CONDA_SH' not found; using current shell python"
fi

PYTHON_BIN="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):$(pwd)/libs/control/object_react/train"

for VAR in gru ema; do
  OUT="logs/temporal_${VAR}"
  mkdir -p "$OUT"
  echo "[chain] === training $VAR -> $OUT ==="
  "$PYTHON_BIN" -m temporal_objectreact.train_temporal \
    --config "temporal_objectreact/configs/train/temporal_${VAR}.yaml" \
    --out "$OUT" --device cuda:0 \
    2>&1 | tee "$OUT/train.log"
done

echo "[chain] all variants done."
