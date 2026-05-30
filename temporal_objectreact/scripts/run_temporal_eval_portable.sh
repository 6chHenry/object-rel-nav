#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="$PWD/portable_envs/nav_env/bin/python"
CKPT="${LOAD_RUN:-logs/temporal_gated_gru/latest.pth}"
AGG="${AGGREGATOR:-gated_gru}"
VARIANT="${1:-temporal}"

if [[ ! -x "$PY" ]]; then
  echo "missing portable python: $PY" >&2
  exit 1
fi
if [[ "$VARIANT" == "temporal" && ! -s "$CKPT" ]]; then
  echo "missing checkpoint: $CKPT" >&2
  exit 1
fi

export PYTHONPATH="$PWD:$PWD/libs/control/object_react/train${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/objectreact-mpl-${USER:-root}}"
mkdir -p "$MPLCONFIGDIR"

LIBS=(
  "$PWD/portable_envs/nav_env/lib"
  "/usr/lib/x86_64-linux-gnu"
  "/lib/x86_64-linux-gnu"
  "/usr/local/nvidia/lib"
  "/usr/local/nvidia/lib64"
)
export LD_LIBRARY_PATH="$(IFS=:; echo "${LIBS[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHON="$PY"

echo "[portable-eval] python=$PY"
echo "[portable-eval] variant=$VARIANT checkpoint=$CKPT aggregator=$AGG"
echo "[portable-eval] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
"$PY" -c "import habitat_sim; print('[portable-eval] habitat_sim import ok')"

if [[ "$VARIANT" == "baseline" ]]; then
  bash temporal_objectreact/scripts/21_eval_all_tasks.sh baseline
else
  LOAD_RUN="$CKPT" AGGREGATOR="$AGG" \
    bash temporal_objectreact/scripts/21_eval_all_tasks.sh temporal
fi
