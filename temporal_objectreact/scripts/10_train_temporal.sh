#!/usr/bin/env bash
# Train one of the temporal-aggregation variants.
#
# Usage:
#   bash temporal_objectreact/scripts/10_train_temporal.sh <variant> [max_iters]
#
# <variant> selects which YAML under temporal_objectreact/configs/train/ to use.
# Currently supported: gru, gated_gru, ema.
#
# [max_iters] caps iterations per epoch (smoke-test mode). Omit for full epochs.
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root

VARIANT="${1:-gated_gru}"
MAX_ITERS="${2:-}"

CFG="temporal_objectreact/configs/train/temporal_${VARIANT}.yaml"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2
  echo "Supported variants: gru, gated_gru, ema" >&2
  exit 1
fi

OUT_DIR="logs/temporal_${VARIANT}"
mkdir -p "$OUT_DIR"

EXTRA=()
if [[ -n "$MAX_ITERS" ]]; then
  EXTRA+=(--max_iters "$MAX_ITERS")
fi

echo "[10_train_temporal] variant=$VARIANT  out=$OUT_DIR  cfg=$CFG"
PYTHONPATH="$(pwd)":"$(pwd)/libs/control/object_react/train" \
python -m temporal_objectreact.train_temporal \
  --config "$CFG" \
  --out "$OUT_DIR" \
  --device cuda:0 \
  "${EXTRA[@]}" \
  2>&1 | tee "$OUT_DIR/train.log"

echo "[10_train_temporal] checkpoints in $OUT_DIR"
