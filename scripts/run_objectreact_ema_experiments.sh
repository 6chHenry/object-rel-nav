#!/usr/bin/env bash
set -euo pipefail

mkdir -p out/logs
MASTER_LOG="out/logs/all_ema_experiments.log"

run_mode() {
  local mode="$1"
  echo "[$(date -u +%F_%T)] START mode=${mode}" | tee -a "$MASTER_LOG"
  bash scripts/run_objectreact_gt_all.sh "$mode" 2>&1 | tee -a "$MASTER_LOG"
  echo "[$(date -u +%F_%T)] DONE mode=${mode}" | tee -a "$MASTER_LOG"
}

run_mode noise
run_mode ema
run_mode ema_noise

echo "[$(date -u +%F_%T)] ALL_DONE" | tee -a "$MASTER_LOG"
