#!/usr/bin/env bash
set -euo pipefail

mkdir -p out/logs
MASTER_LOG="out/logs/all_gt_benchmarks.log"
MODE="${1:-clean}"

run_task() {
  local task="$1"
  echo "[$(date -u +%F_%T)] START ${task} mode=${MODE}" | tee -a "$MASTER_LOG"
  bash scripts/run_objectreact_gt_benchmark.sh "$task" "$MODE" 2>&1 | tee -a "$MASTER_LOG"
  echo "[$(date -u +%F_%T)] DONE ${task} mode=${MODE}" | tee -a "$MASTER_LOG"
}

run_task imitate
run_task alt_goal
run_task shortcut
run_task reverse

echo "[$(date -u +%F_%T)] ALL_DONE mode=${MODE}" | tee -a "$MASTER_LOG"
