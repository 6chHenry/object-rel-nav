#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
MODE="${2:-clean}"
if [[ -z "$TASK" ]]; then
  echo "Usage: $0 {imitate|alt_goal|shortcut|reverse} [clean|noise|ema|ema_noise]"
  exit 1
fi

MICROMAMBA_BIN="${MICROMAMBA_BIN:-${HOME}/.local/bin/micromamba}"
if [[ -x "/root/.local/bin/micromamba" ]]; then
  MICROMAMBA_BIN="/root/.local/bin/micromamba"
fi

if [[ -d "/root/.micromamba/envs/nav" ]]; then
  export MAMBA_ROOT_PREFIX="/root/.micromamba"
fi

PYTHON_BIN=""
if [[ -x "portable_envs/nav_env/bin/python" ]]; then
  PYTHON_BIN="portable_envs/nav_env/bin/python"
fi

case "$TASK" in
  imitate)
    CFG="configs/benchmarks/object_react_gt_imitate.yaml"
    ;;
  alt_goal)
    CFG="configs/benchmarks/object_react_gt_alt_goal.yaml"
    ;;
  shortcut)
    CFG="configs/benchmarks/object_react_gt_shortcut.yaml"
    ;;
  reverse)
    CFG="configs/benchmarks/object_react_gt_reverse.yaml"
    ;;
  *)
    echo "Unknown task: $TASK"
    exit 1
    ;;
esac

case "$MODE" in
  clean|noise|ema|ema_noise)
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 1
    ;;
esac

mkdir -p out/logs
mkdir -p out/generated_configs
CFG="out/generated_configs/${TASK}_${MODE}.yaml"
RUN_LOG="out/logs/${TASK}.${MODE}.run.log"
EVAL_LOG="out/logs/${TASK}.${MODE}.eval.log"

if [[ -n "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" scripts/build_benchmark_cfg.py --task "$TASK" --mode "$MODE" --output "$CFG"
else
  "$MICROMAMBA_BIN" run -n nav python scripts/build_benchmark_cfg.py --task "$TASK" --mode "$MODE" --output "$CFG"
fi

echo "[run] task=$TASK mode=$MODE cfg=$CFG" | tee "$RUN_LOG"
if [[ -n "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" main.py -c "$CFG" 2>&1 | tee -a "$RUN_LOG"
else
  "$MICROMAMBA_BIN" run -n nav python main.py -c "$CFG" 2>&1 | tee -a "$RUN_LOG"
fi

RESULT_DIR="$(grep 'Logging to:' "$RUN_LOG" | tail -n 1 | sed 's/.*Logging to: //')"
if [[ -z "$RESULT_DIR" ]]; then
  echo "Could not determine result directory from $RUN_LOG" | tee "$EVAL_LOG"
  exit 1
fi

echo "[eval] results=$RESULT_DIR" | tee "$EVAL_LOG"
if [[ -n "$PYTHON_BIN" ]]; then
  "$PYTHON_BIN" scripts/evaluate_objecreact.py "$RESULT_DIR" 2>&1 | tee -a "$EVAL_LOG"
else
  "$MICROMAMBA_BIN" run -n nav python scripts/evaluate_objecreact.py "$RESULT_DIR" 2>&1 | tee -a "$EVAL_LOG"
fi

echo "[done] run_log=$RUN_LOG eval_log=$EVAL_LOG result_dir=$RESULT_DIR"
