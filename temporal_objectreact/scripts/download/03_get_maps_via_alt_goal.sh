#!/usr/bin/env bash
# Step 3: 下 shortcut 任务用的额外 maps_via_alt_goal.zip（HuggingFace，公开）。
set -euo pipefail
cd "$(dirname "$0")/../../.."

mkdir -p data/evaluation
cd data

# 用 huggingface-cli（在 nav conda env 里应该已经装好）
# 如果没装：pip install -U huggingface_hub
echo "[hf] 拉 maps_via_alt_goal.zip"
huggingface-cli download oravus/objectreact_hm3d_iin \
  --repo-type dataset --local-dir ./ \
  --include "evaluation/maps_via_alt_goal.zip"

echo "[hf] 解压"
unzip -qo evaluation/maps_via_alt_goal.zip
ls
