#!/usr/bin/env bash
# Step 4 (训练用，不跑训练可跳过): 下上游 ObjectReact 训练数据。
# 数据集名: bigger_bot_0.3-sh_0.4   公开，HuggingFace 放出。
set -euo pipefail
cd "$(dirname "$0")/../../.."

TGT=libs/control/object_react/train/vint_train/data/data_splits/training/bigger_bot_0.3-sh_0.4
mkdir -p "$TGT"

# 上游用的是 HuggingFace 的 oravus/objectreact_train，里面有 trajectories.zip + train.json + test.json
echo "[hf] 拉训练数据"
huggingface-cli download oravus/objectreact_train \
  --repo-type dataset --local-dir "$TGT" \
  --include "*"

# trajectories.zip 解压到 trajectories/
if [[ -f "$TGT/trajectories.zip" ]]; then
  echo "[unzip] trajectories.zip"
  (cd "$TGT" && unzip -qo trajectories.zip)
fi

echo "[done] 训练数据放在 $TGT/"
ls "$TGT/"
