#!/usr/bin/env bash
# Step 1: 解压已经下好的两个 zip 文件。
# - data/instance_imagenav_hm3d_v3.zip  (InstanceImageNav 数据集)
# - data/evaluation/hm3d_iin_val.zip    (我们的测试轨迹)
set -euo pipefail
cd "$(dirname "$0")/../../.."     # repo root: object-rel-nav/

cd data

if [[ -f instance_imagenav_hm3d_v3.zip ]]; then
  echo "[unzip] instance_imagenav_hm3d_v3.zip"
  unzip -qo instance_imagenav_hm3d_v3.zip
else
  echo "[unzip] skip: instance_imagenav_hm3d_v3.zip 不存在"
fi

if [[ -f evaluation/hm3d_iin_val.zip ]]; then
  echo "[unzip] evaluation/hm3d_iin_val.zip"
  unzip -qo evaluation/hm3d_iin_val.zip
else
  echo "[unzip] skip: evaluation/hm3d_iin_val.zip 不存在"
fi

# 顺便去 evaluation 子目录里看一下还有什么
ls -la
