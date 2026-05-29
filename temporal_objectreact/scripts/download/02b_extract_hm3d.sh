#!/usr/bin/env bash
# 解压 HM3D 4 个 tar 包到 data/hm3d_v0.2/val/。
set -euo pipefail
cd "$(dirname "$0")/../../.."

DST=data/hm3d_v0.2/val
for f in hm3d-val-glb-v0.2.tar hm3d-val-habitat-v0.2.tar \
         hm3d-val-semantic-annots-v0.2.tar hm3d-val-semantic-configs-v0.2.tar; do
  if [[ -s "$DST/$f" ]]; then
    echo "[extract] $f"
    tar -xf "$DST/$f" -C "$DST"
  else
    echo "[extract] 缺文件: $DST/$f"
  fi
done
echo "[extract] 完成。检查 $DST/ 应该看到一堆形如 00800-TEEsavR23oF/ 的子目录。"
ls "$DST/" | head -5
