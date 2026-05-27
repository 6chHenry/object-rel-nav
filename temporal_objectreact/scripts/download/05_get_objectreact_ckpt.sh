#!/usr/bin/env bash
# Step 5: 下上游 ObjectReact 预训练权重（约 18 MB）。
# 这步等价于 scripts/00_download_pretrained.sh，单独放在 download/ 下方便统一管理。
set -euo pipefail
cd "$(dirname "$0")/../../.."
bash temporal_objectreact/scripts/00_download_pretrained.sh
