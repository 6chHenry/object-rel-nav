#!/usr/bin/env bash
# Step 2: 下 HM3D v0.2 val 场景（约 9 GB）。
# !! 必须先去 https://matterport.com/habitat-matterport-3d-research-dataset 注册免费账号 !!
# !! 然后到 https://api.matterport.com/resources/habitat 申请数据集访问，会拿到一个 token  !!
#
# 用法：
#   MATTERPORT_TOKEN=<你的 token> bash temporal_objectreact/scripts/download/02_get_hm3d.sh
#
# 也可以手动把 4 个 tar 下到 data/hm3d_v0.2/val/ 下面再跑 unzip 部分。

set -euo pipefail
cd "$(dirname "$0")/../../.."

TOKEN="${MATTERPORT_TOKEN:-}"
if [[ -z "$TOKEN" ]]; then
  cat <<'EOF'
[hm3d] 没设 MATTERPORT_TOKEN。

注册 + 申请步骤：
  1) https://matterport.com/habitat-matterport-3d-research-dataset 上注册免费账号
  2) 同意 Habitat-Matterport 协议，等邮件批准（通常几分钟）
  3) 登录后从下载链接里找到带 token 的 URL，例如：
     https://api.matterport.com/resources/habitat/hm3d-val-glb-v0.2.tar?xxx-token-yyy
     把 ?... 之后那一串（不含问号）作为 MATTERPORT_TOKEN 传进来。

或者直接把 4 个 tar 文件用浏览器下到 data/hm3d_v0.2/val/ 下面，然后跑：
  bash temporal_objectreact/scripts/download/02b_extract_hm3d.sh
EOF
  exit 1
fi

DST=data/hm3d_v0.2/val
mkdir -p "$DST"

FILES=(
  hm3d-val-glb-v0.2.tar
  hm3d-val-habitat-v0.2.tar
  hm3d-val-semantic-annots-v0.2.tar
  hm3d-val-semantic-configs-v0.2.tar
)

for f in "${FILES[@]}"; do
  if [[ -s "$DST/$f" ]]; then
    echo "[hm3d] 已存在 $f，跳过"
    continue
  fi
  url="https://api.matterport.com/resources/habitat/$f?$TOKEN"
  echo "[hm3d] 下载 $f"
  curl -L --fail --retry 5 --retry-delay 5 -o "$DST/$f" "$url"
done

echo "[hm3d] 全部下载完成，开始解压..."
bash temporal_objectreact/scripts/download/02b_extract_hm3d.sh
