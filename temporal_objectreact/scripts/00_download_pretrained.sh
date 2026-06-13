#!/usr/bin/env bash
# Download the upstream ObjectReact pretrained checkpoint (~18 MB) so that the
# temporal variants can warm-start their backbone + head from it.
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root: object-rel-nav/

mkdir -p model_weights

DST=model_weights/object_react_latest.pth
if [[ -s "$DST" ]]; then
  echo "[00_download_pretrained] $DST already present, skipping."
  ln -sfn object_react_latest.pth model_weights/latest.pth
  exit 0
fi

URL="https://huggingface.co/oravus/ObjectReact/resolve/main/latest.pth"
echo "[00_download_pretrained] fetching $URL -> $DST"
# We use curl rather than huggingface-cli so the script has no python deps.
curl -L --fail --retry 5 --retry-delay 3 -o "$DST" "$URL"
ln -sfn object_react_latest.pth model_weights/latest.pth
echo "[00_download_pretrained] done."
