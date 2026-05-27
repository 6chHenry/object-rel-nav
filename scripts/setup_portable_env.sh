#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="portable_envs/nav_env.tar.gz"
TARGET="portable_envs/nav_env"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "Missing archive: $ARCHIVE"
  exit 1
fi

mkdir -p portable_envs
rm -rf "$TARGET"
mkdir -p "$TARGET"
tar -xzf "$ARCHIVE" -C "$TARGET"
"$TARGET/bin/python" "$TARGET/bin/conda-unpack"
echo "Portable env ready at $TARGET"
