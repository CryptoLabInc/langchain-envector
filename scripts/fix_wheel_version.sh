#!/usr/bin/env bash

set -ex

PLATFORM=$(uname)
echo "[INFO] Detected platform: $PLATFORM"

set -euo pipefail

TOML_FILE="pyproject.toml"

: "${WHEEL_VERSION:?[ERROR] WHEEL_VERSION not set}"
echo "[INFO] Target version: $WHEEL_VERSION"

# update pyproject.toml version
echo "[INFO] Updating version in $TOML_FILE"
sed -i.bak -E \
  "s/^version[[:space:]]*=[[:space:]]*\"[^\"]+\"/version = \"$WHEEL_VERSION\"/" \
  "pyproject.toml"

echo "[INFO] All done!"
