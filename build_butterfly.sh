#!/usr/bin/env bash
set -euo pipefail

# Ensure env is loaded
: "${AR_PREFIX:?source env_macos_butterfly.sh first}"
: "${SS_PREFIX:?source env_macos_butterfly.sh first}"

rm -rf build
meson setup build \
  -Dbuildtype=release \
  -Dembree=enabled \
  -Dpython=enabled

meson compile -C build

echo "OK: build finished"
echo "Remember to export PYTHONPATH for this shell:"
echo "  export PYTHONPATH=\"$(pwd)/build/wrappers/python:\$PYTHONPATH\""
