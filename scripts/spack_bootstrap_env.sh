#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" 1>&2; exit 1; }

# -----------------------------------------------------------------------------
# scripts/spack_bootstrap_env.sh
#
# One-time provisioning of a Spack environment for butterfly on the HPC.
# - Creates (if missing) a Spack env named $SPACK_ENV_NAME
# - Adds the dependency set (no hardcoded hashes)
# - Concretizes + installs
# - Leaves behind spack.yaml + spack.lock inside the env directory
#
# Typical use:
#   bash scripts/spack_bootstrap_env.sh
#
# Optional overrides:
#   SPACK_SETUP=/path/to/spack/share/spack/setup-env.sh \
#   SPACK_ENV_NAME=butterfly \
#   WITH_EMBREE=1 \
#   WITH_PYTHON=1 \
#   WITH_LAPACKE=1 \
#   WITH_FLEXIBLAS=1 \
#   bash scripts/spack_bootstrap_env.sh
#
# Notes:
# - This script does NOT build butterfly. Use scripts/build_install_butterfly.sh for that.
# - On some clusters you may want to load a compiler module BEFORE concretizing.
#   Set COMPILER_MODULE (e.g. gcc/12.1.0) if needed.
# -----------------------------------------------------------------------------

# --- config (override via env vars) ---
SPACK_SETUP="${SPACK_SETUP:-$HOME/nobackup/.spack_repo/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-butterfly}"

# Feature toggles
WITH_EMBREE="${WITH_EMBREE:-1}"
WITH_PYTHON="${WITH_PYTHON:-1}"

# These match your Meson expectations on Linux (pkg-config deps)
WITH_FLEXIBLAS="${WITH_FLEXIBLAS:-1}"
WITH_LAPACKE="${WITH_LAPACKE:-1}"

# Optional: add arpack (your current meson.build treats it optional)
WITH_ARPACK="${WITH_ARPACK:-0}"

# Optional: include netlib-lapack (headers sometimes useful; lapacke is usually enough)
WITH_NETLIB_LAPACK="${WITH_NETLIB_LAPACK:-0}"

# Optional cluster module
COMPILER_MODULE="${COMPILER_MODULE:-}"   # e.g. "gcc/12.1.0"

# Concretize/install knobs
CONCRETIZE_FLAGS="${CONCRETIZE_FLAGS:--f}"
INSTALL_FLAGS="${INSTALL_FLAGS:-}"

# -----------------------------------------------------------------------------
# Spack init
# -----------------------------------------------------------------------------
[ -f "$SPACK_SETUP" ] || die "Missing SPACK_SETUP=$SPACK_SETUP"
# shellcheck disable=SC1090
. "$SPACK_SETUP"

if [ -n "$COMPILER_MODULE" ]; then
  command -v module >/dev/null 2>&1 && module load "$COMPILER_MODULE" || true
fi

# -----------------------------------------------------------------------------
# Create env if needed + activate
# -----------------------------------------------------------------------------
if ! spack env list | awk '{print $1}' | grep -qx "$SPACK_ENV_NAME"; then
  echo "[spack] creating env: $SPACK_ENV_NAME"
  spack env create "$SPACK_ENV_NAME"
fi

spack env activate "$SPACK_ENV_NAME" >/dev/null

# -----------------------------------------------------------------------------
# Add packages (idempotent)
# -----------------------------------------------------------------------------
echo "[spack] adding core build tools"
spack add ninja meson pkgconf

echo "[spack] adding math/linear-algebra deps"
spack add openblas
spack add suite-sparse
spack add gsl

if [ "$WITH_FLEXIBLAS" = "1" ]; then
  spack add flexiblas
fi

if [ "$WITH_LAPACKE" = "1" ]; then
  spack add lapacke
fi

if [ "$WITH_NETLIB_LAPACK" = "1" ]; then
  spack add netlib-lapack
fi

if [ "$WITH_ARPACK" = "1" ]; then
  # pick variants here if you need them; your old script used +icb
  spack add arpack-ng
fi

if [ "$WITH_EMBREE" = "1" ]; then
  # match your prior intent
  spack add "embree@4:"
fi

echo "[spack] adding optional test deps"
spack add cmocka

if [ "$WITH_PYTHON" = "1" ]; then
  echo "[spack] adding python build helpers (optional; venv still used for install)"
  spack add py-numpy py-cython py-pip py-setuptools py-scipy
fi

# -----------------------------------------------------------------------------
# Concretize + install (produces spack.lock)
# -----------------------------------------------------------------------------
echo "[spack] concretizing (${CONCRETIZE_FLAGS})"
# shellcheck disable=SC2086
spack concretize ${CONCRETIZE_FLAGS}

echo "[spack] installing (${INSTALL_FLAGS:-<none>})"
# shellcheck disable=SC2086
spack install ${INSTALL_FLAGS}

# -----------------------------------------------------------------------------
# Show env location + where lock lives
# -----------------------------------------------------------------------------
ENV_DIR="$(spack env status | sed -n 's/^Environment: *//p' | awk '{print $1}' || true)"
if [ -n "$ENV_DIR" ] && [ -d "$ENV_DIR" ]; then
  echo "[spack] env dir: $ENV_DIR"
  [ -f "$ENV_DIR/spack.yaml" ] && echo "[spack] wrote: $ENV_DIR/spack.yaml"
  [ -f "$ENV_DIR/spack.lock" ] && echo "[spack] wrote: $ENV_DIR/spack.lock"
else
  echo "[spack] env dir: (could not detect via 'spack env status'; run: spack env status)"
fi

echo "[spack] done."
echo
echo "Next:"
echo "  . \"$SPACK_SETUP\""
echo "  spack env activate \"$SPACK_ENV_NAME\""
echo "  bash scripts/build_install_butterfly.sh"
