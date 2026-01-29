#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" 1>&2; exit 1; }

# COMPILER_MODULE=gcc/12.1.0 USE_SPACK=1 SPACK_SETUP="$HOME/nobackup/.spack_repo/share/spack/setup-env.sh" \
# SPACK_ENV_NAME=butterfly PRIMME_ROOT="$HOME/nobackup/primme" WITH_EMBREE=1 WITH_PYTHON=1 CLEAN=1 \
# bash scripts/build_install_butterfly.sh

# -------------------------
# Config (override via env)
# -------------------------
USE_SPACK="${USE_SPACK:-1}"

SPACK_SETUP="${SPACK_SETUP:-$HOME/nobackup/.spack_repo/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-butterfly}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SRC_DIR:-$REPO_ROOT}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build}"
VENV_DIR="${VENV_DIR:-$HOME/nobackup/venvs/butterfly}"

WITH_EMBREE="${WITH_EMBREE:-1}"
WITH_PYTHON="${WITH_PYTHON:-1}"

# 0 = incremental (default), 1 = wipe build dir and reconfigure
CLEAN="${CLEAN:-0}"

# 0 = always pip install/upgrade numpy/cython/scipy, 1 = skip if already present
SKIP_PIP="${SKIP_PIP:-0}"

# PCC is vendored in your repo
PCC_DIR="${PCC_DIR:-third_party/flux_pcc}"

# PRIMME root (required for your fork)
PRIMME_ROOT="${PRIMME_ROOT:-$HOME/nobackup/primme}"

# Optional: load a compiler module on HPC (leave empty if not needed)
COMPILER_MODULE="${COMPILER_MODULE:-}"   # e.g. "gcc/12.1.0"

pick_libdir() {
  local p="$1"
  if   [ -d "$p/lib64" ]; then echo "$p/lib64"
  elif [ -d "$p/lib"   ]; then echo "$p/lib"
  else die "No lib/ or lib64/ under $p"
  fi
}

add_rpath_L() {
  [ -n "${1:-}" ] || return 0
  RPATH_FLAGS+=("-L$1" "-Wl,-rpath,$1")
}

# -------------------------
# (Optional) HPC module
# -------------------------
if [ -n "$COMPILER_MODULE" ]; then
  command -v module >/dev/null 2>&1 && module load "$COMPILER_MODULE" || true
fi

# -------------------------
# Spack env (optional)
# -------------------------
if [ "$USE_SPACK" = "1" ]; then
  [ -f "$SPACK_SETUP" ] || die "Missing SPACK_SETUP=$SPACK_SETUP"
  # shellcheck disable=SC1090
  . "$SPACK_SETUP"
  spack env activate "$SPACK_ENV_NAME" >/dev/null

  # Ensure pkg-config can find deps (this is the big missing piece vs your old script)
  spack load pkgconf ninja meson >/dev/null 2>&1 || true

  # Load build/runtime deps (no hashes)
  spack load suite-sparse gsl >/dev/null 2>&1 || die "spack load suite-sparse/gsl failed (not installed in env?)"
  spack load netlib-lapack >/dev/null 2>&1 || die "spack load netlib-lapack failed (add them to the env)"
  spack load lapacke >/dev/null 2>&1 || true # die "spack load lapacke failed (add them to the env)"
  spack load flexiblas >/dev/null 2>&1 || true # "spack load flexiblas failed (add flexiblas to the env, or use OpenBLAS fallback in meson.build)"
  # openblas is just a common flexiblas backend; keep optional
  spack load openblas >/dev/null 2>&1 || die "spack load openblas failed (add them to the env)"

  if [ "$WITH_EMBREE" = "1" ]; then
    spack load embree >/dev/null 2>&1 || true
  fi
fi

# -------------------------
# venv
# -------------------------
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel >/dev/null

if [ "$SKIP_PIP" = "0" ]; then
  python -m pip install -U numpy cython scipy >/dev/null
else
  python - <<'PY' >/dev/null 2>&1 || python -m pip install -U numpy cython scipy >/dev/null
import numpy, Cython, scipy
PY
fi

# -------------------------
# Source sanity
# -------------------------
[ -d "$SRC_DIR/include" ] || die "Not a butterfly repo? missing: $SRC_DIR/include"
[ -d "$SRC_DIR/$PCC_DIR" ] || die "PCC_DIR not found: $SRC_DIR/$PCC_DIR"

INC="$SRC_DIR/include"

# -------------------------
# Locate deps (Spack or system)
# -------------------------
SS=""
OB=""
LAP=""

if [ "$USE_SPACK" = "1" ]; then
  SS="$(spack location -i suite-sparse)" || die "spack env missing suite-sparse"
  # openblas is optional (commonly used as flexiblas backend)
  OB="$(spack location -i openblas 2>/dev/null || true)"
  LAP="$(spack location -i netlib-lapack 2>/dev/null || true)"  # optional (headers)
fi

# PRIMME required (your fork)
[ -d "$PRIMME_ROOT" ] || die "PRIMME_ROOT does not exist: $PRIMME_ROOT"
[ -d "$PRIMME_ROOT/include" ] || die "PRIMME_ROOT missing include/: $PRIMME_ROOT"

# IMPORTANT: meson option expects a root with include/ and lib/ (not lib64-only)
# If PRIMME uses lib64, create a lib -> lib64 shim locally inside PRIMME_ROOT.
if [ ! -d "$PRIMME_ROOT/lib" ] && [ -d "$PRIMME_ROOT/lib64" ]; then
  ln -snf "$PRIMME_ROOT/lib64" "$PRIMME_ROOT/lib"
fi
[ -d "$PRIMME_ROOT/lib" ] || die "PRIMME_ROOT missing lib/: $PRIMME_ROOT"

# If PRIMME built only versioned .so without the unversioned linker name, add it.
if [ -f "$PRIMME_ROOT/lib/libprimme.so.3" ] && [ ! -f "$PRIMME_ROOT/lib/libprimme.so" ]; then
  ln -snf "libprimme.so.3" "$PRIMME_ROOT/lib/libprimme.so"
elif ls "$PRIMME_ROOT/lib"/libprimme.so.* >/dev/null 2>&1 && [ ! -f "$PRIMME_ROOT/lib/libprimme.so" ]; then
  ln -snf "$(basename "$(ls -1 "$PRIMME_ROOT/lib"/libprimme.so.* | head -n1)")" "$PRIMME_ROOT/lib/libprimme.so"
fi

PRIMME_LIB="$(pick_libdir "$PRIMME_ROOT")"

# venv lib dirs
VENV_LIB64="$VENV_DIR/lib64"
VENV_LIB="$VENV_DIR/lib"

# -------------------------
# Build dir handling
# -------------------------
if [ "$CLEAN" = "1" ]; then
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

# -------------------------
# ss-compat tree (like old script)
# -------------------------
if [ "$USE_SPACK" = "1" ]; then
  SSCOMPAT="$BUILD_DIR/ss-compat/suitesparse"
  mkdir -p "$SSCOMPAT"
  ln -sf "$SS/include/cholmod.h" "$SSCOMPAT/cholmod.h"
  ln -sf "$SS/include/umfpack.h" "$SSCOMPAT/umfpack.h"
fi

# -------------------------
# RPATH/L flags (match old behavior)
# -------------------------
RPATH_FLAGS=()

if [ "$USE_SPACK" = "1" ]; then
  SS_LIB="$(pick_libdir "$SS")"
  add_rpath_L "$SS_LIB"

  # LAPACK runtime (needed when PRIMME has unresolved LAPACK/BLAS symbols)
  LAPROOT="$(spack location -i netlib-lapack 2>/dev/null || true)"
  if [ -n "$LAPROOT" ] && [ -d "$LAPROOT" ]; then
    add_rpath_L "$(pick_libdir "$LAPROOT")"
  fi

  # OpenBLAS: optional flexiblas backend
  OBROOT="$(spack location -i openblas 2>/dev/null || true)"
  if [ -n "$OBROOT" ] && [ -d "$OBROOT" ]; then
    add_rpath_L "$(pick_libdir "$OBROOT")"
  fi

  # flexiblas optional but your meson.build uses it on linux via pkg-config
  FLEX="$(spack location -i flexiblas 2>/dev/null || true)"
  if [ -n "$FLEX" ] && [ -d "$FLEX" ]; then
    add_rpath_L "$(pick_libdir "$FLEX")"
  fi

  if [ "$WITH_EMBREE" = "1" ]; then
    EMB="$(spack location -i embree 2>/dev/null || true)"
    [ -n "$EMB" ] || die "WITH_EMBREE=1 but embree not in spack env"
    add_rpath_L "$(pick_libdir "$EMB")"
  fi
fi

# PRIMME rpath (do NOT add -lprimme here; avoid Meson compiler sanity failure)
add_rpath_L "$PRIMME_LIB"

# venv rpaths for python extensions
[ -d "$VENV_LIB64" ] && add_rpath_L "$VENV_LIB64"
[ -d "$VENV_LIB" ]   && add_rpath_L "$VENV_LIB"

# preserve $ORIGIN relative rpaths from old script
RPATH_FLAGS+=("-Wl,-rpath,\$ORIGIN/../../lib64" "-Wl,-rpath,\$ORIGIN/../../lib")

# -------------------------
# CFLAGS (match old behavior)
# -------------------------
CFLAGS=(
  "-DBF_DOUBLE"
  "-I$INC"
  "-include" "$INC/bf/blas.h"
)

if [ "$USE_SPACK" = "1" ]; then
  CFLAGS+=("-I$SS/include" "-I$BUILD_DIR/ss-compat")
  if [ -n "${OB:-}" ] && [ -d "$OB/include" ]; then
    CFLAGS+=("-I$OB/include")
  fi
  if [ -n "$LAP" ] && [ -d "$LAP/include" ]; then
    CFLAGS+=("-I$LAP/include")
  fi
fi

# -------------------------
# Meson options
# -------------------------
MESON_ARGS=(
  "--prefix" "$VENV_DIR"
  "-Dbuildtype=release"
  "-Dpcc_dir=$PCC_DIR"
  "-Dprimme=enabled"
  "-Dprimme_root=$PRIMME_ROOT"
)

if [ "$WITH_EMBREE" = "1" ]; then MESON_ARGS+=("-Dembree=enabled"); else MESON_ARGS+=("-Dembree=disabled"); fi
if [ "$WITH_PYTHON" = "1" ]; then MESON_ARGS+=("-Dpython=enabled"); else MESON_ARGS+=("-Dpython=disabled"); fi

# -------------------------
# Configure/build/install
# -------------------------
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
  meson setup "$BUILD_DIR" "${MESON_ARGS[@]}" \
    --native-file <(cat <<EOF
[binaries]
python = '$(command -v python3)'
EOF
) \
    -Dc_args="${CFLAGS[*]}" \
    -Dc_link_args="${RPATH_FLAGS[*]}"
else
  meson configure "$BUILD_DIR" "${MESON_ARGS[@]}" >/dev/null
fi

ninja -C "$BUILD_DIR" -v
meson install -C "$BUILD_DIR"

python -c "import butterfly; print('butterfly OK')"
if [ "$WITH_PYTHON" = "1" ]; then
  python -c "import thermal_bf; print('thermal_bf OK')"
fi
