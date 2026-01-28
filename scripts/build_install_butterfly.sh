#!/usr/bin/env bash
set -e

die() { echo "ERROR: $*" 1>&2; exit 1; }

# --- config (override via env vars) ---
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

# PCC is vendored in your repo
PCC_DIR="${PCC_DIR:-third_party/flux_pcc}"

# PRIMME is required for *your* fork
PRIMME_ROOT="${PRIMME_ROOT:-$HOME/nobackup/primme}"

pick_libdir() {
  local p="$1"
  if   [ -d "$p/lib64" ]; then echo "$p/lib64"
  elif [ -d "$p/lib"   ]; then echo "$p/lib"
  else die "No lib/ or lib64/ under $p"
  fi
}

# --- spack env ---
[ -f "$SPACK_SETUP" ] || die "Missing SPACK_SETUP=$SPACK_SETUP"
# shellcheck disable=SC1090
. "$SPACK_SETUP"
spack env activate "$SPACK_ENV_NAME" >/dev/null

# ensure build tools present (no hashes)
spack load ninja meson pkgconf >/dev/null 2>&1 || true

# --- venv ---
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel
python -m pip install -U numpy cython scipy

# --- source sanity ---
[ -d "$SRC_DIR/include" ] || die "Not a butterfly repo? missing: $SRC_DIR/include"
[ -d "$SRC_DIR/$PCC_DIR" ] || die "PCC_DIR not found: $SRC_DIR/$PCC_DIR"

INC="$SRC_DIR/include"

# --- discover deps like old script did ---
SS="$(spack location -i suite-sparse)" || die "spack env missing suite-sparse"
OB="$(spack location -i openblas)"     || die "spack env missing openblas"
LAP="$(spack location -i netlib-lapack 2>/dev/null || true)"  # optional; old script used it
SS_LIB="$(pick_libdir "$SS")"
OB_LIB="$(pick_libdir "$OB")"

# flexiblas optional but your meson.build on linux uses it via pkg-config
FLEX="$(spack location -i flexiblas 2>/dev/null || true)"
FLEX_LIB=""
if [ -n "$FLEX" ] && [ -d "$FLEX" ]; then
  FLEX_LIB="$(pick_libdir "$FLEX")"
fi

# embree optional
EMB=""
EMB_LIB=""
if [ "$WITH_EMBREE" = "1" ]; then
  EMB="$(spack location -i embree 2>/dev/null || true)"
  [ -n "$EMB" ] || die "WITH_EMBREE=1 but embree not in spack env"
  EMB_LIB="$(pick_libdir "$EMB")"
fi

# primme required for your fork
[ -d "$PRIMME_ROOT" ] || die "PRIMME_ROOT does not exist: $PRIMME_ROOT"
PRIMME_LIB="$(pick_libdir "$PRIMME_ROOT")"

# venv lib dirs for runtime
VENV_LIB64="$VENV_DIR/lib64"
VENV_LIB="$VENV_DIR/lib"

# --- clean (optional) ---
if [ "$CLEAN" = "1" ]; then
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

# --- ss-compat tree (like old script) ---
SSCOMPAT="$BUILD_DIR/ss-compat/suitesparse"
mkdir -p "$SSCOMPAT"
ln -sf "$SS/include/cholmod.h" "$SSCOMPAT/cholmod.h"
ln -sf "$SS/include/umfpack.h" "$SSCOMPAT/umfpack.h"

# --- rpath flags (match old behavior) ---
RPATH_FLAGS=()
add_rpath_L() { RPATH_FLAGS+=("-L$1" "-Wl,-rpath,$1"); }

add_rpath_L "$SS_LIB"
[ -n "$FLEX_LIB" ] && add_rpath_L "$FLEX_LIB"
add_rpath_L "$OB_LIB"
[ -d "$VENV_LIB64" ] && add_rpath_L "$VENV_LIB64"
[ -d "$VENV_LIB" ]   && add_rpath_L "$VENV_LIB"
add_rpath_L "$PRIMME_LIB"
[ -n "$EMB_LIB" ] && add_rpath_L "$EMB_LIB"

# preserve the $ORIGIN relative rpaths from the old script
RPATH_FLAGS+=("-Wl,-rpath,\$ORIGIN/../../lib64" "-Wl,-rpath,\$ORIGIN/../../lib")

# --- include flags (match old behavior) ---
CFLAGS=("-DBF_DOUBLE"
        "-I$INC"
        "-I$SS/include"
        "-I$OB/include"
        "-I$BUILD_DIR/ss-compat")

# netlib-lapack include path if present (old script had it)
if [ -n "$LAP" ] && [ -d "$LAP/include" ]; then
  CFLAGS+=("-I$LAP/include")
fi

# force include like old script
CFLAGS+=("-include" "$INC/bf/blas.h")

# --- meson options ---
MESON_ARGS=(
  "--prefix" "$VENV_DIR"
  "-Dbuildtype=release"
  "-Dpcc_dir=$PCC_DIR"
  "-Dprimme=enabled"
  "-Dprimme_root=$PRIMME_ROOT"
)

if [ "$WITH_EMBREE" = "1" ]; then MESON_ARGS+=("-Dembree=enabled"); else MESON_ARGS+=("-Dembree=disabled"); fi
if [ "$WITH_PYTHON" = "1" ]; then MESON_ARGS+=("-Dpython=enabled"); else MESON_ARGS+=("-Dpython=disabled"); fi

# Configure only if needed (or after CLEAN=1)
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
  meson setup "$BUILD_DIR" "${MESON_ARGS[@]}" \
    --native-file <(cat <<EOF
[binaries]
python = '$(command -v python)'
EOF
) \
    -Dc_args="${CFLAGS[*]}" \
    -Dc_link_args="${RPATH_FLAGS[*]}"
else
  # Keep options in sync for incremental rebuilds
  meson configure "$BUILD_DIR" "${MESON_ARGS[@]}" >/dev/null
fi

ninja -C "$BUILD_DIR" -v
meson install -C "$BUILD_DIR"

python -c "import butterfly; print('butterfly OK')"
if [ "$WITH_PYTHON" = "1" ]; then
  python -c "import thermal_bf; print('thermal_bf OK')"
fi


# # incremental rebuild (default)
  #bash scripts/build_install_butterfly.sh
  #
  ## full clean rebuild
  #CLEAN=1 bash scripts/build_install_butterfly.sh
  #
  ## disable embree
  #WITH_EMBREE=0 bash scripts/build_install_butterfly.sh