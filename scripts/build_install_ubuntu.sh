#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" 1>&2; exit 1; }

# -------------------------
# Config (override via env)
# -------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SRC_DIR:-$REPO_ROOT}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/butterfly}"

WITH_EMBREE="${WITH_EMBREE:-1}"
WITH_PYTHON="${WITH_PYTHON:-1}"

CLEAN="${CLEAN:-0}"          # 0 incremental, 1 wipe build dir
SKIP_PIP="${SKIP_PIP:-0}"    # 0 install/upgrade numpy/cython/scipy, 1 only if missing

PCC_DIR="${PCC_DIR:-third_party/flux_pcc}"

PRIMME_ROOT="${PRIMME_ROOT:-$HOME/opt/primme}"

APT_INSTALL="${APT_INSTALL:-1}"

BUILD_PRIMME_IF_MISSING="${BUILD_PRIMME_IF_MISSING:-0}"
PRIMME_GIT_URL="${PRIMME_GIT_URL:-https://github.com/primme/primme.git}"
PRIMME_GIT_REF="${PRIMME_GIT_REF:-v3.2.1}"

# If 1, after install we symlink PRIMME .so into the venv multiarch libdir.
# This avoids needing LD_LIBRARY_PATH and fixes "libprimme.so.X not found".
VENV_SYMLINK_PRIMME="${VENV_SYMLINK_PRIMME:-1}"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

pick_libdir() {
  local p="$1"
  if   [ -d "$p/lib64" ]; then echo "$p/lib64"
  elif [ -d "$p/lib"   ]; then echo "$p/lib"
  else die "No lib/ or lib64/ under $p"
  fi
}

# -------------------------
# Install system deps (apt)
# -------------------------
if [ "$APT_INSTALL" = "1" ]; then
  have_cmd sudo || die "APT_INSTALL=1 but sudo not found"
  sudo -n true >/dev/null 2>&1 || echo "NOTE: sudo may prompt for your password."

  sudo apt-get update

  sudo apt-get install -y \
    build-essential \
    pkg-config \
    meson \
    ninja-build \
    cmake \
    git \
    python3-venv \
    python3-dev \
    python3-pip

  sudo apt-get install -y \
    libgsl-dev \
    libsuitesparse-dev \
    libopenblas-dev \
    liblapack-dev \
    liblapacke-dev

  # flexiblas optional; naming/availability varies
  if apt-cache show libflexiblas-dev >/dev/null 2>&1; then
    sudo apt-get install -y libflexiblas-dev
  elif apt-cache show flexiblas-dev >/dev/null 2>&1; then
    sudo apt-get install -y flexiblas-dev
  elif apt-cache show libflexiblas3 >/dev/null 2>&1; then
    sudo apt-get install -y libflexiblas3
  else
    echo "NOTE: FlexiBLAS not available in apt repos; continuing without it."
  fi

  if [ "$WITH_EMBREE" = "1" ]; then
    sudo apt-get install -y libembree-dev || die "WITH_EMBREE=1 but libembree-dev not available/failed to install"
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
  python -m pip install -U numpy cython scipy matplotlib >/dev/null
else
  python - <<'PY' >/dev/null 2>&1 || python -m pip install -U numpy cython scipy matplotlib >/dev/null
import numpy, Cython, scipy, matplotlib
PY
fi

# -------------------------
# Source sanity
# -------------------------
[ -d "$SRC_DIR/include" ] || die "Not a butterfly repo? missing: $SRC_DIR/include"
[ -d "$SRC_DIR/$PCC_DIR" ] || die "PCC_DIR not found: $SRC_DIR/$PCC_DIR"
INC="$SRC_DIR/include"

# -------------------------
# PRIMME (required)
# -------------------------
if [ ! -d "$PRIMME_ROOT" ] && [ "$BUILD_PRIMME_IF_MISSING" = "1" ]; then
  mkdir -p "$(dirname "$PRIMME_ROOT")"
  git clone "$PRIMME_GIT_URL" "$PRIMME_ROOT"
  (cd "$PRIMME_ROOT" && git checkout "$PRIMME_GIT_REF" || true)

  cmake -S "$PRIMME_ROOT" -B "$PRIMME_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PRIMME_ROOT" \
    -DBUILD_SHARED_LIBS=ON
  cmake --build "$PRIMME_ROOT/build" -j"$(nproc)"
  cmake --install "$PRIMME_ROOT/build"
fi

[ -d "$PRIMME_ROOT/include" ] || die "PRIMME_ROOT missing include/: $PRIMME_ROOT"

# Ensure PRIMME has lib/ (Meson option expects it)
if [ ! -d "$PRIMME_ROOT/lib" ] && [ -d "$PRIMME_ROOT/lib64" ]; then
  ln -snf "$PRIMME_ROOT/lib64" "$PRIMME_ROOT/lib"
fi
[ -d "$PRIMME_ROOT/lib" ] || die "PRIMME_ROOT missing lib/: $PRIMME_ROOT"

# Create unversioned linker name if only versioned .so exists (helps linkers/tools)
if [ -f "$PRIMME_ROOT/lib/libprimme.so.3" ] && [ ! -f "$PRIMME_ROOT/lib/libprimme.so" ]; then
  ln -snf "libprimme.so.3" "$PRIMME_ROOT/lib/libprimme.so"
elif ls "$PRIMME_ROOT/lib"/libprimme.so.* >/dev/null 2>&1 && [ ! -f "$PRIMME_ROOT/lib/libprimme.so" ]; then
  ln -snf "$(basename "$(ls -1 "$PRIMME_ROOT/lib"/libprimme.so.* | head -n1)")" "$PRIMME_ROOT/lib/libprimme.so"
fi

PRIMME_LIB="$(pick_libdir "$PRIMME_ROOT")"

# -------------------------
# venv lib dirs (Ubuntu multiarch!)
# -------------------------
VENV_LIB="$VENV_DIR/lib"
VENV_LIB64="$VENV_DIR/lib64"
VENV_MULTIARCH_LIB="$VENV_DIR/lib/$(gcc -dumpmachine 2>/dev/null || true)"

# -------------------------
# Build dir handling
# -------------------------
if [ "$CLEAN" = "1" ]; then
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

# -------------------------
# CFLAGS
# -------------------------
CFLAGS=(
  "-DBF_DOUBLE"
  "-I$INC"
  "-include" "$INC/bf/blas.h"
)
if [ -d /usr/include/suitesparse ]; then
  CFLAGS+=("-I/usr/include/suitesparse")
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
# Link args: force RUNPATH for both PRIMME + venv multiarch
# (this matches what you did manually with `meson configure ... -Dc_link_args=...`)
# -------------------------
LINK_ARGS=()
LINK_ARGS+=("-L$PRIMME_LIB" "-Wl,-rpath,$PRIMME_LIB")

# Multiarch is where Meson installs libbutterfly.so on Ubuntu/Mint
if [ -n "${VENV_MULTIARCH_LIB:-}" ]; then
  mkdir -p "$VENV_MULTIARCH_LIB"
  LINK_ARGS+=("-Wl,-rpath,$VENV_MULTIARCH_LIB")
fi

# Also keep older assumptions (harmless; helps on non-multiarch installs)
[ -d "$VENV_LIB" ]   && LINK_ARGS+=("-Wl,-rpath,$VENV_LIB")
[ -d "$VENV_LIB64" ] && LINK_ARGS+=("-Wl,-rpath,$VENV_LIB64")

# Preserve $ORIGIN relative rpaths from your old script
LINK_ARGS+=("-Wl,-rpath,\$ORIGIN/../../lib64" "-Wl,-rpath,\$ORIGIN/../../lib")

# -------------------------
# Configure/build/install
# Always run meson configure so link args updates take effect
# -------------------------
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
  meson setup "$BUILD_DIR" "${MESON_ARGS[@]}" \
    --native-file <(cat <<EOF
[binaries]
python = '$(command -v python3)'
EOF
) \
    -Dc_args="${CFLAGS[*]}" \
    -Dc_link_args="${LINK_ARGS[*]}"
else
  meson configure "$BUILD_DIR" "${MESON_ARGS[@]}" \
    -Dc_args="${CFLAGS[*]}" \
    -Dc_link_args="${LINK_ARGS[*]}" >/dev/null
fi

ninja -C "$BUILD_DIR" -v
meson install -C "$BUILD_DIR"

# -------------------------
# Make PRIMME discoverable at runtime without env vars:
# symlink into the same multiarch dir where libbutterfly.so lives
# -------------------------
if [ "$VENV_SYMLINK_PRIMME" = "1" ]; then
  if [ -z "${VENV_MULTIARCH_LIB:-}" ]; then
    die "VENV_MULTIARCH_LIB empty; cannot symlink primme into venv"
  fi
  mkdir -p "$VENV_MULTIARCH_LIB"

  # Symlink SONAME(s) and linker name if present
  for f in "$PRIMME_LIB"/libprimme.so.*; do
    [ -e "$f" ] || continue
    ln -snf "$f" "$VENV_MULTIARCH_LIB/$(basename "$f")"
  done
  [ -e "$PRIMME_LIB/libprimme.so" ] && ln -snf "$PRIMME_LIB/libprimme.so" "$VENV_MULTIARCH_LIB/libprimme.so"
fi

python -c "import butterfly; print('butterfly OK')"
if [ "$WITH_PYTHON" = "1" ]; then
  python -c "import thermal_bf; print('thermal_bf OK')"
fi
