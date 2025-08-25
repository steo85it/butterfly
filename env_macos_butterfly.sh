# env_macos_butterfly.sh
# Usage: source ./env_macos_butterfly.sh
## after ##
# brew update
# brew install arpack suite-sparse openblas embree tbb gcc meson ninja pkg-config

# Homebrew prefixes
export AR_PREFIX="$(brew --prefix arpack)"
export SS_PREFIX="$(brew --prefix suite-sparse)"
export OB_PREFIX="$(brew --prefix openblas)"
export EM_PREFIX="$(brew --prefix embree)"
export TBB_PREFIX="$(brew --prefix tbb)"
export GCC_LIBDIR="$(brew --prefix gcc)/lib/gcc/current"

# Header search path
export CPATH="$SS_PREFIX/include:$AR_PREFIX/include:$CPATH"

# Library search paths for compile/link
export LIBRARY_PATH="$SS_PREFIX/lib:$AR_PREFIX/lib:$OB_PREFIX/lib:$GCC_LIBDIR:$LIBRARY_PATH"
export LDFLAGS="-L$SS_PREFIX/lib -L$AR_PREFIX/lib -L$OB_PREFIX/lib -L$GCC_LIBDIR ${LDFLAGS}"

# Where Python will find the .dylibs at runtime
export DYLD_LIBRARY_PATH="$SS_PREFIX/lib:$AR_PREFIX/lib:$OB_PREFIX/lib:$GCC_LIBDIR:$DYLD_LIBRARY_PATH"

# Helpful for Meson’s pkg-config lookups (usually not needed, but safe)
export PKG_CONFIG_PATH="$AR_PREFIX/lib/pkgconfig:$SS_PREFIX/lib/pkgconfig:$OB_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# additional stuff
# Set prefixes
export SS_PREFIX="$(brew --prefix suite-sparse)"
export OB_PREFIX="$(brew --prefix openblas)"
export GCC_LIBDIR="$(brew --prefix gcc)/lib/gcc/current"

# Add header search paths so clang finds umfpack.h and cblas.h
export CPATH="$SS_PREFIX/include:$SS_PREFIX/include/suitesparse:$OB_PREFIX/include:$CPATH"

# (Keep the link/runtime paths you already had)
export LIBRARY_PATH="$SS_PREFIX/lib:$OB_PREFIX/lib:$GCC_LIBDIR:$LIBRARY_PATH"
export LDFLAGS="-L$SS_PREFIX/lib -L$OB_PREFIX/lib -L$GCC_LIBDIR $LDFLAGS"
export DYLD_LIBRARY_PATH="$SS_PREFIX/lib:$OB_PREFIX/lib:$GCC_LIBDIR:$DYLD_LIBRARY_PATH"
