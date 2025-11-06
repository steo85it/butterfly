# If Spack isn’t on your PATH, source it first
# . /path/to/spack/share/spack/setup-env.sh

#spack env create butterfly
. /panfs/ccds02/nobackup/people/sberton2/.spack_repo/share/spack/setup-env.sh
spack env activate butterfly

# Core toolchain
spack add cmake ninja meson pkgconf

# Math stack
spack add openblas  # (or intel-oneapi-mkl if you prefer)
spack add suite-sparse   # (CHOLMOD/UMFPACK come from here)
spack add arpack-ng +icb
spack add gsl

# Geometry & ray tracing
#spack add onetbb         # Embree’s threading backend
#spack add embree@4:

# Optional (tests / CLI parsing)
#spack add argtable3
spack add cmocka

# Python bits (build the wrapper inside your venv, but these help if needed)
spack add py-numpy py-cython py-pip py-setuptools py-scipy

# Optional: OpenMP (usually provided by compiler, but make sure we can link it)
#spack add llvm-openmp  # if using clang; for GCC you don’t need this

spack concretize -f
spack install


# when installed
. /panfs/ccds02/nobackup/people/sberton2/.spack_repo/share/spack/setup-env.sh
spack env activate butterfly

#spack load openblas suite-sparse arpack-ng gsl
#spack load embree
#spack load py-numpy py-cython
#spack load /rbonybh ninja meson pkgconf
#spack load suite-sparse openblas arpack-ng gsl embree
spack load /k5oyloa /rbonybh /yz4faop /kg47p7y /mpuudue py-cython ninja meson suite-sparse arpack-ng embree

python3 -m venv ~/nobackup/venvs/butterfly
source ~/nobackup/venvs/butterfly/bin/activate
pip install --upgrade pip wheel
pip install numpy cython scipy  # use venv’s NumPy for headers
pip install matplotlib cached_property

# add deps
cd ~/nobackup/illumrad/python-flux
pip install -e .

source ~/nobackup/illumrad/embree-3.12.1.x86_64.linux/embree-vars.sh
cd ~/nobackup/illumrad/python-embree
pip install .

# (1) Load everything you need:
# if you use clang+OpenMP, also: spack load llvm-openmp

# 0) Vars
INC="$PWD/include"
SS=$(spack location -i suite-sparse)
LAP=$(spack location -i netlib-lapack)
OB=$(spack location -i openblas)
for d in "$SS/lib64" "$SS/lib"; do [ -d "$d" ] && LIBDIR="$d" && break; done

# 1) Clean
ninja -C build -t clean || true
rm -rf build

# 2) Create a tiny compat include tree that mimics "suitesparse/<hdr>.h"
mkdir -p build/ss-compat/suitesparse
ln -sf "$SS/include/cholmod.h"  build/ss-compat/suitesparse/cholmod.h
ln -sf "$SS/include/umfpack.h"  build/ss-compat/suitesparse/umfpack.h

# Discover FlexiBLAS + OpenBLAS lib dirs
FLEX=$(spack location -i flexiblas 2>/dev/null || true)
for d in "$FLEX/lib64" "$FLEX/lib"; do [ -d "$d" ] && FLEXLIB="$d" && break; done

OB=$(spack location -i openblas)
for d in "$OB/lib64" "$OB/lib"; do [ -d "$d" ] && OBLIB="$d" && break; done

# Discover venv lib dirs (both, some distros use lib not lib64)
VENV_LIB64="$VIRTUAL_ENV/lib64"
[ -d "$VIRTUAL_ENV/lib" ] && VENV_LIB="$VIRTUAL_ENV/lib" || VENV_LIB=""

meson setup build \
  --prefix "$VIRTUAL_ENV" \
  -Dbuildtype=release \
  -Dembree=enabled \
  -Dpython=enabled \
  -Dc_args="-DBF_DOUBLE -I$INC -I$SS/include -I$OB/include -I$LAP/include -I$PWD/build/ss-compat -include $INC/bf/blas.h" \
  -Dc_link_args="\
      -L$LIBDIR   -Wl,-rpath,$LIBDIR \
      -L$FLEXLIB  -Wl,-rpath,$FLEXLIB \
      -L$OBLIB    -Wl,-rpath,$OBLIB \
      ${VENV_LIB64:+-L$VENV_LIB64 -Wl,-rpath,$VENV_LIB64} \
      ${VENV_LIB:+-L$VENV_LIB   -Wl,-rpath,$VENV_LIB} \
      -Wl,-rpath,'\$ORIGIN/../../lib64' \
      -Wl,-rpath,'\$ORIGIN/../../lib'"

ninja -C build -v
meson install -C build

# this should now work with both active
. /panfs/ccds02/nobackup/people/sberton2/.spack_repo/share/spack/setup-env.sh
spack env activate butterfly # (needed for blas, gsl, etc)
source ~/nobackup/illumrad/embree-3.12.1.x86_64.linux/embree-vars.sh
source ~/nobackup/venvs/butterfly/bin/activate # (needed to import butterfly)
python -c "import butterfly; print('butterfly OK')"

