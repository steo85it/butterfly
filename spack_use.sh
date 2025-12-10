# (from butterfly source)
source ~/nobackup/venvs/butterfly/bin/activate
. /panfs/ccds02/nobackup/people/sberton2/.spack_repo/share/spack/setup-env.sh
spack env activate butterfly
spack load meson ninja pkgconf suite-sparse openblas flexiblas arpack-ng gsl

SS=$(spack location -i suite-sparse);  for d in "$SS/lib64" "$SS/lib"; do [ -d "$d" ] && SSLIB="$d" && break; done
OB=$(spack location -i openblas);      for d in "$OB/lib64" "$OB/lib"; do [ -d "$d" ] && OBLIB="$d" && break; done
FX=$(spack location -i flexiblas);     for d in "$FX/lib64" "$FX/lib"; do [ -d "$d" ] && FLEXLIB="$d" && break; done
AR=$(spack location -i arpack-ng);     for d in "$AR/lib64" "$AR/lib"; do [ -d "$d" ] && ARLIB="$d" && break; done
GS=$(spack location -i gsl);           for d in "$GS/lib64" "$GS/lib"; do [ -d "$d" ] && GSLLIB="$d" && break; done

# PRIMME (built in-tree under primme/)
PRIMME_ROOT="$PWD/../primme"

# headers (adjust if you copied them into include/ already)
PRIMME_INC="$PRIMME_ROOT/include"

# prefer lib64 if present, otherwise lib
for d in "$PRIMME_ROOT/lib64" "$PRIMME_ROOT/lib"; do
  [ -d "$d" ] && PRIMME_LIB="$d" && break
done

INC="$PWD/include"
rm -rf build build/ss-compat
mkdir -p build/ss-compat/suitesparse
ln -sf "$SS/include/cholmod.h"  build/ss-compat/suitesparse/cholmod.h
ln -sf "$SS/include/umfpack.h"  build/ss-compat/suitesparse/umfpack.h

meson setup build \
  --prefix "$VIRTUAL_ENV" -Dbuildtype=release -Dembree=enabled -Dpython=enabled \
  -Dc_args="-DBF_DOUBLE -I$INC -I$PRIMME_INC -I$SS/include -I$OB/include -I$PWD/build/ss-compat -include $INC/bf/blas.h" \
  -Dc_link_args="\
    -L$SSLIB   -Wl,-rpath,$SSLIB \
    -L$OBLIB   -Wl,-rpath,$OBLIB \
    -L$FLEXLIB -Wl,-rpath,$FLEXLIB \
    -L$ARLIB   -Wl,-rpath,$ARLIB \
    -L$GSLLIB  -Wl,-rpath,$GSLLIB \
    -L$PRIMME_LIB -Wl,-rpath,$PRIMME_LIB -lprimme \
    -lopenblas \
    -L$VIRTUAL_ENV/lib64 -Wl,-rpath,$VIRTUAL_ENV/lib64 \
    -L$VIRTUAL_ENV/lib   -Wl,-rpath,$VIRTUAL_ENV/lib \
    -Wl,-rpath,'\$ORIGIN/../../lib64' -Wl,-rpath,'\$ORIGIN/../../lib'"

ninja -C build -v
meson install -C build

# Now this should work with just the venv:
deactivate; source ~/nobackup/venvs/butterfly/bin/activate
python -c "import butterfly; print('butterfly OK')"
