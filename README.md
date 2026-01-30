# butterfly

*Overview to come...*

## Quick start (Python extension)

If you want the Python modules (`butterfly`, `thermal_bf`) on a Linux/HPC system, use:

    bash scripts/build_install_butterfly.sh

The script can (optionally) activate a Spack env, create/use a venv, build with Meson+Ninja,
install into the venv prefix, and verify imports.

## Documentation

Woefully incomplete documentation for butterfly is available [here](https://sampotter.github.io/butterfly).

Additionally:

1. Check out the [examples](./examples) directory to see demonstrations of the library.
2. If nothing fits, contact [me](https://sampotter.github.io) by email, or create a GitHub issue. I'm happy to work with you to make this library suit your needs.

## Compilation

### Generic build (Meson only)

Use [Meson](https://mesonbuild.com/) to compile this library:

```bash
meson setup builddir
cd builddir
meson compile
```

This will build all of the examples, as well. Afterwards, the compiled executables for the examples will be in `./builddir/examples`.

For Mac machines using homebrew, it may be necessary to specify paths to dependencies by passing the `-Dc_args` flag to meson, e.g.

```bash
meson setup builddir -Dc_args='-I/opt/homebrew/Cellar/suite-sparse/7.1.0/include/ -I/opt/homebrew/Cellar/openblas/0.3.24/include/'
```

### HPC / Linux build (Spack + Python venv) — recommended

On Linux/HPC we typically manage native dependencies with Spack and install the Python
extensions into a Python virtualenv. The recommended entry point is:

    bash scripts/build_install_butterfly.sh

The script can (optionally):

* load a compiler module (COMPILER_MODULE=...)
* activate a Spack env (USE_SPACK=1, SPACK_SETUP=..., SPACK_ENV_NAME=...)
* create/use a Python venv (VENV_DIR=...)
* build with Meson+Ninja and install into the venv prefix
* verify imports: butterfly and (if enabled) thermal_bf

#### 1) Spack environment (one-time)

If this repo includes spack.yaml / spack.lock (recommended), you can reproduce the dependency
environment like this:

    . $HOME/nobackup/.spack_repo/share/spack/setup-env.sh
    spack env activate butterfly
    spack concretize -f
    spack install

Notes:

* `spack.yaml` describes the dependency intent (packages/variants).
* `spack.lock` pins a concretized solution for a given platform/compiler stack. On the same HPC (same OS/arch/compiler), using the lock file helps colleagues get the same dependency DAG.

If the Spack environment does not exist yet on a given system, you can create/provision it using:

    bash scripts/spack_bootstrap_env.sh

This is a one-time step that:
* creates the Spack env (default name: butterfly) if missing
* adds the dependency set (no hardcoded hashes)
* concretizes + installs
* leaves spack.yaml + spack.lock in the env directory

Defaults: WITH_EMBREE=1, WITH_PYTHON=1, WITH_LAPACKE=0, WITH_FLEXIBLAS=0.

Common overrides:

    COMPILER_MODULE=gcc/12.1.0 SPACK_ENV_NAME=butterfly bash scripts/spack_bootstrap_env.sh

The bootstrap script does NOT build butterfly; after it finishes, build/install with:

    bash scripts/build_install_butterfly.sh

#### 2) Build + install into a venv (repeatable)

    . $HOME/nobackup/.spack_repo/share/spack/setup-env.sh
    spack env activate butterfly
    bash scripts/build_install_butterfly.sh

By default this does an incremental rebuild (keeps build/). To force a clean reconfigure/rebuild:

    CLEAN=1 bash scripts/build_install_butterfly.sh

The script installs into:

* VENV_DIR=$HOME/nobackup/venvs/butterfly (override via env var)

and verifies imports:

    python -c "import butterfly; print('butterfly OK')"
    python -c "import thermal_bf; print('thermal_bf OK')"

#### 3) PGDA / local HPC example (validated)

Adapt paths to your account:

    git clone -b develop /home/sberton2/git/butterfly.git
    cd butterfly

    COMPILER_MODULE=gcc/12.1.0 \
    USE_SPACK=1 \
    SPACK_SETUP="/home/sberton2/nobackup/.spack_repo/share/spack/setup-env.sh" \
    SPACK_ENV_NAME=butterfly \
    PRIMME_ROOT="/home/sberton2/nobackup/primme" \
    WITH_EMBREE=1 \
    WITH_PYTHON=1 \
    CLEAN=1 \
    bash scripts/build_install_butterfly.sh

For subsequent incremental rebuilds:

    COMPILER_MODULE=gcc/12.1.0 USE_SPACK=1 CLEAN=0 bash scripts/build_install_butterfly.sh

#### 4) Script configuration reference (env vars)

Spack:

* USE_SPACK=1 (default): activate/load from Spack
* SPACK_SETUP=/path/to/spack/share/spack/setup-env.sh
* SPACK_ENV_NAME=butterfly

Compiler module (HPC):

* COMPILER_MODULE=gcc/12.1.0 (optional)

Build/install locations:

* BUILD_DIR=$PWD/build (default)
* VENV_DIR=$HOME/nobackup/venvs/butterfly (default)

Features:

* WITH_PYTHON=1 (default): build/install Python extensions
* WITH_EMBREE=1 (default): enable Embree integration when available

Rebuild behavior:

* CLEAN=0 (default): incremental rebuild
* CLEAN=1: wipe build dir and reconfigure

Python deps:

* SKIP_PIP=0 (default): ensure numpy/cython/scipy are installed in the venv
* SKIP_PIP=1: skip pip unless missing

PRIMME (required for this fork):

* PRIMME_ROOT=/path/to/primme (required)

The script expects:

* $PRIMME_ROOT/include exists
* $PRIMME_ROOT/lib exists (if you only have lib64/, the script creates lib -> lib64)
* if PRIMME only provides a versioned shared object (e.g. libprimme.so.3) without an
  unversioned linker name, the script will create the libprimme.so symlink.

#### 5) Common overrides / recipes

    # disable embree features
    WITH_EMBREE=0 bash scripts/build_install_butterfly.sh

    # build C library only (no Python extensions)
    WITH_PYTHON=0 bash scripts/build_install_butterfly.sh

    # set a non-default primme location
    PRIMME_ROOT=/path/to/primme bash scripts/build_install_butterfly.sh

## Error handling

This library features "OpenGL-style" error handling (e.g., [see this page](https://www.khronos.org/opengl/wiki/OpenGL_Error)). The guiding principles are three-fold:

1. It should allow you to pinpoint exactly where runtime user error occurs, useful for debugging the library.
2. If an error occurs when calling a function, the function should be a no-op; control should procede normally afterwards (modulo the downstream effects of the error!).
3. It exists "in parallel" with the rest of the code, so that the error handling system can be completely stripped from a build if desired.

As this library is still in development, these principles are not fully realized yet, but this is the goal.

See [Intel Embree](https://www.embree.org/) for another example of a library with this style of error handling.

## Using Emacs lsp-mode

Emacs's [lsp-mode](https://emacs-lsp.github.io/lsp-mode/tutorials/CPP-guide/) can be used to add some IDE-like features to Emacs. For this to work with something like [clangd](https://clangd.llvm.org/) (recommended), clangd needs to be able to find a `compile_commands.json` file which describes the build. Meson generates this automatically, but it stores it in the build directory (e.g., `builddir` above). When opening a project file for the first time, lsp-mode will ask for the location of the "project root". Make sure that this directory contains a copy of the most recent `compile_commands.json` file.

## The hierarchical matrix types

This library includes a set of types for modeling recursively composed hierarchical matrices. These types support runtime polymorphism implemented using macros defined in [interface.h](./include/bf/interface.h). The root "class" in the hierarchy is [BfMat](./include/bf/mat.h).

## Troubleshooting

### Problems with OpenBLAS and GCC 12

See the following GitHub issues:

* [https://github.com/msys2/MINGW-packages/issues/12857](https://github.com/msys2/MINGW-packages/issues/12857)
* [https://github.com/xianyi/OpenBLAS/issues/3740](https://github.com/xianyi/OpenBLAS/issues/3740)
* [https://github.com/xianyi/OpenBLAS/issues/4013](https://github.com/xianyi/OpenBLAS/issues/4013)

A bug in GCC 12's optimizer causes `zgemv` to segfault. This problem will arise if you ask butterfly to compute a complex SVD while running GCC 12 and OpenBLAS.

One fix is to use [flexiblas](https://github.com/mpimd-csc/flexiblas) and select a different backend. Alternatively, use a different version of GCC.
