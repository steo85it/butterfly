# butterfly

*Overview to come...*

## Documentation

Woefully incomplete documentation for butterfly is available [here](https://sampotter.github.io/butterfly).

Additionally:

1. Check out the [examples](./examples) directory to see demonstrations of the library.
2. If nothing fits, contact [me](https://sampotter.github.io) by email, or create a GitHub issue. I'm happy to work with you to make this library suit your needs.

## Compilation

### Generic build (Meson)

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

### HPC build (Spack + Python venv)

On our HPC systems we typically manage native dependencies with **Spack** and install the Python extensions into a **Python virtualenv**.

#### 1) Spack environment (one-time)

If this repo includes `spack.yaml` / `spack.lock` (recommended), you can reproduce the dependency environment like this:

```bash
. $HOME/nobackup/.spack_repo/share/spack/setup-env.sh
spack env activate butterfly
spack concretize -f
spack install
```

Notes:

* `spack.yaml` describes the dependency intent (packages/variants).
* `spack.lock` pins a concretized solution for a given platform/compiler stack. On the same HPC (same OS/arch/compiler), using the lock file helps colleagues get the same dependency DAG.

#### 2) Build + install butterfly into a venv (repeatable)

Use the repo script:

```bash
. $HOME/nobackup/.spack_repo/share/spack/setup-env.sh
spack env activate butterfly
bash scripts/build_install_butterfly.sh
```

By default this does an incremental rebuild (keeps `build/`). To force a clean reconfigure/rebuild:

```bash
CLEAN=1 bash scripts/build_install_butterfly.sh
```

The script installs into:

* `VENV_DIR=$HOME/nobackup/venvs/butterfly` (override via env var)

and verifies imports:

```bash
python -c "import butterfly; print('butterfly OK')"
python -c "import thermal_bf; print('thermal_bf OK')"
```

Common overrides:

```bash
# disable embree features
WITH_EMBREE=0 bash scripts/build_install_butterfly.sh

# build C library only (no Python extensions)
WITH_PYTHON=0 bash scripts/build_install_butterfly.sh

# set a non-default primme location
PRIMME_ROOT=/path/to/primme bash scripts/build_install_butterfly.sh
```

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
