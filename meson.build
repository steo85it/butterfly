project(
  'butterfly',
  'c', 'cpp', 'cython',
  default_options : ['c_std=c11', 'cpp_std=c++11'],
  meson_version : '>= 1.2.2')

fs = import('fs')
pkg = import('pkgconfig')

message(get_option('python').enabled())

if get_option('buildtype') == 'debug'
  add_project_arguments('-DBF_DEBUG', language : 'c')
endif
if get_option('embree').enabled()
  add_project_arguments('-DBF_EMBREE', language : 'c')
endif
if get_option('mfem').enabled()
  add_project_arguments('-DBF_MFEM', language : 'c')
endif
if get_option('python').enabled()
  add_project_arguments('-DBF_PYTHON', language : 'c')
endif
add_project_arguments('-DBF_DOUBLE', language : 'c')

host_system = host_machine.system()
if host_system == 'linux'
  add_project_arguments('-DBF_LINUX', language : 'c')
elif host_system == 'darwin'
  add_project_arguments('-DBF_DARWIN', language : 'c')
else
  error('unsupported platform:', host_system)
endif

cc = meson.get_compiler('c')

libm = cc.find_library('m', required : true)
fortran_dep = cc.find_library('gfortran', required : true)
cholmod_dep = cc.find_library('cholmod', required : true)
m_dep = cc.find_library('m', required : false)
umfpack_dep = cc.find_library('umfpack', required : true)
arpack_dep = dependency('arpack', required : false)
gsl_dep = dependency('gsl', required : true)

if get_option('embree').enabled()
  embree_dep = dependency('embree', required : true)
endif

if get_option('python').enabled()
  python3 = import('python').find_installation('python3')
  python3_dep = python3.dependency(embed : true)

  ############################################################################
  # NOTE: the following blob is used to set up numpy_dep
  # below. Following this GitHub issue:
  #
  #   https://github.com/mesonbuild/meson/issues/9598
  #
  # it was copied from the SciPy source and modified:
  #
  #   https://github.com/scipy/scipy/blob/main/scipy/meson.build#L30-L73
  #
  # Reason being that Meson doesn't yet support something like:
  #
  #   numpy_dep = dependency('numpy')
  #
  # TODO: keep an eye on this GitHub issue and replace with something
  # simpler once this feature is finished.
  numpy_incdir = meson.get_external_property('numpy-include-dir', 'not-given')
  if numpy_incdir == 'not-given'
    numpy_incdir = run_command(python3,
                               [
                                 '-c',
                                 'import numpy as np; print(np.get_include())'
                               ],
                               check: true
                              ).stdout().strip()
    # We do need an absolute path to feed to `cc.find_library` below
    _numpy_incdir_abs = run_command(python3,
                                    ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
                                    check: true
                                   ).stdout().strip()
  else
    _numpy_incdir_abs = numpy_incdir
  endif
  numpy_inc = include_directories(numpy_incdir)
  numpy_dep = declare_dependency(include_directories: numpy_inc)
endif

butterfly_dependencies = [
  libm,
  fortran_dep,
  cholmod_dep,
  m_dep,
  # arpack_dep,
  gsl_dep,
  umfpack_dep,
]

if arpack_dep.found()
  add_project_arguments('-DBF_HAVE_ARPACK', language : 'c')
  butterfly_dependencies += [arpack_dep]
else
  message('ARPACK not found — building without ARPACK-dependent code')
endif

if host_machine.system() == 'linux'
  blas_dep = dependency('flexiblas', required : true)
  lapacke_dep = dependency('lapacke', required : true)
  butterfly_dependencies += [blas_dep, lapacke_dep]
  add_project_arguments('-D_DEFAULT_SOURCE', language : 'c')
elif host_machine.system() == 'darwin'
  blas_dep = cc.find_library('openblas', required : true)
  butterfly_dependencies += [blas_dep]
  add_project_arguments('-D_DARWIN_C_SOURCE=1', language : 'c')
  add_project_arguments('-D__DARWIN_UNIX03=1', language : 'c')
elif host_machine.system() == 'windows'
  error('Windows unsupported currently')
endif
if get_option('embree').enabled()
  butterfly_dependencies += [embree_dep]
endif
if get_option('python').enabled()
  butterfly_dependencies += [python3_dep, numpy_dep]
endif

fmm2d_dep = dependency('fmm2d', required : false)
argtable3_dep = dependency('argtable3', required : false)

# OpenMP (optional but recommended)
openmp_dep = dependency('openmp', required : false)
if openmp_dep.found()
  add_project_arguments('-DBF_OPENMP', language : 'c')
  butterfly_dependencies += [openmp_dep]
endif

# --- PRIMME (optional, but REQUIRED when enabled) --------------------------
primme_dep = disabler()

if get_option('primme').enabled()
  primme_root = get_option('primme_root')

  if primme_root != ''
    primme_incdir = include_directories(join_paths(primme_root, 'include'))

    primme_libdir64 = join_paths(primme_root, 'lib64')
    primme_libdir   = join_paths(primme_root, 'lib')
    primme_libdirs  = fs.is_dir(primme_libdir64) ? [primme_libdir64] : [primme_libdir]

    primme_lib = cc.find_library(
      'primme',
      dirs : primme_libdirs,
      required : true
    )

    primme_dep = declare_dependency(
      include_directories : primme_incdir,
      dependencies        : [primme_lib]
    )
  else
    # Requires a primme.pc in PKG_CONFIG_PATH
    primme_dep = dependency('primme', required : true)
  endif

  message('PRIMME found — enabling primme_svds backend')
  add_project_arguments('-DBF_HAVE_PRIMME_SVDS', language : 'c')
  butterfly_dependencies += [primme_dep]
endif
# --------------------------------------------------------------------------

butterfly_lib_src = [
  'src/array.c',
  'src/bbox.c',
  'src/bf.c',
  'src/bessel.c',
  'src/cheb.c',
  'src/circle.c',
  'src/chol.c',
  'src/chol_csr_real.c',
  'src/cholmod.c',
  'src/disjoint_interval_list.c',
  'src/dtype.c',
  'src/ellipse.c',
  'src/error.c',
  'src/eval_tree.c',
  'src/fac.c',
  'src/fac_helm2.c',
  'src/fac_span.c',
  'src/fac_streamer.c',
  'src/fiedler_tree.c',
  'src/fiedler_tree_node.c',
  'src/helm2.c',
  'src/indexed_mat.c',
  'src/interval.c',
  'src/interval_tree.c',
  'src/interval_tree_node.c',
  'src/lbo.c',
  'src/linalg.c',
  'src/logging.c',
  'src/lu.c',
  'src/lu_csr_real.c',
  'src/lu_dense_complex.c',
  'src/mat.c',
  'src/mat_block.c',
  'src/mat_block_coo.c',
  'src/mat_block_dense.c',
  'src/mat_block_diag.c',
  'src/mat_coo_complex.c',
  'src/mat_coo_real.c',
  'src/mat_csr_real.c',
  'src/mat_dense.c',
  'src/mat_dense_complex.c',
  'src/mat_dense_real.c',
  'src/mat_diag_real.c',
  'src/mat_diff.c',
  'src/mat_func.c',
  'src/mat_givens.c',
  'src/mat_identity.c',
  'src/mat_perm.c',
  'src/mat_product.c',
  'src/mat_sum.c',
  'src/mat_vf_hier.c',
  'src/mat_zero.c',
  'src/mem.c',
  'src/node_array.c',
  'src/node_span.c',
  'src/octree.c',
  'src/octree.get_nearest_neighbors.c',
  'src/octree_node.c',
  'src/perm.c',
  'src/points.c',
  'src/poisson_disk_sampling.c',
  'src/ptr_array.c',
  'src/quadrature.c',
  'src/quadtree.c',
  'src/quadtree_node.c',
  'src/rand.c',
  'src/real_array.c',
  'src/size_array.c',
  'src/splitmix64.c',
  'src/timer.c',
  'src/tree.c',
  'src/tree_iter.c',
  'src/tree_iter_post_order.c',
  'src/tree_level_iter.c',
  'src/tree_node.c',
  'src/trimesh.c',
  'src/trimesh.get_level_set_submesh.c',
  'src/types.c',
  'src/util.c',
  'src/vec.c',
  'src/vec_complex.c',
  'src/vec_real.c',
  'src/vec_zero.c',
  'src/vectors.c',
  'src/vf_hier.c',
  'src/xoshiro256plus.c',
]
if get_option('python').enabled()
  butterfly_lib_src += ['src/mat_python.c']
endif

butterfly_inc = include_directories('include')

butterfly_include_directories = [
  butterfly_inc,
]

butterfly_lib = library(
  'butterfly',
  butterfly_lib_src,
  dependencies : butterfly_dependencies,
  include_directories : butterfly_include_directories,
  install : true
)

butterfly_dep = declare_dependency(
  link_with : butterfly_lib,
  include_directories : butterfly_include_directories,
  dependencies : openmp_dep.found() ? [openmp_dep] : []
)

install_subdir('include/bf', install_dir : 'include', install_tag : 'devel')
pkg.generate(butterfly_lib)

if get_option('python').enabled()
  # ---- PCC (vendored) ----
  pcc_dir = get_option('pcc_dir')
  pcc_root = join_paths(meson.project_source_root(), pcc_dir)

  pcc_inc = include_directories(pcc_root)

  pcc_src = files(
    join_paths(pcc_root, 'conductionQ.c'),
    join_paths(pcc_root, 'conductionT.c'),
    join_paths(pcc_root, 'tridag.c'),
  )
endif

if get_option('examples').enabled()
  subdir('examples')
endif
if get_option('tests').enabled()
  subdir('tests')
endif
subdir('wrappers')
