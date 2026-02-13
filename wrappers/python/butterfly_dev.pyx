# cython: language_level=3
# distutils: language=c++

"""
Developer/legacy prototype code.

Not required by the production VF/thermal stack. Keeps old Python-side hierarchical VF
prototype code, benchmarks, plotting helpers, etc.
"""

from __future__ import annotations

import os
import numpy as np

# Optional plotting deps (avoid import-time failure on minimal environments)
try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    plt = None  # type: ignore

cimport numpy as cnp
cnp.import_array()

from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

from bf cimport *

bfInit()

# low-level types used by the legacy prototype/test code below
from bbox cimport *
from defs cimport *
from ellipse cimport *
from fac cimport *
from fac_helm2 cimport *
from fac_streamer cimport *
from geom cimport *
from helm2 cimport *
from indexed_mat cimport *
from layer_pot cimport *
from linalg cimport *
from logging cimport *
from mat cimport *
from mat_block_coo cimport *
from mat_block_dense cimport *
from mat_block_diag cimport *
from mat_csr_real cimport *
from mat_dense_complex cimport *
from mat_dense_real cimport *
from mat_diag_real cimport *
from mat_diff cimport *
from mat_func cimport *
from mat_identity cimport *
from mat_product cimport *
from mat_python cimport *
from mat_vf_hier cimport *
from node_span cimport *
from perm cimport *
from points cimport *
from ptr_array cimport *
from quadtree cimport *
from quadtree_node cimport *
from rand cimport *
from real_array cimport *
from size_array cimport *
from tree cimport *
from tree_level_iter cimport *
from tree_node cimport *
from tree_traversals cimport *
from trimesh cimport *
from types cimport *
from vec cimport *
from vec_real cimport *
from vectors cimport *
from vf_hier cimport *

from butterfly cimport (
    Mat, MatCsrReal, Trimesh, Quadtree, QuadtreeNode,
    VecReal, Vec
)
from butterfly_hier cimport (
    FFBlock, FFSparseLeaf, FFSvdLeaf, FFNode,
    HierarchicalFormFactor, FormFactorMat,
    _quadtree_from_trimesh_xy, _build_ff_block, VfHier as VfHierHier
)


def build_hierarchical_form_factor(tm,
                                   double tol=1e-3,
                                   double eta=2.0,
                                   BfSize leaf_max=256,
                                   BfSize leaf_min=1,
                                   BfSize min_svd_size=5000,  #16384,
                                   double max_svd_rank_frac=0.9):
    """
    Build a hierarchical form-factor operator for the given stereographic
    mesh `tm`.

    Parameters
    ----------
    tm : Trimesh
        Stereographic mesh (xy already in projected coords).
    tol : float
        Relative SVD tolerance for far blocks.
    eta : float
        Admissibility parameter (geometric far/near test).
    leaf_max : BfSize
        Max block dimension before forcing a split.
    leaf_min : BfSize
        Min block dimension to allow as a leaf.
    min_svd_size : BfSize
        Minimum number of entries (m * n) in a far block before
        SVD is even attempted (cf. python-flux min_size).
    max_svd_rank_frac : float
        Maximum allowed rank fraction k / min(m, n) for accepting
        an SVD leaf; above this, we keep CSR instead.
    """
    cdef Quadtree qt = _quadtree_from_trimesh_xy(tm)
    cdef QuadtreeNode root = qt.root

    cdef FFBlock root_block = _build_ff_block(
        tm, root, root,
        tol, eta,
        leaf_max, leaf_min,
        min_svd_size, max_svd_rank_frac)

    if root_block is None:
        raise RuntimeError("Failed to build hierarchical FF (empty root block)")

    return HierarchicalFormFactor(root_block, tm.num_faces)

def _test_truncated_svd(int m=10, int n=8, double tol=1e-3):
    """
    Minimal sanity test for Mat.truncated_svd.

    Returns the relative error ||A x - U S V^T x|| / ||A x||
    for a random vector x.
    """
    cdef cnp.ndarray[double, ndim=2, mode="c"] A = np.random.rand(m, n)

    # Build a Mat from a NumPy array; uses existing static constructor
    cdef Mat A_mat = Mat.from_ndarray(A)

    # Compute truncated SVD
    U, S, VT, truncated = A_mat.truncated_svd(tol=tol)

    # Check reconstruction on a random vector
    cdef cnp.ndarray[double, ndim=1] x = np.random.rand(n)

    # A x (via butterfly)
    Ax_vec = A_mat @ x  # VecReal
    Ax = Ax_vec.to_array()  # numpy 1D

    # U S V^T x (via butterfly)
    USVT_x_vec = U @ (S @ (VT @ x))
    USVT_x = USVT_x_vec.to_array()

    rel_err = float(np.linalg.norm(Ax - USVT_x) / np.linalg.norm(Ax))

    print(f"truncated = {truncated}, relative error = {rel_err}")
    return rel_err

cdef void _collect_block_stats_rec(FFBlock block, dict stats):
    """
    Recursively traverse the FFBlock tree and accumulate:

      - num_sparse_leaves
      - num_svd_leaves
      - nnz_sparse_total
      - mem_bytes_sparse_est
      - mem_bytes_svd_est

    The estimates assume:
      - BfSize is 8 bytes
      - double is 8 bytes
    """
    cdef BfSize mA, nnz
    cdef const BfSize *rp
    cdef BfSize m, n, r
    cdef FFNode node
    cdef FFSparseLeaf s_leaf
    cdef FFSvdLeaf svd_leaf
    cdef FFBlock child

    # Sparse leaf
    if isinstance(block, FFSparseLeaf):
        s_leaf = <FFSparseLeaf> block

        mA = bfMatGetNumRows(s_leaf.A.mat)
        rp = bfMatCsrRealGetRowptrConstPtr(s_leaf.A.mat_csr_real)
        nnz = rp[mA]

        stats["num_sparse_leaves"] += 1
        stats["nnz_sparse_total"] += int(nnz)

        # CSR memory estimate: data + indices + indptr
        # assume BfSize is 8 bytes
        stats["mem_bytes_sparse_est"] += int(
            nnz * 8  # data
            + nnz * 8  # colind
            + (mA + 1) * 8  # rowptr
        )
        return

    # SVD leaf
    if isinstance(block, FFSvdLeaf):
        svd_leaf = <FFSvdLeaf> block

        m = <BfSize> svd_leaf.row_idx.shape[0]
        n = <BfSize> svd_leaf.col_idx.shape[0]
        r = svd_leaf.rank

        stats["num_svd_leaves"] += 1
        stats["rank_total"] += int(r)

        # Memory for U (m x r), S (r), VT (r x n)
        stats["mem_bytes_svd_est"] += int(
            (m * r + r + r * n) * 8
        )
        return

    # Internal node
    if isinstance(block, FFNode):
        node = <FFNode> block
        for child in node.children:
            _collect_block_stats_rec(child, stats)
        return

    # Should not happen
    raise RuntimeError("Unknown FFBlock subclass in _collect_block_stats_rec")

def _test_hierarchical_form_factor(path="examples/radiosity/67p.obj",
                                   double tol=1e-2,
                                   double eta=2.0,
                                   BfSize leaf_max=128,
                                   BfSize leaf_min=1):
    """
    Minimal sanity test for the hierarchical FF builder.

    Assumes `path` is an OBJ mesh in stereographic coordinates
    (so that triangle centers' xy are already P[:, :2]).
    """
    tm = Trimesh.from_obj(path)
    # tm.ensure_face_geometry()
    # tm.init_embree()
    print(f'loaded mesh with {tm.num_verts} vertices and {tm.num_faces} faces')

    # Full FF (CSR) for reference
    cdef MatCsrReal FF = FormFactorMat.from_trimesh(tm)  #.mat_csr_real
    # cdef MatCsrReal FF = MatCsrReal.new_view_factor_matrix_from_trimesh(tm)

    # # after building FF = FormFactorMat.from_trimesh(tm)
    # FF_sp = FF.to_scipy_csr()
    #
    # # manually plug some row_idx, col_idx you saw in logs
    # row_idx = np.array([...], dtype=np.uintp)
    # col_idx = np.array([...], dtype=np.uintp)
    #
    # A_ref = FF_sp[row_idx][:, col_idx]  # reference block from full matrix
    #
    # # independently build block from builder
    # A_block = _build_csr_block(tm, row_idx, col_idx).to_scipy_csr()
    #
    # diff = A_block - A_ref
    # print("block vs ref: max abs diff", np.abs(diff.data).max()
    # if diff.nnz else 0.0)
    # print("NaNs in block?", np.isnan(A_block.data).any())
    # print("NaNs in ref?", np.isnan(A_ref.data).any())
    #
    cdef BfSize n = tm.num_faces

    import numpy as _np
    x = _np.random.rand(n)

    # Full MVP via BF
    y_full_vec = FF @ x  # VecReal
    y_full = y_full_vec.to_array()

    # Hierarchical FF
    H = build_hierarchical_form_factor(tm,
                                       tol=tol,
                                       eta=eta,
                                       leaf_max=leaf_max,
                                       leaf_min=leaf_min)
    y_hier = H.apply(x)

    # NaN / Inf diagnostics
    print("y_full: nan?", _np.isnan(y_full).any(),
          "inf?", _np.isinf(y_full).any(),
          "min", _np.nanmin(y_full), "max", _np.nanmax(y_full))

    print("y_hier: nan?", _np.isnan(y_hier).any(),
          "inf?", _np.isinf(y_hier).any(),
          "min", _np.nanmin(y_hier), "max", _np.nanmax(y_hier))

    if _np.isnan(y_full).any() or _np.isinf(y_full).any():
        raise RuntimeError("y_full contains NaN/Inf")

    if _np.isnan(y_hier).any() or _np.isinf(y_hier).any():
        raise RuntimeError("y_hier contains NaN/Inf")

    rel_err = float(
        _np.linalg.norm(y_full - y_hier) / _np.linalg.norm(y_full)
    )
    print(f"[hier FF] n={n}, tol={tol}, eta={eta}, leaf_max={leaf_max}: "
          f"relative error = {rel_err:.3e}")
    return rel_err

def collect_hierarchical_stats(H):
    """
    Collect stats from a HierarchicalFormFactor H.

    Returns a dict with:
      - num_sparse_leaves
      - num_svd_leaves
      - nnz_sparse_total
      - rank_total
      - mem_bytes_sparse_est
      - mem_bytes_svd_est
      - mem_bytes_total_est
    """
    if not isinstance(H, HierarchicalFormFactor):
        raise TypeError("collect_hierarchical_stats expects a HierarchicalFormFactor")

    stats = {
        "num_sparse_leaves": 0,
        "num_svd_leaves": 0,
        "nnz_sparse_total": 0,
        "rank_total": 0,
        "mem_bytes_sparse_est": 0,
        "mem_bytes_svd_est": 0,
    }

    _collect_block_stats_rec(<FFBlock> H.root_block, stats)
    stats["mem_bytes_total_est"] = (
            stats["mem_bytes_sparse_est"] + stats["mem_bytes_svd_est"]
    )
    return stats

cdef bint ff_debug = 0  # 0 = off, 1 = on

def set_ff_debug(bint flag=True):
    """
    Turn hierarchical FF debug prints on/off.
    """
    global ff_debug
    ff_debug = flag

def benchmark_form_factor(path="examples/radiosity/67p.obj",
                          double tol=1e-2,
                          double eta=2.0,
                          BfSize leaf_max=128,
                          BfSize leaf_min=1,
                          int mvp_repeats=5,
                          BfSize min_svd_size=16384,
                          double max_svd_rank_frac=0.9):
    """
    Benchmark full vs hierarchical form-factor:

      - build time
      - MVP time
      - memory (approx)

    Returns a dict of metrics and also prints a short summary.
    """
    import time
    import numpy as _np

    tm = Trimesh.from_obj(path)
    print(f"[bench] loaded mesh with {tm.num_verts} verts, {tm.num_faces} faces")

    # -------------------------
    # Full CSR FF: build + mem
    # -------------------------
    t0 = time.perf_counter()
    FF = FormFactorMat.from_trimesh(tm)
    t_full_build = time.perf_counter() - t0
    print(f"[bench] full FF build: {t_full_build:.3f} s")

    FF_sp = FF.to_scipy_csr()
    mem_full = (
            FF_sp.data.nbytes +
            FF_sp.indices.nbytes +
            FF_sp.indptr.nbytes
    )
    print(f"[bench] full FF memory: {mem_full / 1e6:.3f} MB")

    # -------------------------
    # Hierarchical FF: build + mem
    # -------------------------
    t0 = time.perf_counter()
    H = build_hierarchical_form_factor(tm,
                                       tol=tol,
                                       eta=eta,
                                       leaf_max=leaf_max,
                                       leaf_min=leaf_min,
                                       min_svd_size=min_svd_size,
                                       max_svd_rank_frac=max_svd_rank_frac)
    t_hier_build = time.perf_counter() - t0
    print(f"[bench] hier FF build: {t_hier_build:.3f} s")

    hstats = collect_hierarchical_stats(H)
    mem_hier = hstats["mem_bytes_total_est"]
    print(f"[bench] hier FF memory est: {mem_hier / 1e6:.3f} MB "
          f"(sparse ~{hstats['mem_bytes_sparse_est'] / 1e6:.3f} MB, "
          f"SVD ~{hstats['mem_bytes_svd_est'] / 1e6:.3f} MB)")

    print(f"[bench] leaves: sparse={hstats['num_sparse_leaves']} "
          f"svd={hstats['num_svd_leaves']} total_rank={hstats['rank_total']}")

    # -------------------------
    # MVP timings
    # -------------------------
    n = tm.num_faces
    x = _np.random.rand(n)

    # Warm-up
    _ = (FF @ x).to_array()
    _ = H.apply(x)

    # Full FF MVP
    t0 = time.perf_counter()
    for _i in range(mvp_repeats):
        y_full_vec = FF @ x
        y_full = y_full_vec.to_array()
    t_full_mvp = (time.perf_counter() - t0) / mvp_repeats
    print(f"[bench] full FF MVP: {t_full_mvp * 1e3:.2f} ms (avg over {mvp_repeats})")

    # Hierarchical FF MVP
    t0 = time.perf_counter()
    for _i in range(mvp_repeats):
        y_hier = H.apply(x)
    t_hier_mvp = (time.perf_counter() - t0) / mvp_repeats
    print(f"[bench] hier FF MVP: {t_hier_mvp * 1e3:.2f} ms (avg over {mvp_repeats})")

    # Relative error
    rel_err = float(
        _np.linalg.norm(y_full - y_hier) / _np.linalg.norm(y_full)
    )
    print(f"[bench] relative error: {rel_err:.3e}")

    return {
        "t_full_build": t_full_build,
        "t_hier_build": t_hier_build,
        "mem_full_bytes": mem_full,
        "mem_hier_bytes": mem_hier,
        "mem_hier_sparse_bytes": hstats["mem_bytes_sparse_est"],
        "mem_hier_svd_bytes": hstats["mem_bytes_svd_est"],
        "num_sparse_leaves": hstats["num_sparse_leaves"],
        "num_svd_leaves": hstats["num_svd_leaves"],
        "rank_total": hstats["rank_total"],
        "t_full_mvp": t_full_mvp,
        "t_hier_mvp": t_hier_mvp,
        "rel_err": rel_err,
    }

def build_vf_block_no_svd(Trimesh tm,
                          double eta=2.0,
                          BfSize leaf_max=128,
                          BfSize leaf_min=1,
                          BfReal min_area=0.0):
    """
    Build a CSR-only block view-factor operator using BF's BfMatBlockCoo.

    This mirrors VfHier.from_trimesh (same quadtree construction and
    same leafMax/leafMin stopping rules), but returns a generic Mat
    backed by a BfMatBlockCoo.
    """
    cdef Quadtree qt = _quadtree_from_trimesh_xy(tm)

    cdef BfMat *A_ptr = bfMatVfHierNewFromQuadtree(
        <const BfTrimesh *> tm.trimesh,
        qt.quadtree,
        eta,
        leaf_max,
        leaf_min,
        min_area,
        0.0,  # tol (unused when minSvdSize == 0)
        0,  # minSvdSize == 0 => disable SVD
        0.0)  # maxSvdRankFrac (ignored when SVD disabled)

    if A_ptr == NULL:
        raise RuntimeError("bfMatVfHierNewFromQuadtree returned NULL")

    # Wrap as a generic Mat; BF handles the dynamic type internally.
    cdef Mat A = Mat.__new__(Mat)
    A.mat = A_ptr
    return A

def test_vf_block_no_svd(path=None,
                         double eta=2.0,
                         BfSize leaf_max=128,
                         BfSize leaf_min=1,
                         int seed_val=123):
    """
    Sanity test for the CSR-only block view-factor operator.

    Compares:
        y_tree  = VfHier.apply(x)
        y_block = A_block @ x

    on the same mesh and quadtree parameters.
    """
    import os
    import time
    import numpy as _np

    # Default mesh path: examples/radiosity/67p.obj next to this module
    if path is None:
        mod_dir = os.path.dirname(__file__)
        path = os.path.join(mod_dir, "examples", "radiosity", "67p.obj")

    print(f"[vf-block] loading mesh from: {path}")
    tm = Trimesh.from_obj(path)
    tm.ensure_face_geometry()
    tm.init_embree()
    print(f"[vf-block] mesh: {tm.num_verts} verts, {tm.num_faces} faces")

    # --- hierarchical C operator (pure sparse tree) ---
    print("[vf-block] building VfHier ...")
    t0 = time.perf_counter()
    H = VfHierHier.from_trimesh(tm,
                            eta=eta,
                            leaf_max=leaf_max,
                            leaf_min=leaf_min)
    t_hier_build = time.perf_counter() - t0
    print(f"[vf-block] VfHier build: {t_hier_build:.3f} s")

    # --- block COO operator (CSR-only blocks) ---
    print("[vf-block] building BfMatBlockCoo ...")
    t0 = time.perf_counter()
    A_block = build_vf_block_no_svd(tm,
                                    eta=eta,
                                    leaf_max=leaf_max,
                                    leaf_min=leaf_min)
    t_block_build = time.perf_counter() - t0
    print(f"[vf-block] BfMatBlockCoo build: {t_block_build:.3f} s")

    # --- random test vector ---
    _np.random.seed(seed_val)
    x = _np.random.rand(tm.num_faces)

    # tree MVP
    t0 = time.perf_counter()
    y_tree = H.apply(x)
    t_hier_mvp = time.perf_counter() - t0

    # block MVP
    print(f"[vf-block] pre-matmul")
    t0 = time.perf_counter()
    y_block_vec = A_block @ x
    t_block_mvp = time.perf_counter() - t0
    print(f"[vf-block] matmul in {t_block_mvp:.2g}")

    # reify y_block_vec -> numpy
    if isinstance(y_block_vec, Vec):
        # __matmul__ with a vector returns a VecReal
        y_block = (<VecReal> y_block_vec).to_array()
    else:
        y_block = _np.asarray(y_block_vec, dtype=_np.float64)

    print(f"[vf-block] ||y_block|| = {np.linalg.norm(y_block):.6g}")

    # --- diagnostics ---
    print("[vf-block] y_tree:  nan?", _np.isnan(y_tree).any(),
          "inf?", _np.isinf(y_tree).any(),
          "||y_tree||", float(_np.linalg.norm(y_tree)))

    print("[vf-block] y_block: nan?", _np.isnan(y_block).any(),
          "inf?", _np.isinf(y_block).any(),
          "||y_block||", float(_np.linalg.norm(y_block)))

    diff = y_tree - y_block
    print("[vf-block] diff:   nan?", _np.isnan(diff).any(),
          "inf?", _np.isinf(diff).any(),
          "||diff||", float(_np.linalg.norm(diff)))

    denom = float(_np.linalg.norm(y_tree))
    if denom == 0.0:
        print("[vf-block] WARNING: ||y_tree|| == 0, rel_err ill-defined")
        rel_err = _np.nan
    else:
        rel_err = float(_np.linalg.norm(diff) / denom)

    print(f"[vf-block] VfHier MVP: {t_hier_mvp * 1e3:.2f} ms")
    print(f"[vf-block] Block MVP:  {t_block_mvp * 1e3:.2f} ms")
    print(f"[vf-block] relative error = {rel_err:.3e}")

    return rel_err

def test_vf_hier_no_svd(path=None,
                        double eta=2.0,
                        BfSize leaf_max=128,
                        BfSize leaf_min=1,
                        int seed_val=123):
    """
    Sanity test for the C-side hierarchical VF (VfHier, no SVD).

    Compares y_ref = F_csr @ x versus y_hier = H.apply(x) on the same mesh,
    prints timings and basic NaN/Inf diagnostics, and returns the relative error.
    """
    import os, time
    import numpy as _np

    # Default mesh: examples/radiosity/67p.obj inside the installed package
    if path is None:
        mod_dir = os.path.dirname(__file__)
        path = os.path.join(mod_dir, "examples", "radiosity", "67p.obj")

    print(f"[vf-hier] loading mesh from: {path}")
    tm = Trimesh.from_obj(path)
    tm.ensure_face_geometry()
    tm.init_embree()
    print(f"[vf-hier] mesh: {tm.num_verts} verts, {tm.num_faces} faces")

    qt = _quadtree_from_trimesh_xy(tm)
    print("quadtree max depth:", qt.get_max_depth())

    # --- full CSR FF (reference) ---
    t0 = time.perf_counter()
    F_csr = MatCsrReal.new_view_factor_matrix_from_trimesh(tm)
    t_full_build = time.perf_counter() - t0
    print(f"[vf-hier] full FF (CSR) build: {t_full_build:.3f} s")

    # --- hierarchical C operator (pure sparse, no SVD) ---
    t0 = time.perf_counter()
    H = build_vf_block_no_svd(tm,
                              eta=eta,
                              leaf_max=leaf_max,
                              leaf_min=leaf_min)
    t_hier_build = time.perf_counter() - t0
    print(f"[vf-hier] hier FF (C) build: {t_hier_build:.3f} s")

    # --- random test vector ---
    _np.random.seed(seed_val)
    x = _np.random.rand(tm.num_faces)

    # full MVP
    t0 = time.perf_counter()
    y_ref_vec = F_csr @ x  # VecReal
    y_ref = y_ref_vec.to_array()  # 1D numpy
    t_full_mvp = time.perf_counter() - t0

    # hier MVP
    t0 = time.perf_counter()
    y_block_vec = H @ x
    t_hier_mvp = time.perf_counter() - t0
    print(f"[vf-hier] matmul in {t_hier_mvp:.2g}")

    # reify y_block_vec -> numpy
    if isinstance(y_block_vec, Vec):
        # __matmul__ with a vector returns a VecReal
        y_hier = (<VecReal> y_block_vec).to_array()
    else:
        y_hier = _np.asarray(y_block_vec, dtype=_np.float64)

    # --- NaN / Inf diagnostics ---
    print("[vf-hier] y_ref:  nan?", _np.isnan(y_ref).any(),
          "inf?", _np.isinf(y_ref).any(),
          "||y_ref||", float(_np.linalg.norm(y_ref)))

    print("[vf-hier] y_hier: nan?", _np.isnan(y_hier).any(),
          "inf?", _np.isinf(y_hier).any(),
          "||y_hier||", float(_np.linalg.norm(y_hier)))

    diff = y_ref - y_hier
    print("[vf-hier] diff:   nan?", _np.isnan(diff).any(),
          "inf?", _np.isinf(diff).any(),
          "||diff||", float(_np.linalg.norm(diff)))

    denom = float(_np.linalg.norm(y_ref))
    if denom == 0.0:
        print("[vf-hier] WARNING: ||y_ref|| == 0, rel_err ill-defined")
        rel_err = _np.nan
    else:
        rel_err = float(_np.linalg.norm(diff) / denom)

    print(f"[vf-hier] full MVP:  {t_full_mvp * 1e3:.2f} ms")
    print(f"[vf-hier] hier MVP:  {t_hier_mvp * 1e3:.2f} ms")
    print(f"[vf-hier] relative error = {rel_err:.3e}")

    return rel_err

def test_vf_hier(path=None,
                 double eta=2.0,
                 BfSize leaf_max=128,
                 BfSize leaf_min=1,
                 double tol=1e-2,
                 BfSize min_svd_size=16384,
                 double max_svd_rank_frac=0.9,
                 int seed_val=123):
    """
    Test the compressed C-side hierarchical VF (VfHier with SVD)
    against the full CSR FF.

    Builds:
        F_csr  = full view-factor matrix (MatCsrReal)
        H_comp = VfHier.from_trimesh(...)

    and compares y_ref = F_csr @ x to y_comp = H_comp.apply(x)
    for a random test vector x.
    """
    import os, time
    import numpy as _np
    cdef VfHierHier H_comp

    # Default mesh: examples/radiosity/67p.obj inside the installed package
    if path is None:
        mod_dir = os.path.dirname(__file__)
        path = os.path.join(mod_dir, "examples", "radiosity", "67p.obj")

    print(f"[vf-hier-SVD] loading mesh from: {path}")
    tm = Trimesh.from_obj(path)
    tm.ensure_face_geometry()
    tm.init_embree()
    print(f"[vf-hier-SVD] mesh: {tm.num_verts} verts, {tm.num_faces} faces")

    # --- full CSR FF (reference) ---
    t0 = time.perf_counter()
    F_csr = MatCsrReal.new_view_factor_matrix_from_trimesh(tm)
    t_full_build = time.perf_counter() - t0
    print(f"[vf-hier-SVD] full FF (CSR) build: {t_full_build:.3f} s")

    FF_sp = F_csr.to_scipy_csr()
    mem_full = (
            FF_sp.data.nbytes +
            FF_sp.indices.nbytes +
            FF_sp.indptr.nbytes
    )
    print(f"[bench] full FF memory: {mem_full / 1e6:.3f} MB")

    # --- compressed C hierarchical FF (with SVD) ---
    t0 = time.perf_counter()
    H_comp = VfHierHier.from_trimesh(tm,
                                 eta=eta,
                                 leaf_max=leaf_max,
                                 leaf_min=leaf_min,
                                 tol=tol,
                                 min_svd_size=min_svd_size,
                                 max_svd_rank_frac=max_svd_rank_frac)
    t_comp_build = time.perf_counter() - t0
    print(f"[vf-hier-SVD] compressed VF (C) build: {t_comp_build:.3f} s")

    # --- stats on hierarchy ---
    cdef BfVfHierStats stats
    bfVfHierCollectStats(H_comp.vfHier, &stats)

    print(f"[vf-hier-SVD] blocks: sparse={stats.numSparseLeaves} "
          f"svd={stats.numSvdLeaves} nodes={stats.numNodeBlocks}")
    print(f"[vf-hier-SVD] nnz(sparse leaves)={stats.nnzSparseTotal}")
    print(f"[vf-hier-SVD] mem est: sparse={stats.memBytesSparseEst / 1e6:.3f} MB, "
          f"SVD={stats.memBytesSvdEst / 1e6:.3f} MB, "
          f"total={(stats.memBytesSparseEst + stats.memBytesSvdEst) / 1e6:.3f} MB")
    print(f"[vf-hier-SVD] total SVD rank sum={stats.rankTotal}")

    # --- random test vector ---
    _np.random.seed(seed_val)
    x = _np.random.rand(tm.num_faces)

    # full MVP
    t0 = time.perf_counter()
    y_ref_vec = F_csr @ x  # VecReal
    y_ref = y_ref_vec.to_array()  # 1D numpy
    t_full_mvp = time.perf_counter() - t0

    # compressed MVP
    t0 = time.perf_counter()
    y_comp = H_comp.apply(x)
    t_comp_mvp = time.perf_counter() - t0

    print(f"[vf-hier-SVD] full MVP:  {t_full_mvp * 1e3:.2f} ms")
    print(f"[vf-hier-SVD] comp MVP:  {t_comp_mvp * 1e3:.2f} ms")

    # --- NaN / Inf diagnostics ---
    print("[vf-hier-SVD] y_ref:  nan?", _np.isnan(y_ref).any(),
          "inf?", _np.isinf(y_ref).any(),
          "||y_ref||", float(_np.linalg.norm(y_ref)))

    print("[vf-hier-SVD] y_comp: nan?", _np.isnan(y_comp).any(),
          "inf?", _np.isinf(y_comp).any(),
          "||y_comp||", float(_np.linalg.norm(y_comp)))

    diff = y_ref - y_comp
    print("[vf-hier-SVD] diff:   nan?", _np.isnan(diff).any(),
          "inf?", _np.isinf(diff).any(),
          "||diff||", float(_np.linalg.norm(diff)))

    denom = float(_np.linalg.norm(y_ref))
    if denom == 0.0:
        print("[vf-hier-SVD] WARNING: ||y_ref|| == 0, rel_err ill-defined")
        rel_err = _np.nan
    else:
        rel_err = float(_np.linalg.norm(diff) / denom)

    print(f"[vf-hier-SVD] relative error = {rel_err:.3e}")

    return rel_err

def compare_vf_compressions(path=None,
                            double eta=2.0,
                            BfSize leaf_max=128,
                            BfSize leaf_min=1,
                            BfReal min_area=0.0,
                            double tol=1e-2,
                            BfSize min_svd_size=16384,
                            double max_svd_rank_frac=0.9,
                            int mvp_repeats=5,
                            int seed_val=123):
    """
    Compare:
      - full FF (CSR)                    [reference]
      - old compression (VfHier.from_trimesh, trimesh-based)
      - new compression (VfHier.from_csr_and_trimesh, CSR-based)

    This follows the python-flux compressed_form_factors.py logic:
      - build Afull once (ray tracing)
      - build a compressed operator from Afull + quadtree
      - compare timing, memory, and MVP accuracy.
    """
    import os, time
    import numpy as _np

    # -----------------------------
    # Load mesh
    # -----------------------------
    if path is None:
        mod_dir = os.path.dirname(__file__)
        path = os.path.join(mod_dir, "examples", "radiosity", "67p.obj")

    print(f"[cmp] loading mesh from: {path}")
    tm = Trimesh.from_obj(path)
    tm.ensure_face_geometry()
    tm.init_embree()
    print(f"[cmp] mesh: {tm.num_verts} verts, {tm.num_faces} faces")

    cdef BfSize n_faces = tm.num_faces

    # # -----------------------------
    # # Full FF (CSR) build + memory
    # # -----------------------------
    # t0 = time.perf_counter()
    # F_csr = MatCsrReal.new_view_factor_matrix_from_trimesh(tm)
    # t_full_build = time.perf_counter() - t0
    # print(f"[cmp] full FF (CSR) build: {t_full_build:.3f} s")
    #
    # F_sp = F_csr.to_scipy_csr()
    # mem_full = (
    #     F_sp.data.nbytes +
    #     F_sp.indices.nbytes +
    #     F_sp.indptr.nbytes
    # )
    # print(f"[cmp] full FF memory: {mem_full/1e6:.3f} MB")

    # -----------------------------
    # Old compression: trimesh-based VfHier
    # -----------------------------
    cdef VfHierHier H_old
    cdef VfHierHier H_new

    t0 = time.perf_counter()
    H_old = VfHierHier.from_trimesh(tm,
                                eta=eta,
                                leaf_max=leaf_max,
                                leaf_min=leaf_min,
                                min_area=min_area,
                                tol=tol,
                                min_svd_size=min_svd_size,
                                max_svd_rank_frac=max_svd_rank_frac,
                                ensure_geometry=False,
                                init_embree_scene=False)
    t_old_build = time.perf_counter() - t0
    print(f"[cmp] OLD VfHier (trimesh) build: {t_old_build:.3f} s")

    cdef BfVfHierStats stats_old
    bfVfHierCollectStats(H_old.vfHier, &stats_old)

    mem_old_sparse = stats_old.memBytesSparseEst
    mem_old_svd = stats_old.memBytesSvdEst
    mem_old_total = mem_old_sparse + mem_old_svd

    print(f"[cmp] OLD VfHier memory: total={mem_old_total / 1e6:.3f} MB "
          f"(sparse={mem_old_sparse / 1e6:.3f} MB, "
          f"SVD={mem_old_svd / 1e6:.3f} MB)")
    print(f"[cmp] OLD blocks: sparse={stats_old.numSparseLeaves} "
          f"svd={stats_old.numSvdLeaves} nodes={stats_old.numNodeBlocks} "
          f"rank_total={stats_old.rankTotal}")

    # # -----------------------------
    # # New compression: CSR + quadtree
    # # -----------------------------
    # t0 = time.perf_counter()
    # H_new = VfHier.from_csr_and_trimesh(F_csr,
    #                                     tm,
    #                                     eta=eta,
    #                                     leaf_max=leaf_max,
    #                                     leaf_min=leaf_min,
    #                                     tol=tol,
    #                                     min_svd_size=min_svd_size,
    #                                     max_svd_rank_frac=max_svd_rank_frac)
    # t_new_build = time.perf_counter() - t0
    # print(f"[cmp] NEW VfHier (CSR+quadtree) build: {t_new_build:.3f} s")
    #
    # cdef BfVfHierStats stats_new
    # bfVfHierCollectStats(H_new.vfHier, &stats_new)
    #
    # mem_new_sparse = stats_new.memBytesSparseEst
    # mem_new_svd    = stats_new.memBytesSvdEst
    # mem_new_total  = mem_new_sparse + mem_new_svd
    #
    # print(f"[cmp] NEW VfHier memory: total={mem_new_total/1e6:.3f} MB "
    #       f"(sparse={mem_new_sparse/1e6:.3f} MB, "
    #       f"SVD={mem_new_svd/1e6:.3f} MB)")
    # print(f"[cmp] NEW blocks: sparse={stats_new.numSparseLeaves} "
    #       f"svd={stats_new.numSvdLeaves} nodes={stats_new.numNodeBlocks} "
    #       f"rank_total={stats_new.rankTotal}")
    #
    # print(f"[cmp] combined FULL+NEW build (for flux-style pipeline): "
    #       f"{(t_full_build + t_new_build):.3f} s")

    # -----------------------------
    # MVP timings and accuracy
    # -----------------------------
    _np.random.seed(seed_val)
    x = _np.random.rand(n_faces)

    # warm-up
    # _ = (F_csr @ x).to_array()
    _ = H_old.apply(x)
    # _ = H_new.apply(x)

    # # full MVP
    # t0 = time.perf_counter()
    # for _i in range(mvp_repeats):
    #     y_ref_vec = F_csr @ x
    #     y_ref = y_ref_vec.to_array()
    # t_full_mvp = (time.perf_counter() - t0) / mvp_repeats
    # print(f"[cmp] full FF MVP: {t_full_mvp*1e3:.2f} ms (avg over {mvp_repeats})")

    # old MVP
    t0 = time.perf_counter()
    for _i in range(mvp_repeats):
        y_old = H_old.apply(x)
    t_old_mvp = (time.perf_counter() - t0) / mvp_repeats
    print(f"[cmp] OLD VfHier MVP: {t_old_mvp * 1e3:.2f} ms (avg over {mvp_repeats})")

    # # new MVP
    # t0 = time.perf_counter()
    # for _i in range(mvp_repeats):
    #     y_new = H_new.apply(x)
    # t_new_mvp = (time.perf_counter() - t0) / mvp_repeats
    # print(f"[cmp] NEW VfHier MVP: {t_new_mvp*1e3:.2f} ms (avg over {mvp_repeats})")

    # # diagnostics & relative errors
    # print("[cmp] y_ref: nan?", _np.isnan(y_ref).any(),
    #       "inf?", _np.isinf(y_ref).any(),
    #       "||y_ref||", float(_np.linalg.norm(y_ref)))
    # print("[cmp] y_old: nan?", _np.isnan(y_old).any(),
    #       "inf?", _np.isinf(y_old).any(),
    #       "||y_old||", float(_np.linalg.norm(y_old)))
    # print("[cmp] y_new: nan?", _np.isnan(y_new).any(),
    #       "inf?", _np.isinf(y_new).any(),
    #       "||y_new||", float(_np.linalg.norm(y_new)))
    #
    # diff_old = y_ref - y_old
    # diff_new = y_ref - y_new
    #
    # denom = float(_np.linalg.norm(y_ref))
    # if denom == 0.0:
    #     print("[cmp] WARNING: ||y_ref|| == 0, rel_err ill-defined")
    #     rel_err_old = _np.nan
    #     rel_err_new = _np.nan
    # else:
    #     rel_err_old = float(_np.linalg.norm(diff_old) / denom)
    #     rel_err_new = float(_np.linalg.norm(diff_new) / denom)
    #
    # print(f"[cmp] relative error OLD  = {rel_err_old:.3e}")
    # print(f"[cmp] relative error NEW  = {rel_err_new:.3e}")
    #
    # return {
    #     "n_faces": int(n_faces),
    #
    #     "t_full_build": t_full_build,
    #     "t_old_build": t_old_build,
    #     "t_new_build": t_new_build,
    #     "t_full_plus_new_build": t_full_build + t_new_build,
    #
    #     "mem_full_bytes": int(mem_full),
    #
    #     "mem_old_sparse_bytes": int(mem_old_sparse),
    #     "mem_old_svd_bytes": int(mem_old_svd),
    #     "mem_old_total_bytes": int(mem_old_total),
    #
    #     "mem_new_sparse_bytes": int(mem_new_sparse),
    #     "mem_new_svd_bytes": int(mem_new_svd),
    #     "mem_new_total_bytes": int(mem_new_total),
    #
    #     "num_sparse_leaves_old": int(stats_old.numSparseLeaves),
    #     "num_svd_leaves_old": int(stats_old.numSvdLeaves),
    #     "num_nodes_old": int(stats_old.numNodeBlocks),
    #     "rank_total_old": int(stats_old.rankTotal),
    #
    #     "num_sparse_leaves_new": int(stats_new.numSparseLeaves),
    #     "num_svd_leaves_new": int(stats_new.numSvdLeaves),
    #     "num_nodes_new": int(stats_new.numNodeBlocks),
    #     "rank_total_new": int(stats_new.rankTotal),
    #
    #     "t_full_mvp": t_full_mvp,
    #     "t_old_mvp": t_old_mvp,
    #     "t_new_mvp": t_new_mvp,
    #
    #     "rel_err_old": rel_err_old,
    #     "rel_err_new": rel_err_new,
    # }

cdef extern from "omp.h":
    int omp_get_max_threads()
    int omp_get_num_procs()
    int omp_get_dynamic()
    void omp_set_dynamic(int)
    void omp_set_num_threads(int)

def omp_info():
    return {
        "num_procs": omp_get_num_procs(),
        "max_threads": omp_get_max_threads(),
        "dynamic": omp_get_dynamic(),
    }

def omp_force(int n):
    omp_set_dynamic(0)
    omp_set_num_threads(n)

from cython.parallel cimport prange

# from libc.stdlib cimport malloc, free

def omp_smoke(long n=200_000_000, int nt=0):
    cdef double s = 0
    cdef Py_ssize_t i
    if nt <= 0:
        nt = 0  # let OMP decide / env
    for i in prange(n, schedule='static', num_threads=nt, nogil=True):
        s += (i % 97) * 1e-12
    return s