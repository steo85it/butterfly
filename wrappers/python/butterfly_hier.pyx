# cython: language_level=3
# distutils: language=c++

"""
Hier/thermal-facing wrapper layer.

This module contains the Trimesh/VfHier/tree infrastructure used by thermal_from_ff.py
and related VF/radiosity workflows.
"""

# cython: language_level=3
from __future__ import annotations

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdio cimport printf
from libc.math cimport sqrt

# C API
from bf cimport *

from butterfly cimport *
bfInit()

# Local debug toggle used in a few branches in this file too
cdef bint ff_debug = 0

# -----------------------------------------------------------------------------
# Cython imports from BF pxd modules (same style as butterfly_original.pyx)
# -----------------------------------------------------------------------------
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
from octree cimport *
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

def set_bf_log_level(level: str):
    cdef bytes b = level.encode()
    if level == "todo":
        bfSetLogLevel(BF_LOG_LEVEL_TODO)
    elif level == "debug":
        bfSetLogLevel(BF_LOG_LEVEL_DEBUG)
    elif level == "info":
        bfSetLogLevel(BF_LOG_LEVEL_INFO)
    elif level == "warn":
        bfSetLogLevel(BF_LOG_LEVEL_WARN)
    elif level == "error":
        bfSetLogLevel(BF_LOG_LEVEL_ERROR)
    else:
        raise ValueError("level must be one of: todo, debug, info, warn, error")

def sparse_svd_print_stats(tag=""):
    cdef bytes b = (<str>tag).encode()
    bfSparseSvdPrintStats(b)

def sparse_svd_reset_stats():
    bfSparseSvdResetStats()

cdef class FormFactorMat(MatCsrReal):
    """
    Wrapper around a full view-factor matrix (CSR) with SVD-based compression.
    """

    @staticmethod
    def from_trimesh(Trimesh tm,
                     bint ensure_geometry=True,
                     bint init_embree_scene=True):
        """
        Build the full NxN view-factor matrix using Embree-based ray tracing.

        This will, by default:
          - ensure face normals exist, and
          - initialize Embree for the mesh,
        since both are required by the underlying C builder.
        """
        if ensure_geometry:
            tm.ensure_face_geometry()

        if init_embree_scene:
            tm.init_embree()

        # Build CSR FF using the standard BF path
        cdef MatCsrReal M = MatCsrReal.new_view_factor_matrix_from_trimesh(tm)

        # Re-wrap into FormFactorMat using the same underlying pointers.
        # (Assumes MatCsrReal doesn't free the pointer in __dealloc__.)
        cdef FormFactorMat FF = FormFactorMat.__new__(FormFactorMat)
        FF.mat_csr_real = M.mat_csr_real
        FF.mat = M.mat

        # STEAL: prevent M.__dealloc__ from freeing the pointers we just moved
        M.mat_csr_real = NULL
        M.mat = NULL

        return FF

    def truncated_svd(self, double tol):
        """
        Convenience wrapper: call the base Mat.truncated_svd with a float tol.
        """
        return Mat.truncated_svd(self, tol=tol)

    def as_low_rank_operator(self, double tol):
        """
        Return a MatProduct implementing FF ≈ U S V^T at tolerance tol.
        """
        cdef Mat U
        cdef MatDiagReal S
        cdef Mat VT
        cdef bint truncated

        # only works with DenseMat, not very useful
        U, S, VT, truncated = self.truncated_svd(tol)

        # MatProduct.from_factors(U, S, VT) uses BF MatProduct underneath
        cdef MatProduct P = MatProduct.from_factors(U, S, VT)
        return P

# ============================================================
# Hierarchical form-factor compression (stereographic quadtree)
# ============================================================

# Build quadtree from face centroids/normals using V[:, :2] (stereographic already)
cdef Quadtree _quadtree_from_trimesh_xy(Trimesh tm):
    cdef cnp.ndarray V = tm.verts  # (Nv, 3), dtype float64
    cdef cnp.ndarray F = tm.faces  # (Nf, 3), dtype uintp or int64

    cdef cnp.ndarray VF = V[F]  # (Nf, 3, 3)

    cdef cnp.ndarray P3 = VF.mean(axis=1)  # (Nf, 3)
    cdef cnp.ndarray P2 = P3[:, :2].astype(np.float64, copy=False)  # (Nf, 2)

    cdef cnp.ndarray C = np.cross(VF[:, 1, :] - VF[:, 0, :],
                                  VF[:, 2, :] - VF[:, 0, :])  # (Nf, 3)
    cdef cnp.ndarray N2 = C[:, :2].astype(np.float64, copy=False)  # (Nf, 2)
    cdef cnp.ndarray nrm = np.sqrt((N2 * N2).sum(axis=1))  # (Nf,)

    cdef cnp.ndarray m = nrm > 0
    if m.any():
        N2[m] /= nrm[m][:, None]
    else:
        N2[...] = 0.0

    cdef Points2   pts = Points2(P2.shape[0])
    cdef Vectors2  nn = Vectors2()
    cdef BfPoint2  p, q
    cdef Py_ssize_t i

    for i in range(P2.shape[0]):
        p[0] = <double> P2[i, 0]
        p[1] = <double> P2[i, 1]
        bfPoints2Append(pts.points, p)

        q[0] = <double> N2[i, 0]
        q[1] = <double> N2[i, 1]
        bfVectors2Append(nn.vectors, q)

    return Quadtree.from_points_and_normals(pts, nn)

cdef inline bint _is_far_bbox(const BfBbox2 * br,
                              const BfBbox2 * bc,
                              double eta) nogil:
    """
    Pure-C admissibility test using raw BfBbox2 structs.
    Safe to call without the GIL.
    """
    cdef double br_xmin = br.min[0]
    cdef double br_xmax = br.max[0]
    cdef double br_ymin = br.min[1]
    cdef double br_ymax = br.max[1]

    cdef double bc_xmin = bc.min[0]
    cdef double bc_xmax = bc.max[0]
    cdef double bc_ymin = bc.min[1]
    cdef double bc_ymax = bc.max[1]

    cdef double crx = 0.5 * (br_xmin + br_xmax)
    cdef double cry = 0.5 * (br_ymin + br_ymax)
    cdef double ccx = 0.5 * (bc_xmin + bc_xmax)
    cdef double ccy = 0.5 * (bc_ymin + bc_ymax)

    cdef double dxr = br_xmax - br_xmin
    cdef double dyr = br_ymax - br_ymin
    cdef double dxc = bc_xmax - bc_xmin
    cdef double dyc = bc_ymax - bc_ymin

    cdef double diam_r = dxr if dxr > dyr else dyr
    cdef double diam_c = dxc if dxc > dyc else dyc
    cdef double diam = diam_r if diam_r > diam_c else diam_c

    cdef double dx = crx - ccx
    cdef double dy = cry - ccy
    cdef double dist = sqrt(dx * dx + dy * dy)

    if dist == 0:
        # identical or overlapping boxes → treat as near
        return 0

    return diam <= eta * dist

cdef bint _is_far(QuadtreeNode rowNode,
                  QuadtreeNode colNode,
                  double eta):
    """
    Admissibility test in the stereographic plane.

    Extracts BfBbox2 from the QuadtreeNodes (requires GIL because
    rowNode/colNode are Python objects), then calls the nogil helper.
    """
    cdef BfBbox2 br = bfQuadtreeNodeGetBbox(rowNode.quadtreeNode)
    cdef BfBbox2 bc = bfQuadtreeNodeGetBbox(colNode.quadtreeNode)
    return _is_far_bbox(&br, &bc, eta)

cdef cnp.ndarray _sizearray_to_numpy(SizeArray idx):
    """
    Convert a BfSizeArray to a 1D numpy array of dtype uintp.

    The copy ensures Python owns the memory (safe if BF frees its own).
    """
    cdef BfSize n = len(idx)
    cdef const BfSize * ptr = bfSizeArrayGetDataPtr(idx.sizeArray)
    return np.asarray(<BfSize[:n]> ptr, dtype=np.uintp).copy()

def _build_csr_block(Trimesh tm,
                     cnp.ndarray row_idx,
                     cnp.ndarray col_idx):
    """
    Build a CSR sub-block F_{I,J} by calling the native
    bfMatCsrRealNewViewFactorMatrixFromTrimesh on the given
    row/column index sets.

    `row_idx` and `col_idx` must be 1D uintp arrays.

    Returns:
        MatCsrReal block
    """
    if row_idx.shape[0] == 0 or col_idx.shape[0] == 0:
        # return an explicit empty CSR with correct shape if you have a constructor,
        # otherwise raise cleanly
        raise ValueError("cannot build CSR block with empty row/col index set")

    if row_idx.ndim != 1 or col_idx.ndim != 1:
        raise ValueError("row_idx and col_idx must be 1D arrays")

    # Typed memoryviews backed by the numpy arrays
    cdef BfSize[::1] I = row_idx
    cdef BfSize[::1] J = col_idx

    cdef BfSizeArray *rowInds = bfSizeArrayNewView(I.shape[0], &I[0])
    cdef BfSizeArray *colInds = bfSizeArrayNewView(J.shape[0], &J[0])

    cdef MatCsrReal A = MatCsrReal.__new__(MatCsrReal)
    A.mat_csr_real = bfMatCsrRealNewViewFactorMatrixFromTrimesh(
        tm.trimesh, rowInds, colInds)
    A.mat = bfMatCsrRealToMat(A.mat_csr_real)

    # The builder should have copied what it needs; free the views immediately.
    bfSizeArrayDeinitAndDealloc(&rowInds)
    bfSizeArrayDeinitAndDealloc(&colInds)

    if A.mat_csr_real == NULL or A.mat == NULL:
        raise RuntimeError("bfMatCsrRealNewViewFactorMatrixFromTrimesh returned NULL")

    return A

cdef class FFBlock:
    """
    Abstract base for hierarchical FF blocks.

    Each block knows which global row/column indices it acts on.
    """

    def __init__(self, cnp.ndarray row_idx, cnp.ndarray col_idx):
        self.row_idx = row_idx
        self.col_idx = col_idx

    cpdef apply(self, cnp.ndarray x, cnp.ndarray y):
        """
        Accumulate y[row_idx] += F_block @ x[col_idx].

        (Implemented in subclasses.)
        """
        raise NotImplementedError()

cdef class FFSparseLeaf(FFBlock):
    """
    Leaf block storing a sparse CSR submatrix (MatCsrReal).

    This is used for "near" blocks where SVD compression is not
    applied.
    """

    def __init__(self,
                 cnp.ndarray row_idx,
                 cnp.ndarray col_idx,
                 MatCsrReal A):
        FFBlock.__init__(self, row_idx, col_idx)
        self.A = A

    def __dealloc__(self):
        # Nothing special: MatCsrReal's own dealloc will clean up its BF-side memory.
        pass

    # cpdef apply(self, cnp.ndarray x, cnp.ndarray y):
    #     """
    #     y[row_idx] += A @ x[col_idx], with A a CSR block whose colind
    #     entries have been reindexed to local [0, len(col_idx)).
    #
    #     This uses BF's native MatCsrReal @ VecReal kernel.
    #     """
    #     # Extract local subvector and ensure it's float64, contiguous
    #     cdef cnp.ndarray xsub = np.asarray(x[self.col_idx], dtype=np.float64)
    #
    #     if xsub.ndim != 1:
    #         raise ValueError("FFSparseLeaf.apply: xsub must be 1D")
    #
    #     # Wrap xsub as a VecReal view for BF
    #     cdef VecReal x_vec = VecReal.from_ndarray(xsub)
    #
    #     # BF MVP (MatCsrReal @ VecReal -> VecReal)
    #     cdef object ysub_vec = self.A @ x_vec
    #     cdef cnp.ndarray ysub = ysub_vec.to_array()
    #
    #     if ysub.shape[0] != self.row_idx.shape[0]:
    #         raise RuntimeError(
    #             f"FFSparseLeaf.apply: ysub length {ysub.shape[0]} "
    #             f"!= row_idx length {self.row_idx.shape[0]}"
    #         )
    #
    #     # Optional debug:
    #     # import numpy as _np
    #     # if _np.isnan(ysub).any() or _np.isinf(ysub).any():
    #     #     if ff_debug:
    #     #         print("[ff-debug] FFSparseLeaf.apply: NaN/Inf in ysub "
    #     #               f"rows[{int(self.row_idx.min())}:{int(self.row_idx.max())}]")
    #
    #     # Scatter-add into global y
    #     y[self.row_idx] += ysub
    cpdef apply(self, cnp.ndarray x, cnp.ndarray y):
        """
        y[row_idx] += A @ x[col_idx], where:

          - A is a CSR block with colind in [0, mj)
          - self.col_idx[0..mj-1] are the corresponding *global* column indices

        This is a pure C-level CSR matvec:
          acc_i = sum_k A[i, j_local] * x[col_idx[j_local]]

        and then y[row_idx[i]] += acc_i.
        """

        cdef BfSize mA = bfMatGetNumRows(self.A.mat)
        cdef const BfSize *rp = bfMatCsrRealGetRowptrConstPtr(self.A.mat_csr_real)
        cdef const BfSize *ci = bfMatCsrRealGetColindConstPtr(self.A.mat_csr_real)
        cdef const BfReal *da = bfMatCsrRealGetDataConstPtr(self.A.mat_csr_real)

        # Sanity: block rows should match len(row_idx)
        if mA != <BfSize> self.row_idx.shape[0]:
            raise RuntimeError(
                f"FFSparseLeaf.apply: CSR rows {mA} != len(row_idx) {self.row_idx.shape[0]}"
            )

        # Typed views for x, y, row_idx, col_idx
        cdef double[:] x_view = x
        cdef double[:] y_view = y
        cdef BfSize[:] row_idx_view = self.row_idx
        cdef BfSize[:] col_idx_view = self.col_idx

        cdef Py_ssize_t n_x = x_view.shape[0]
        cdef Py_ssize_t n_y = y_view.shape[0]

        cdef BfSize i, k, row_start, row_end
        cdef BfSize j_local
        cdef Py_ssize_t row_global, col_global
        cdef double acc

        for i in range(mA):
            row_start = rp[i]
            row_end = rp[i + 1]

            acc = 0.0
            for k in range(row_start, row_end):
                j_local = ci[k]  # 0..mj-1
                col_global = col_idx_view[j_local]

                if col_global < 0 or col_global >= n_x:
                    # should never happen; guard defensively
                    continue

                acc += da[k] * x_view[col_global]

            row_global = row_idx_view[i]
            if row_global < 0 or row_global >= n_y:
                # again, defensive guard
                continue

            y_view[row_global] += acc

cdef class FFSvdLeaf(FFBlock):
    """
    Leaf block storing a low-rank MatProduct U S V^T.

    This is used for "far" blocks where SVD compression is applied.
    """

    def __init__(self,
                 cnp.ndarray row_idx,
                 cnp.ndarray col_idx,
                 MatProduct P,
                 BfSize rank):
        FFBlock.__init__(self, row_idx, col_idx)
        self.P = P
        self.rank = rank

    cpdef apply(self, cnp.ndarray x, cnp.ndarray y):
        """
        y[row_idx] += (U S V^T) @ x[col_idx]
        """
        cdef cnp.ndarray xsub = np.ascontiguousarray(x[self.col_idx], dtype=np.float64)
        cdef object ysub_vec = self.P @ xsub  # VecReal
        y[self.row_idx] += ysub_vec.to_array()

cdef class FFNode(FFBlock):
    """
    Internal node: just a container of child blocks.
    """

    def __init__(self, list children):
        # row/col indices are not used at this level; store empty arrays
        FFBlock.__init__(self,
                         np.empty(0, dtype=np.uintp),
                         np.empty(0, dtype=np.uintp))
        self.children = children

    cpdef apply(self, cnp.ndarray x, cnp.ndarray y):
        cdef FFBlock child
        for child in self.children:
            child.apply(x, y)

from libc.stdio  cimport printf, fflush, stdout

cdef bint _check_index_range(SizeArray idx,
                             BfSize nFaces,
                             const char *label):
    """
    Return 1 if all indices are in [0, nFaces), else 0 and print a debug line.

    This runs with the GIL, so we can safely access idx.sizeArray (a Python wrapper).
    """
    cdef BfSize i
    cdef BfSize v
    cdef BfSize n = len(idx)
    cdef const BfSize * ptr = bfSizeArrayGetDataPtr(idx.sizeArray)

    for i in range(n):
        v = ptr[i]
        if v >= nFaces:
            if ff_debug:
                printf("[ff-debug] %s index expected of range: %zu >= %zu (pos=%zu)\n",
                       label, v, nFaces, i)
                fflush(stdout)
            return 0

    return 1

cdef FFBlock _build_ff_block(Trimesh tm,
                             QuadtreeNode rowNode,
                             QuadtreeNode colNode,
                             double tol,
                             double eta,
                             BfSize leaf_max,
                             BfSize leaf_min,
                             BfSize min_svd_size,
                             double max_svd_rank_frac):
    """
    Recursively build a hierarchical FF block.

      - If the node pair is "far": build a CSR sub-block and compress
        it via truncated SVD → FFSvdLeaf.

      - If the node pair is "near" and small enough: build CSR
        sub-block and keep it as FFSparseLeaf.

      - Otherwise: split (up to 4x4 children) and recurse → FFNode.
    """

    cdef BfTreeNode *ni_node = rowNode.treeNode
    cdef BfTreeNode *nj_node = colNode.treeNode

    # Get total number of faces as a *C* value (no Python objects in printf)
    cdef BfSize nFaces = bfTrimeshGetNumFaces(tm.trimesh)

    cdef int a, b
    cdef bint leafI = 1
    cdef bint leafJ = 1

    # Detect children -> leaf flags
    for a in range(rowNode.max_num_children):
        if bfTreeNodeHasChild(ni_node, a):
            leafI = 0
            break
    for b in range(colNode.max_num_children):
        if bfTreeNodeHasChild(nj_node, b):
            leafJ = 0
            break

    cdef BfSize i0 = rowNode.i0
    cdef BfSize i1 = rowNode.i1
    cdef BfSize j0 = colNode.i0
    cdef BfSize j1 = colNode.i1

    cdef BfSize mi = 0
    cdef BfSize mj = 0

    # Debug: log basic block info
    if ff_debug:
        printf("[ff-debug] enter block: "
               "i0=%zu i1=%zu j0=%zu j1=%zu leafI=%d leafJ=%d\n",
               i0, i1, j0, j1, leafI, leafJ)
        fflush(stdout)

    if i1 <= i0 or j1 <= j0:
        if ff_debug:
            printf("[ff-debug] empty block, returning None\n")
            fflush(stdout)
        return None  # empty block

    mi = i1 - i0
    mj = j1 - j0

    # Sanity: block sizes cannot exceed nFaces
    if mi > nFaces or mj > nFaces:
        if ff_debug:
            printf("[ff-debug] ERROR: mi=%zu mj=%zu > nFaces=%zu\n",
                   mi, mj, nFaces)
            fflush(stdout)
        raise RuntimeError("hierarchical block size expected of range")

    cdef bint small = (mi <= leaf_max) and (mj <= leaf_max)  # check if ok or restate OR

    cdef SizeArray I_inds
    cdef SizeArray J_inds

    cdef cnp.ndarray row_idx
    cdef cnp.ndarray col_idx

    cdef MatCsrReal A
    # cdef size_t row_handle
    # cdef size_t col_handle
    cdef bint far

    cdef object sp
    cdef cnp.ndarray dense

    cdef Mat A_dense_mat
    cdef object U
    cdef MatDiagReal S
    cdef object VT
    cdef bint truncated

    cdef MatProduct P
    # cdef BfSizeArray * rowOwned
    # cdef BfSizeArray * colOwned

    cdef BfSize mA
    cdef BfSize nA
    cdef BfSize rank

    cdef const BfSize * rp
    cdef BfSize nnz

    cdef double bytes_csr_val
    cdef double bytes_svd_val
    cdef unsigned long long block_size
    cdef BfSize k
    cdef BfSize min_dim
    cdef double rank_frac

    # Leaf-level block (both leaves or size small-enough)
    # if ((leafI and leafJ) or small) and mi >= leaf_min and mj >= leaf_min:
    if (leafI and leafJ) or (small and mi >= leaf_min and mj >= leaf_min):
        if ff_debug:
            printf("[ff-debug] leaf candidate: mi=%zu mj=%zu leafI=%d leafJ=%d small=%d\n",
                   mi, mj, leafI, leafJ, small)
            fflush(stdout)

        # Global index sets for this node pair
        I_inds = rowNode.get_inds()
        J_inds = colNode.get_inds()

        row_idx = _sizearray_to_numpy(I_inds)
        col_idx = _sizearray_to_numpy(J_inds)

        if ff_debug:
            printf("[ff-debug] building CSR block: mi=%zu mj=%zu\n", mi, mj)
            fflush(stdout)

        # Build CSR block using the "view then free" pattern
        A = _build_csr_block(tm, row_idx, col_idx)

        # Make CSR column indices local (0..mj-1) instead of global face indices
        _reindex_csr_block_columns(A, J_inds)

        # Extra guard: ensure CSR shape matches our expectation
        mA = bfMatGetNumRows(A.mat)
        nA = bfMatGetNumCols(A.mat)
        if mA != mi or nA != mj:
            if ff_debug:
                printf("[ff-debug] WARNING: CSR shape mismatch: mA=%zu nA=%zu mi=%zu mj=%zu\n",
                       mA, nA, mi, mj)

        if ff_debug:
            printf("[ff-debug] CSR block OK, testing admissibility\n")
        far = _is_far(rowNode, colNode, eta)
        if ff_debug:
            printf("[ff-debug] admissibility: far=%d\n", far)

        if far:
            # Far block: decide between CSR and SVD based on size and memory
            mA = bfMatGetNumRows(A.mat)
            nA = bfMatGetNumCols(A.mat)

            # 1) skip SVD for tiny blocks (like python-flux min_size)
            block_size = <unsigned long long> mA * <unsigned long long> nA
            if block_size < <unsigned long long> min_svd_size:
                if ff_debug:
                    printf("[ff-debug] FAR block too small for SVD (size=%llu < %zu) -> CSR leaf\n",
                           block_size, min_svd_size)
                    fflush(stdout)
                return FFSparseLeaf(row_idx, col_idx, A)

            if ff_debug:
                printf("[ff-debug] FAR block CSR shape: mA=%zu nA=%zu (size=%llu)\n",
                       mA, nA, block_size)
                fflush(stdout)

            # 2) build dense block and compute truncated SVD
            A_dense_mat = _dense_from_csr_block(A)
            U, S, VT, truncated = A_dense_mat.truncated_svd(tol=tol)

            # rank = number of columns of U
            k = <BfSize> U.shape[1]

            # 3) approximate memory cost: CSR vs SVD
            rp = bfMatCsrRealGetRowptrConstPtr(A.mat_csr_real)
            nnz = rp[mA]

            bytes_csr_val = 0.0
            bytes_svd_val = 0.0

            # crude estimate: data (nnz doubles) + indices (nnz + mA+1 size_t)
            bytes_csr_val = 8.0 * nnz + 8.0 * (nnz + mA + 1)

            # SVD: U (mA x k), VT (k x nA), S (k)
            bytes_svd_val = 8.0 * (mA * k + nA * k + k)

            # 4) reject SVD if rank is essentially full
            min_dim = mA if mA < nA else nA
            rank_frac = 0.0
            if min_dim > 0:
                rank_frac = <double> k / <double> min_dim

            if rank_frac > max_svd_rank_frac:
                if ff_debug:
                    printf("[ff-debug] SVD rank too high: k=%zu (%.3f of min_dim=%zu) -> CSR leaf\n",
                           k, rank_frac, min_dim)
                    fflush(stdout)
                return FFSparseLeaf(row_idx, col_idx, A)

            # 5) reject SVD if it doesn't win in memory
            if bytes_svd_val >= bytes_csr_val:
                if ff_debug:
                    printf("[ff-debug] SVD not beneficial: "
                           "bytes_svd=%.3g >= bytes_csr=%.3g -> CSR leaf\n",
                           bytes_svd_val, bytes_csr_val)
                    fflush(stdout)
                return FFSparseLeaf(row_idx, col_idx, A)

            if ff_debug:
                printf("[ff-debug] SVD accepted: k=%zu, bytes_svd=%.3g < bytes_csr=%.3g\n",
                       k, bytes_svd_val, bytes_csr_val)
                fflush(stdout)

            P = MatProduct.from_factors(<Mat> U, S, <Mat> VT)
            return FFSvdLeaf(row_idx, col_idx, P, k)
        else:
            if ff_debug:
                printf("[ff-debug] NEAR block -> FFSparseLeaf\n")
                fflush(stdout)
            return FFSparseLeaf(row_idx, col_idx, A)

    # Otherwise: need to split at least one side
    if ff_debug:
        printf("[ff-debug] splitting block: mi=%zu mj=%zu leafI=%d leafJ=%d small=%d\n",
               mi, mj, leafI, leafJ, small)
        fflush(stdout)

    cdef list children = []
    cdef QuadtreeNode nai, nbj
    cdef BfTreeNode *child_node

    if not leafI and not leafJ:
        for a in range(rowNode.max_num_children):
            if bfTreeNodeHasChild(ni_node, a):
                child_node = bfTreeNodeGetChild(ni_node, a)
                nai = reify_tree_node(child_node)
                for b in range(colNode.max_num_children):
                    if bfTreeNodeHasChild(nj_node, b):
                        child_node = bfTreeNodeGetChild(nj_node, b)
                        nbj = reify_tree_node(child_node)
                        children.append(
                            _build_ff_block(tm, nai, nbj,
                                            tol, eta,
                                            leaf_max, leaf_min,
                                            min_svd_size, max_svd_rank_frac))
    elif not leafI:
        for a in range(rowNode.max_num_children):
            if bfTreeNodeHasChild(ni_node, a):
                child_node = bfTreeNodeGetChild(ni_node, a)
                nai = reify_tree_node(child_node)
                children.append(
                    _build_ff_block(tm, nai, colNode,
                                    tol, eta,
                                    leaf_max, leaf_min,
                                    min_svd_size, max_svd_rank_frac))
    else:  # not leafJ
        for b in range(colNode.max_num_children):
            if bfTreeNodeHasChild(nj_node, b):
                child_node = bfTreeNodeGetChild(nj_node, b)
                nbj = reify_tree_node(child_node)
                children.append(
                    _build_ff_block(tm, rowNode, nbj,
                                    tol, eta,
                                    leaf_max, leaf_min,
                                    min_svd_size, max_svd_rank_frac))

    # Filter expected any empty children
    children = [c for c in children if c is not None]
    if not children:
        if ff_debug:
            printf("[ff-debug] all children empty -> returning None\n")
            fflush(stdout)
        return None

    if ff_debug:
        printf("[ff-debug] built FFNode with %d children\n", <int> len(children))
        fflush(stdout)

    return FFNode(children)

cdef class HierarchicalFormFactor:
    """
    High-level wrapper for a hierarchical FF operator:

        y = H.apply(x) ≈ F x

    where F is the full form-factor matrix.
    """

    def __init__(self, FFBlock root, BfSize n):
        self.root = root
        self.n = n
        self.sparse_leaves = []
        self.svd_leaves = []
        _collect_leaves(root, self.sparse_leaves, self.svd_leaves)

    @property
    def root_block(self):
        """Python-visible access to the root FFBlock."""
        return self.root

    cpdef cnp.ndarray apply(self, cnp.ndarray x):
        cdef cnp.ndarray x_flat = np.ascontiguousarray(x, dtype=np.float64)
        if x_flat.ndim != 1 or x_flat.shape[0] != self.n:
            raise ValueError(f"x must be 1D of length {self.n}, got {(<object> x_flat).shape}")

        cdef cnp.ndarray y = np.zeros_like(x_flat)
        cdef FFBlock leaf

        # Sparse leaves
        for leaf in self.sparse_leaves:
            (<FFSparseLeaf> leaf).apply(x_flat, y)

        # SVD leaves
        for leaf in self.svd_leaves:
            (<FFSvdLeaf> leaf).apply(x_flat, y)

        return y

cdef Mat _dense_from_csr_block(MatCsrReal A):
    """
    Build a dense Mat from a CSR block A, assuming its column indices
    are already local in [0, nA).

    This is correct after `_reindex_csr_block_columns`, which enforces
    0 <= colind < numCols.
    """
    cdef BfSize mA = bfMatGetNumRows(A.mat)
    cdef BfSize nA = bfMatGetNumCols(A.mat)

    cdef const BfSize *rp = bfMatCsrRealGetRowptrConstPtr(A.mat_csr_real)
    cdef const BfSize *ci = bfMatCsrRealGetColindConstPtr(A.mat_csr_real)
    cdef const BfReal *da = bfMatCsrRealGetDataConstPtr(A.mat_csr_real)

    cdef BfSize nnz = rp[mA]

    # Allocate dense (mA x nA)
    cdef cnp.ndarray[double, ndim=2] dense_mv = \
        np.zeros((mA, nA), dtype=np.float64)

    cdef BfSize i, k, row_start, row_end, j

    if ff_debug:
        printf("[ff-debug] _dense_from_csr_block: mA=%zu nA=%zu nnz=%zu\n",
               mA, nA, nnz)
        fflush(stdout)

    for i in range(mA):
        row_start = rp[i]
        row_end = rp[i + 1]

        if row_end < row_start or row_end > nnz:
            if ff_debug:
                printf("[ff-debug] CSR row %zu has invalid rowptr: "
                       "row_start=%zu row_end=%zu nnz=%zu\n",
                       i, row_start, row_end, nnz)
                fflush(stdout)
            continue

        for k in range(row_start, row_end):
            j = ci[k]
            if j >= nA:
                if ff_debug:
                    printf("[ff-debug] _dense_from_csr_block: "
                           "col index %zu >= nA=%zu\n", j, nA)
                    fflush(stdout)
                continue
            dense_mv[i, j] += da[k]

    # Optional NaN/Inf diagnostics
    if np.isnan(dense_mv).any() or np.isinf(dense_mv).any():
        if ff_debug:
            printf("[ff-debug] _dense_from_csr_block produced NaN/Inf block: "
                   "mA=%zu nA=%zu\n", mA, nA)
            fflush(stdout)
            mn = float(np.nanmin(dense_mv))
            mx = float(np.nanmax(dense_mv))
            printf("[ff-debug] block stats: min=%g max=%g\n", mn, mx)
            fflush(stdout)

    return Mat.from_ndarray(dense_mv)

cdef void _reindex_csr_block_columns(MatCsrReal A,
                                     SizeArray J_inds):
    """
    In-place remap A.mat_csr_real->colind from *global* face indices
    to local indices in [0, mj), where mj = len(J_inds).

    After this, the CSR block is consistent with BF's expectation
    that 0 <= colind < numCols, and we can safely do:

        xsub = x[col_idx]           # len = mj
        ysub = A @ xsub             # BF CSR MVP
    """
    cdef cnp.ndarray J_np
    cdef BfSize mj, mA, nnz
    cdef const BfSize *rp_const
    cdef BfSize *ci
    cdef BfSize k
    cdef BfSize j_global
    cdef Py_ssize_t j_local_py
    cdef dict col_map = {}

    # Build numpy copy of J_inds and a Python dict: global -> local
    J_np = _sizearray_to_numpy(J_inds)  # dtype=uintp, length mj
    mj = <BfSize> J_np.shape[0]

    for k in range(mj):
        # J_np[k] is a global face index; local index is k
        col_map[int(J_np[k])] = k

    # Raw CSR layout
    mA = bfMatGetNumRows(A.mat)
    rp_const = bfMatCsrRealGetRowptrConstPtr(A.mat_csr_real)
    ci = <BfSize *> bfMatCsrRealGetColindConstPtr(A.mat_csr_real)
    nnz = rp_const[mA]

    if ff_debug:
        printf("[ff-debug] _reindex_csr_block_columns: mA=%zu mj=%zu nnz=%zu\n",
               mA, mj, nnz)
        fflush(stdout)

    for k in range(nnz):
        j_global = ci[k]
        j_local_py = <Py_ssize_t> col_map.get(int(j_global), -1)

        if j_local_py < 0 or j_local_py >= <Py_ssize_t> mj:
            if ff_debug:
                printf("[ff-debug] _reindex_csr_block_columns: "
                       "j_global=%zu not in J_inds (mj=%zu)\n",
                       j_global, mj)
                fflush(stdout)
            # This indicates a logic bug between builder and index sets,
            # so fail fast rather than silently corrupting MVP.
            raise RuntimeError(
                f"_reindex_csr_block_columns: j_global={j_global} "
                f"not in J_inds (mj={mj})"
            )

        ci[k] = <BfSize> j_local_py

cdef void _collect_leaves(FFBlock block,
                          list sparse_leaves,
                          list svd_leaves):
    cdef FFNode node

    if isinstance(block, FFSparseLeaf):
        sparse_leaves.append(block)
        return
    if isinstance(block, FFSvdLeaf):
        svd_leaves.append(block)
        return
    if isinstance(block, FFNode):
        node = <FFNode> block
        for child in node.children:
            _collect_leaves(child, sparse_leaves, svd_leaves)
        return
    raise RuntimeError("Unknown FFBlock subclass")

## C hier impl

cdef class Octree:
    cdef BfOctree *octree

    def __cinit__(self):
        self.octree = NULL

    def __dealloc__(self):
        if self.octree != NULL:
            bfOctreeDelete(&self.octree)
            self.octree = NULL


cdef class VfHier:

    def __cinit__(self):
        self.vfHier = NULL
        self._n = 0
        self._tree_owner = None

    cdef inline void _apply_ptr(self,
                                const double *x,
                                double *y) noexcept nogil:
        """
        Low-level apply that assumes:
          - self.vfHier != NULL
          - x and y point to length-n buffers (n = self.n)
          - y is already allocated
        Runs nogil so bfVfHierApply can use OpenMP, and so callers can
        call it from prange/nogil regions.
        """
        bfVfHierApply(self.vfHier, <BfReal *> x, <BfReal *> y)

    @staticmethod
    def from_trimesh(Trimesh tm,
                     *,
                     topology="auto",
                     object quadtree=None,
                     object octree=None,
                     double eta=2.0,
                     BfSize leaf_max=128,
                     BfSize leaf_min=1,
                     BfReal min_area=0.0,
                     double tol=1e-2,
                     BfSize min_svd_size=16384,
                     double max_svd_rank_frac=0.9,
                     bint ensure_geometry=True,
                     bint init_embree_scene=True):
        """
        Build a compressed (hierarchical/SVD) C-side form-factor operator.

        Parameters
        ----------
        tm : Trimesh
            The mesh used for VF computation.
        quadtree : Quadtree or None
            If provided, use this quadtree instead of building one from tm.
            This enables the "flux-style" split: quadtree from projected coords,
            flux computations on 3D coords, as long as face ordering matches.
        """
        cdef VfHier H = VfHier.__new__(VfHier)
        cdef Quadtree qt
        cdef Octree ot

        if ensure_geometry:
            tm.ensure_face_geometry()
        if init_embree_scene:
            tm.init_embree()

        # Normalize topology string
        cdef object topo = topology
        if topo is None:
            topo = "auto"
        topo = (<str> topo).lower()

        # Priority: explicit tree objects win
        if quadtree is not None:
            if not isinstance(quadtree, Quadtree):
                raise TypeError("VfHier.from_trimesh: quadtree must be a butterfly.Quadtree (or None)")
            qt = <Quadtree> quadtree
            H._tree_owner = qt
            H.vfHier = bfVfHierNewFromQuadtree(
                tm.trimesh, qt.quadtree,
                eta, leaf_max, leaf_min, min_area,
                tol, min_svd_size, max_svd_rank_frac)

        elif octree is not None:
            if not isinstance(octree, Octree):
                raise TypeError("VfHier.from_trimesh: octree must be a butterfly.Octree (or None)")
            ot = <Octree> octree
            H._tree_owner = ot
            H.vfHier = bfVfHierNewFromOctree(
                tm.trimesh, ot.octree,
                eta, leaf_max, leaf_min, min_area,
                tol, min_svd_size, max_svd_rank_frac)

        else:
            # Backward compatible default: "auto" -> quadtree-from-XY
            if topo in ("auto", "quadtree", "quad", "2d"):
                qt = _quadtree_from_trimesh_xy(tm)
                H._tree_owner = qt
                H.vfHier = bfVfHierNewFromQuadtree(
                    tm.trimesh, qt.quadtree,
                    eta, leaf_max, leaf_min, min_area,
                    tol, min_svd_size, max_svd_rank_frac)
            elif topo in ("octree", "oct", "3d"):
                # Let C build internal octree from face centroids
                H.vfHier = bfVfHierNewFromTrimeshAndAutoTree(
                    tm.trimesh,
                    BF_VF_TOPO_OCTREE,
                    <BfQuadtree *> NULL,
                    <BfOctree *> NULL,
                    eta, leaf_max, leaf_min, min_area,
                    tol, min_svd_size, max_svd_rank_frac)
            else:
                raise ValueError("topology must be one of: 'auto', 'quadtree', 'octree'")

        if H.vfHier == NULL:
            raise RuntimeError("bfVfHier constructor failed")

        H._n = tm.num_faces
        return H

    @staticmethod
    def from_quadtree(Trimesh tm,
                      Quadtree quadtree,
                      double eta=2.0,
                      BfSize leaf_max=128,
                      BfSize leaf_min=1,
                      BfReal min_area=0.0,
                      double tol=1e-2,
                      BfSize min_svd_size=16384,
                      double max_svd_rank_frac=0.9,
                      bint ensure_geometry=True,
                      bint init_embree_scene=True):
        """
        Explicit constructor taking a pre-built Quadtree.

        This matches the API your illum_common.py expects:
            VfHier.from_quadtree(bf_tm, quadtree, **kwargs)
        """
        return VfHier.from_trimesh(tm,
                                   quadtree=quadtree,
                                   eta=eta,
                                   leaf_max=leaf_max,
                                   leaf_min=leaf_min,
                                   min_area=min_area,
                                   tol=tol,
                                   min_svd_size=min_svd_size,
                                   max_svd_rank_frac=max_svd_rank_frac,
                                   ensure_geometry=ensure_geometry,
                                   init_embree_scene=init_embree_scene)

    @staticmethod
    def from_csr_and_trimesh(MatCsrReal Afull,
                             Trimesh tm,
                             double eta=2.0,
                             BfSize leaf_max=128,
                             BfSize leaf_min=1,
                             BfReal min_area=0.0,
                             double tol=1e-2,
                             BfSize min_svd_size=16384,
                             double max_svd_rank_frac=0.9):
        """
        Build a compressed C-side VF hierarchy from a *full* CSR FF matrix
        and a quadtree built from the stereographic mesh.

        This uses bfVfHierNewFromCsrAndQuadtree, so no additional ray-tracing
        is performed beyond whatever was used to build Afull.
        """
        cdef VfHier H = VfHier.__new__(VfHier)
        cdef Quadtree qt = _quadtree_from_trimesh_xy(tm)

        # Keep alive for same reason as from_trimesh
        H._tree_owner = qt

        H.vfHier = bfVfHierNewFromCsrAndQuadtree(
            Afull.mat_csr_real,
            qt.quadtree,
            eta,
            leaf_max,
            leaf_min,
            min_area,
            tol,
            min_svd_size,
            max_svd_rank_frac)

        if H.vfHier == NULL:
            raise RuntimeError("bfVfHierNewFromCsrAndQuadtree failed")

        H._n = tm.num_faces
        return H

    @staticmethod
    def from_csr_and_octree(MatCsrReal Afull,
                            object octree,
                            Trimesh tm,
                            double eta=2.0,
                            BfSize leaf_max=128,
                            BfSize leaf_min=1,
                            BfReal min_area=0.0,
                            double tol=1e-2,
                            BfSize min_svd_size=16384,
                            double max_svd_rank_frac=0.9):
        cdef VfHier H = VfHier.__new__(VfHier)
        cdef Octree ot
        if not isinstance(octree, Octree):
            raise TypeError("octree must be a butterfly.Octree")
        ot = <Octree> octree
        H._tree_owner = ot

        H.vfHier = bfVfHierNew()
        if H.vfHier == NULL:
            raise RuntimeError("bfVfHierNew failed")

        bfVfHierInitFromCsrAndAutoTree(
            H.vfHier,
            Afull.mat_csr_real,
            <BfQuadtree *> NULL,
            ot.octree,
            eta,
            leaf_max,
            leaf_min,
            min_area,
            tol,
            min_svd_size,
            max_svd_rank_frac)

        if H.vfHier == NULL:
            raise RuntimeError("bfVfHierNewFromCsrAndOctree failed")
        H._n = tm.num_faces
        return H

    def get_stats(self):
        """
        Return C-side VfHier statistics as a Python dict.

        Keys:
          - num_sparse_leaves
          - num_svd_leaves
          - num_nodes
          - nnz_sparse_total
          - mem_bytes_sparse
          - mem_bytes_svd
          - mem_bytes_total
          - rank_total
        """
        cdef BfVfHierStats s
        bfVfHierCollectStats(self.vfHier, &s)

        return {
            "num_sparse_leaves": int(s.numSparseLeaves),
            "num_svd_leaves": int(s.numSvdLeaves),
            "num_nodes": int(s.numNodeBlocks),
            "nnz_sparse_total": int(s.nnzSparseTotal),
            "mem_bytes_sparse": int(s.memBytesSparseEst),
            "mem_bytes_svd": int(s.memBytesSvdEst),
            "mem_bytes_total": int(s.memBytesSparseEst + s.memBytesSvdEst),
            "rank_total": int(s.rankTotal),
        }

    cpdef dump_leaf_blocks(self):
        """
        Dump leaf rectangles in perm-index space.

        Returns:
          row_i0, row_i1, col_j0, col_j1 : np.ndarray uintp, shape (nleaf,)
          kind : np.ndarray uint8, shape (nleaf,)   (0=sparse,1=svd,2=none)
          rank : np.ndarray uintp, shape (nleaf,)   (0 unless svd)
          nnz  : np.ndarray uint64, shape (nleaf,)  (CSR nnz if sparse, 0 if svd)
        """
        import numpy as np
        cdef BfSize nleaf

        if self.vfHier == NULL:
            raise ValueError("VfHier is NULL")

        nleaf = bfVfHierGetNumLeafBlocks(self.vfHier)

        cdef cnp.ndarray row_i0 = np.empty((<Py_ssize_t> nleaf,), dtype=np.uintp)
        cdef cnp.ndarray row_i1 = np.empty((<Py_ssize_t> nleaf,), dtype=np.uintp)
        cdef cnp.ndarray col_j0 = np.empty((<Py_ssize_t> nleaf,), dtype=np.uintp)
        cdef cnp.ndarray col_j1 = np.empty((<Py_ssize_t> nleaf,), dtype=np.uintp)
        cdef cnp.ndarray kind = np.empty((<Py_ssize_t> nleaf,), dtype=np.uint8)
        cdef cnp.ndarray rank = np.empty((<Py_ssize_t> nleaf,), dtype=np.uintp)
        cdef cnp.ndarray nnz = np.empty((<Py_ssize_t> nleaf,), dtype=np.uint64)

        bfVfHierDumpLeafBlocks(self.vfHier,
                               <BfSize *> row_i0.data,
                               <BfSize *> row_i1.data,
                               <BfSize *> col_j0.data,
                               <BfSize *> col_j1.data,
                               <unsigned char *> kind.data,
                               <BfSize *> rank.data,
                               <unsigned long long *> nnz.data)

        return row_i0, row_i1, col_j0, col_j1, kind, rank, nnz

    cpdef cnp.ndarray apply_vec(self, cnp.ndarray x):
        cdef cnp.ndarray x_flat = np.ascontiguousarray(x, dtype=np.float64)
        if x_flat.ndim != 1 or x_flat.shape[0] != self._n:
            raise ValueError(f"x must be 1D of length {self._n}")

        cdef cnp.ndarray y = np.zeros_like(x_flat)

        with nogil:
            self._apply_ptr(<const double *> x_flat.data,
                            <double *> y.data)
        return y

    cpdef apply_inplace(self, cnp.ndarray x, cnp.ndarray y, bint accumulate=False):
        cdef cnp.ndarray x_flat = np.ascontiguousarray(x, dtype=np.float64)
        if x_flat.ndim != 1 or x_flat.shape[0] != self._n:
            raise ValueError(f"x must be 1D of length {self._n}")

        if y.dtype != np.float64 or not y.flags.c_contiguous:
            raise ValueError("y must be float64 and C-contiguous")
        if y.ndim != 1 or y.shape[0] != self._n:
            raise ValueError(f"y must be 1D of length {self._n}")

        if not accumulate:
            y.fill(0.0)
        with nogil:
            self._apply_ptr(<const double *> x_flat.data,
                            <double *> y.data)
        return y

    cdef inline void _apply_mat_ptr(self,
                                    const double *X,
                                    double *Y,
                                    BfSize k) noexcept nogil:
        """
        Apply to a block of k RHS stored column-major (Fortran):
          X is (n, k) with leading dimension ldX = n
          Y is (n, k) with leading dimension ldY = n
        """
        bfVfHierApplyMany(self.vfHier,
                          <const BfReal *> X, <BfSize> self._n,
                          <BfReal *> Y, <BfSize> self._n,
                          k)

    cpdef cnp.ndarray apply_mat(self, cnp.ndarray X):
        """
        Return Y = H * X for X shaped (n, k). Uses the C “many RHS” path.

        Enforces Fortran order so the C side can treat X,Y as column-major
        and call GEMM-friendly kernels.
        """
        cdef cnp.ndarray X2 = np.asarray(X, dtype=np.float64)
        if X2.ndim != 2:
            raise ValueError("apply_mat expects a 2D array (n, k)")
        if X2.shape[0] != self._n:
            raise ValueError(
                f"X must have shape (n, k) with n={self.n}; got {(<object> X2).shape}"
            )

        # Force column-major for best kernels and simplest C contract
        cdef cnp.ndarray Xf = np.asfortranarray(X2, dtype=np.float64)
        cdef Py_ssize_t k_py = Xf.shape[1]
        cdef BfSize k = <BfSize> k_py

        # IMPORTANT: C side accumulates into Y (+=), so start from zeros.
        cdef cnp.ndarray Yf = np.empty((self._n, k_py), dtype=np.float64, order='F')

        with nogil:
            self._apply_mat_ptr(<const double *> Xf.data,
                                <double *> Yf.data,
                                k)
        return Yf

    cpdef apply_mat_inplace(self, cnp.ndarray X, cnp.ndarray Y, bint accumulate=False):
        """
        In-place: Y[:] = H * X  (or Y += ... if your C function accumulates).
        Requires Fortran-contiguous X and Y (or we copy X).
        """
        cdef cnp.ndarray X2 = np.asarray(X, dtype=np.float64)
        if X2.ndim != 2 or X2.shape[0] != self.n:
            raise ValueError(f"X must be 2D with shape (n,k), n={self.n}")
        if Y.dtype != np.float64 or Y.ndim != 2 or Y.shape[0] != self.n:
            raise ValueError("Y must be float64 2D with shape (n,k)")
        if Y.shape[1] != X2.shape[1]:
            raise ValueError("X and Y must have same number of RHS (k)")

        cdef cnp.ndarray Xf = X2 if X2.flags.f_contiguous else np.asfortranarray(X2)
        if not Y.flags.f_contiguous:
            raise ValueError("Y must be Fortran-contiguous for apply_mat_inplace")

        cdef BfSize k = <BfSize> Xf.shape[1]

        if not accumulate:
            Y.fill(0.0)
        with nogil:
            self._apply_mat_ptr(<const double *> Xf.data,
                                <double *> Y.data,
                                k)
        return Y

    cpdef apply(self, object x):
        """
        Convenience overload:
          - 1D -> vector apply
          - 2D -> block apply
        """
        cdef cnp.ndarray arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            return self.apply_vec(arr)  # make a private helper if you want
        elif arr.ndim == 2:
            return self.apply_mat(arr)
        else:
            raise ValueError("apply expects 1D or 2D array")

    @staticmethod
    cdef VfHier from_ptr(BfVfHier *ptr):
        cdef VfHier obj = VfHier.__new__(VfHier)
        obj.vfHier = ptr
        obj._n = bfVfHierGetNumFaces(ptr)
        return obj

    def __dealloc__(self):
        if self.vfHier != NULL:
            bfVfHierDeinitAndDealloc(&self.vfHier)
            self.vfHier = NULL

    def save(self, path):
        """
        Save hierarchy to a single binary file.
        """
        if self.vfHier == NULL:
            raise ValueError("VfHier is NULL")

        cdef bytes bpath = str(path).encode("utf-8")
        cdef const char *cpath = bpath

        if not bfVfHierSave(self.vfHier, cpath):
            raise OSError(f"bfVfHierSave failed for {path}")

    @staticmethod
    def load(path):
        """
        Load hierarchy from a single binary file.
        """
        cdef bytes bpath = str(path).encode("utf-8")
        cdef const char *cpath = bpath

        cdef BfVfHier *ptr = bfVfHierLoad(cpath)
        if ptr == NULL:
            raise OSError(f"bfVfHierLoad failed for {path}")
        return VfHier.from_ptr(ptr)

    @property
    def n(self):
        """
        Backward-compatible alias used by downstream code:
        number of faces / DOFs of the operator.
        """
        return <Py_ssize_t> self._n

    @property
    def num_faces(self):
        # Keep existing API; prefer cached _n when available.
        if self.vfHier == NULL:
            return 0
        return <Py_ssize_t> self._n

