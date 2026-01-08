# cython: language_level=3, embedsignature=True, auto_pickle=False

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()

from enum import Enum

from bf cimport bfInit

bfInit()

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

def seed(BfSize seed):
    bfSeed(seed)

cdef reify_mat(BfMat *mat):
    cdef BfType type_ = bfMatGetType(mat)
    if type_ == BF_TYPE_MAT:
        raise RuntimeError()
    elif type_ == BF_TYPE_MAT_DIFF:
        return MatDiff.from_ptr(bfMatToMatDiff(mat))
    elif type_ == BF_TYPE_MAT_PRODUCT:
        return MatProduct.from_ptr(bfMatToMatProduct(mat))
    elif type_ == BF_TYPE_MAT_PYTHON:
        return MatPython.from_ptr(bfMatToMatPython(mat))
    elif type_ == BF_TYPE_MAT_BLOCK_DENSE:
        return MatBlockDense.from_ptr(bfMatToMatBlockDense(mat))
    elif type_ == BF_TYPE_MAT_DENSE_COMPLEX:
        return MatDenseComplex.from_ptr(bfMatToMatDenseComplex(mat))
    elif type_ == BF_TYPE_MAT_DENSE_REAL:
        return MatDenseReal.from_ptr(bfMatToMatDenseReal(mat))
    # elif type_ == BF_TYPE_MAT_VF_HIER:
    #     # No dedicated wrapper; just treat it as a generic Mat.
    #     cdef Mat _ = Mat.__new__(Mat)
    #     _.mat = mat
    #     return _
    else:
        raise TypeError(f'failed to reify BfMat: got {type_}')

cdef reify_vec(BfVec *vec):
    cdef BfType type_ = bfVecGetType(vec)
    if type_ == BF_TYPE_VEC:
        raise RuntimeError()
    elif type_ == BF_TYPE_VEC_REAL:
        return VecReal.from_ptr(bfVecToVecReal(vec))
    else:
        raise TypeError(f'failed to reify BfVec: got {type_}')

cdef reify_tree(BfTree *tree):
    cdef BfType type_ = bfTreeGetType(tree)
    if type_ == BF_TYPE_QUADTREE:
        return Quadtree.from_ptr(bfTreeToQuadtree(tree))
    else:
        raise TypeError(f'failed to reify BfTree: got {type_}')

cdef reify_tree_node(BfTreeNode *treeNode):
    cdef BfType type_ = bfTreeNodeGetType(treeNode)
    if type_ == BF_TYPE_QUADTREE_NODE:
        return QuadtreeNode.from_ptr(bfTreeNodeToQuadtreeNode(treeNode))
    else:
        raise TypeError(f'failed to reify BfTreeNode: got {type_}')

cdef class Bbox2:
    cdef BfBbox2 bbox

    def __cinit__(self, xmin, xmax, ymin, ymax):
        self.bbox.min[0] = xmin
        self.bbox.max[0] = xmax
        self.bbox.min[1] = ymin
        self.bbox.max[1] = ymax

    @property
    def xy(self):
        return (self.xmin, self.ymin)

    @property
    def xmin(self):
        return self.bbox.min[0]

    @property
    def xmax(self):
        return self.bbox.max[0]

    @property
    def ymin(self):
        return self.bbox.min[1]

    @property
    def ymax(self):
        return self.bbox.max[1]

    @property
    def dx(self):
        return self.xmax - self.xmin

    @property
    def dy(self):
        return self.ymax - self.ymin

    def __str__(self):
        return f'[{self.xmin}, {self.xmax}] x [{self.ymin}, {self.ymax}]'

class Policy(Enum):
    View = BF_POLICY_VIEW
    Copy = BF_POLICY_COPY
    Steal = BF_POLICY_STEAL

cdef class Point2:
    cdef BfPoint2 point

    def __init__(self, BfReal x, BfReal y):
        self.point[0] = x
        self.point[1] = y

    @property
    def x(self):
        return self.point[0]

    @x.setter
    def x(self, value):
        self.point[0] = value

    @property
    def y(self):
        return self.point[1]

    @y.setter
    def y(self, value):
        self.point[1] = value

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __getitem__(self, i):
        if i != 0 and i != 1:
            raise IndexError('index should be 0 or 1')
        return self.point[i]

    def __setitem__(self, i, value):
        if i != 0 and i != 1:
            raise IndexError('index should be 0 or 1')
        self.point[i] = value

cdef class Ellipse:
    cdef BfEllipse ellipse

    def __init__(self, semi_major_axis, semi_minor_axis, center, theta):
        self.ellipse.semiMajorAxis = semi_major_axis
        self.ellipse.semiMinorAxis = semi_minor_axis
        self.ellipse.center[0] = center[0]
        self.ellipse.center[1] = center[1]
        self.ellipse.theta = theta

    @property
    def perimeter(self):
        return bfEllipseGetPerimeter(&self.ellipse)

    def sample_linspaced(self, num_points):
        X = Points2()
        T = Vectors2()
        N = Vectors2()
        W = RealArray()
        bfEllipseSampleLinspaced(
            &self.ellipse, num_points, X.points, T.vectors, N.vectors, W.real_array)
        return X, T, N, W

cdef class FacHelm2:
    def __init__(self):
        raise TypeError("FacHelm2 can't be instantiated")

    @staticmethod
    def make_multilevel(Helm2 helm, Quadtree srcTree, Quadtree tgtTree=None):
        return reify_mat(bfFacHelm2MakeMultilevel(
            &helm.helm,
            srcTree.quadtree,
            srcTree.quadtree if tgtTree is None else tgtTree.quadtree))

cdef class FacStreamer:
    cdef BfFacStreamer *facStreamer

    @staticmethod
    def from_trees(Tree rowTree, Tree colTree,
                   rowTreeInitDepth=None,
                   BfReal tol=1e-15,
                   BfSize minNumRows=20, BfSize minNumCols=20,
                   bint compareRelativeErrors=False):
        if rowTreeInitDepth is None:
            rowTreeInitDepth = 1

        cdef BfFacSpec facSpec
        facSpec.rowTree = rowTree.tree
        facSpec.colTree = colTree.tree
        facSpec.rowTreeInitDepth = rowTreeInitDepth
        facSpec.tol = tol
        facSpec.minNumRows = minNumRows
        facSpec.minNumCols = minNumCols
        facSpec.compareRelativeErrors = compareRelativeErrors

        cdef FacStreamer _ = FacStreamer.__new__(FacStreamer)
        _.facStreamer = bfFacStreamerNew()
        bfFacStreamerInit(_.facStreamer, &facSpec)
        return _

    def _feed_ndarray(self, cnp.ndarray arr):
        self._feed_Mat(Mat.from_ndarray(arr))

    def _feed_Mat(self, Mat mat):
        bfFacStreamerFeed(self.facStreamer, mat.mat)

    def feed(self, mat):
        if isinstance(mat, np.ndarray):
            self._feed_ndarray(mat)
        elif isinstance(mat, Mat):
            self._feed_Mat(mat)
        else:
            raise NotImplementedError()

    def is_done(self):
        return bfFacStreamerIsDone(self.facStreamer)

    def toMat(self):
        cdef BfFac *fac = bfFacStreamerGetFac(self.facStreamer)
        cdef MatProduct _ = MatProduct.__new__(MatProduct)
        _.matProduct = bfFacGetMatProduct(fac, BF_POLICY_STEAL)
        bfFacDeinitAndDealloc(&fac)
        _.mat = bfMatProductToMat(_.matProduct)
        return _

cdef class Helm2:
    cdef BfHelm2 helm

    def __init__(self, k, layerPot, alpha=None, beta=None):
        self.helm.k = k
        self.helm.layerPot = layerPot.value
        if layerPot == LayerPot.CombinedField:
            if alpha is None or beta is None:
                raise ValueError('alpha and beta must both be set if layerPot is CombinedField')
            self.helm.alpha.real = alpha.real
            self.helm.alpha.imag = alpha.imag
            self.helm.beta.real = beta.real
            self.helm.beta.imag = beta.imag
        else:
            if alpha is not None or beta is not None:
                raise ValueError('only pass alpha and beta when layerPot is CombinedField')

    @property
    def k(self):
        return self.helm.k

    @property
    def layer_pot(self):
        return self.helm.layerPot

    @property
    def alpha(self):
        return self.helm.alpha

    @property
    def beta(self):
        return self.helm.beta

    def get_kernel_matrix(self, Points2 Xsrc, Points2 Xtgt=None, Vectors2 Nsrc=None, Vectors2 Ntgt=None):
        return reify_mat(bfHelm2GetKernelMatrix(
            &self.helm,
            Xsrc.points,
            NULL if Xtgt is None else Xtgt.points,
            NULL if Nsrc is None else Nsrc.vectors,
            NULL if Ntgt is None else Ntgt.vectors))

    def apply_KR_correction(self, Mat mat, BfSize KR_order, Points2 X, Vectors2 N, Tree tree=None):
        if tree is None:
            bfHelm2ApplyKrCorrection(&self.helm, KR_order, X.points, N.vectors, mat.mat)
        else:
            bfHelm2ApplyKrCorrectionTree(&self.helm, KR_order, X.points, N.vectors, tree.tree, mat.mat)

    def apply_block_KR_correction(self, Mat mat, offsets, BfSize KR_order, Points2 X, Vectors2 N, Tree tree=None):
        cdef BfSizeArray *offsets_ = bfSizeArrayNewWithCapacity(len(offsets))
        for offset in offsets:
            bfSizeArrayAppend(offsets_, offset)
        if tree is None:
            bfHelm2ApplyBlockCorrection(&self.helm, offsets_, KR_order, X.points, N.vectors, mat.mat)
        else:
            bfHelm2ApplyBlockCorrectionTree(&self.helm, offsets_, KR_order, X.points, N.vectors, tree.tree, mat.mat)
        bfSizeArrayDeinitAndDealloc(&offsets_)

class LayerPot(Enum):
    Unknown = BF_LAYER_POTENTIAL_UNKNOWN
    Single = BF_LAYER_POTENTIAL_SINGLE
    PvDouble = BF_LAYER_POTENTIAL_PV_DOUBLE
    PvNormalDerivSingle = BF_LAYER_POTENTIAL_PV_NORMAL_DERIV_SINGLE
    PvNormalDerivDouble = BF_LAYER_POTENTIAL_PV_NORMAL_DERIV_DOUBLE
    CombinedField = BF_LAYER_POTENTIAL_COMBINED_FIELD

    # Aliases:
    S = Single
    Sp = PvNormalDerivSingle
    D = PvDouble
    Dp = PvNormalDerivDouble

def solve_gmres(Mat A, Mat B, Mat X0=None, float tol=1e-15, max_num_iter=None, Mat M=None):
    cdef BfSize num_iter
    Y = reify_mat(bfSolveGMRES(
        A.mat,
        B.mat,
        NULL if X0 is None else X0.mat,
        tol,
        A.shape[0] - 1 if max_num_iter is None else max_num_iter,
        &num_iter,
        NULL if M is None else M.mat))
    return Y, num_iter

cdef class Mat:
    cdef BfMat *mat

    @staticmethod
    cdef from_ndarray(cnp.ndarray arr):
        if arr.dtype == np.float64:
            return MatDenseReal.from_ndarray(arr)
        elif arr.dtype == np.complex128:
            return MatDenseComplex.from_ndarray(arr)
        else:
            raise NotImplementedError()

    @property
    def shape(self):
        return (bfMatGetNumRows(self.mat), bfMatGetNumCols(self.mat))

    def __iadd__(Mat self, Mat mat):
        bfMatAddInplace(self.mat, mat.mat)
        return self

    def _matmul_ndarray(self, cnp.ndarray arr):
        if arr.ndim == 1:
            return self._matmul_vec(Vec.from_ndarray(arr))
        elif arr.ndim == 2:
            return self._matmul_mat(Mat.from_ndarray(arr))
        else:
            raise ValueError(f"can't multiply matrix with arr ({arr.ndim = })")

    def _matmul_mat(self, Mat mat):
        return reify_mat(bfMatMul(self.mat, mat.mat))

    def _matmul_vec(self, Vec vec):
        return reify_vec(bfMatMulVec(self.mat, vec.vec))

    def __matmul__(self, mat):
        if isinstance(mat, np.ndarray):
            return self._matmul_ndarray(mat)
        elif isinstance(mat, Vec):
            return self._matmul_vec(mat)
        elif isinstance(mat, Mat):
            return self._matmul_mat(mat)
        else:
            raise NotImplementedError()

    def _rmatmul_ndarray(self, cnp.ndarray arr):
        return self._rmatmul_mat(Mat.from_ndarray(arr))

    def _rmatmul_mat(self, Mat mat):
        return reify_mat(bfMatRmul(self.mat, mat.mat))

    def __rmatmul__(self, mat):
        if isinstance(mat, np.ndarray):
            return self._rmatmul_ndarray(mat)
        elif isinstance(mat, Mat):
            return self._rmatmul_mat(mat)
        else:
            raise NotImplementedError()

    def _sub_ndarray(self, cnp.ndarray arr):
        return self._sub_mat(Mat.from_ndarray(arr))

    def _sub_mat(self, Mat mat):
        return reify_mat(bfMatSub(self.mat, mat.mat))

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            return self._sub_ndarray(other)
        elif isinstance(other, Mat):
            return self._sub_mat(other)
        else:
            raise NotImplementedError()

    def to_mat_dense_complex(self):
        return MatDenseComplex.from_ptr(
            bfMatToMatDenseComplex(
                bfMatToType(self.mat, BF_TYPE_MAT_DENSE_COMPLEX)))

    def to_mat_dense_real(self):
        """
        Convert this matrix to a BF dense-real matrix.

        This uses bfMatToType(..., BF_TYPE_MAT_DENSE_REAL) in the core
        library and wraps the resulting BfMatDenseReal* as MatDenseReal.
        """
        return MatDenseReal.from_ptr(
            bfMatToMatDenseReal(
                bfMatToType(self.mat, BF_TYPE_MAT_DENSE_REAL)))

    def transpose(self):
        bfMatTranspose(self.mat)

    def truncated_svd(self, tol=None, k=None, backend=None):
        """
        Compute a (possibly) truncated SVD of this matrix using bfGetTruncatedSvd.

        You must pass exactly one of `tol` (relative tolerance) or `k`
        (target rank). The return value is (U, S, VT, truncated),
        where `truncated` is a bool indicating whether the result was
        actually truncated (True) or not (False).
        """
        cdef BfTruncSpec truncSpec
        cdef BfMat *U_ptr = NULL
        cdef BfMatDiagReal *S_ptr = NULL
        cdef BfMat *VT_ptr = NULL
        cdef bint truncated

        # Argument checks
        if (tol is None and k is None) or (tol is not None and k is not None):
            raise ValueError("Specify exactly one of tol or k")

        # --- new bit: guard unsupported types ---
        t = bfMatGetType(self.mat)
        if t != BF_TYPE_MAT_DENSE_REAL:
            raise NotImplementedError(
                f"Mat.truncated_svd currently only supports BF_TYPE_MAT_DENSE_REAL; got {t}"
            )

        if tol is not None:
            truncSpec.usingTol = 1
            truncSpec.tol = <BfReal> tol
        else:
            truncSpec.usingTol = 0
            truncSpec.k = <BfSize> k

        # Choose backend; for now only LAPACK is implemented on the C side
        if backend is None:
            # backend_c = BF_BACKEND_LAPACK
            backend_c = BF_BACKEND_ARPACK
        else:
            backend_c = <BfBackend> backend

        truncated = bfGetTruncatedSvd(
            <const BfMat *> self.mat,
            &U_ptr,
            &S_ptr,
            &VT_ptr,
            &truncSpec,
            <BfBackend> backend_c)

        if U_ptr == NULL or S_ptr == NULL or VT_ptr == NULL:
            raise RuntimeError("bfGetTruncatedSvd failed (one of U, S, VT is NULL)")

        # Wrap outputs:
        #   - U, VT: go through reify_mat (now handles dense real)
        #   - S: MatDiagReal.from_ptr
        U = reify_mat(U_ptr)
        S = MatDiagReal.from_ptr(S_ptr)
        VT = reify_mat(VT_ptr)

        return U, S, VT, bool(truncated)

cdef class MatBlockCoo(Mat):
    cdef BfMatBlockCoo *matBlockCoo

    @staticmethod
    def from_indexed_blocks(shape, indexedBlocks, policy):
        cdef BfSize numRows, numCols
        numRows, numCols = shape

        cdef BfPtrArray indexedBlocksPtrArray
        bfInitPtrArray(&indexedBlocksPtrArray, len(indexedBlocks))

        cdef Mat block
        cdef BfSize i0, j0
        cdef BfIndexedMat *indexedMat
        for i0, j0, arg in indexedBlocks:
            if isinstance(arg, np.ndarray):
                block = Mat.from_ndarray(arg)
            else:
                raise NotImplementedError()
            indexedMat = bfIndexedMatAlloc()
            indexedMat.i0 = i0
            indexedMat.j0 = j0
            indexedMat.mat = block.mat
            bfPtrArrayAppend(&indexedBlocksPtrArray, indexedMat)

        cdef MatBlockCoo _ = MatBlockCoo.__new__(MatBlockCoo)
        _.matBlockCoo = bfMatBlockCooNewFromIndexedBlocks(numRows, numCols, &indexedBlocksPtrArray, policy.value)
        _.mat = bfMatBlockCooToMat(_.matBlockCoo)

        return _

cdef class MatBlockDense(Mat):
    cdef BfMatBlockDense *mat_block_dense

    @staticmethod
    cdef MatBlockDense from_ptr(BfMatBlockDense *mat_block_dense):
        _ = MatBlockDense()
        _.mat = bfMatBlockDenseToMat(mat_block_dense)
        _.mat_block_dense = mat_block_dense
        return _

    def scale_cols(self, RealArray real_array):
        cdef BfVec *vec = bfRealArrayGetVecView(real_array.real_array)
        bfMatBlockDenseScaleCols(self.mat_block_dense, vec)
        bfVecDelete(&vec)

    def get_blocks(self, I, J, policy=Policy.View):
        cdef BfSize mBlk = len(I)
        cdef BfSize nBlk = len(J)
        cdef BfPtrArray blocks
        bfInitPtrArray(&blocks, mBlk*nBlk)
        cdef BfSize i
        cdef BfSize j
        cdef BfMat *block
        for i in range(mBlk):
            for j in range(nBlk):
                block = bfMatBlockDenseGetBlock(self.mat_block_dense, I[i], J[j])
                bfPtrArrayAppend(&blocks, block)
        blocks_ = MatBlockDense()
        blocks_.mat_block_dense = \
            bfMatBlockDenseNewFromBlocks(mBlk, nBlk, &blocks, policy.value)
        blocks_.mat = bfMatBlockDenseToMat(blocks_.mat_block_dense)
        return blocks_

    def get_row_blocks(self, I, policy=Policy.View):
        cdef BfSize numColBlocks = bfMatBlockDenseGetNumColBlocks(self.mat_block_dense)

        cdef BfPtrArray blocks
        bfInitPtrArray(&blocks, len(I)*numColBlocks)

        cdef BfMat *block = NULL
        cdef BfSize j
        for i in I:
            for j in range(numColBlocks):
                block = bfMatBlockDenseGetBlock(self.mat_block_dense, i, j)
                bfPtrArrayAppend(&blocks, block)

        cdef MatBlockDense _ = MatBlockDense.__new__(MatBlockDense)
        _.mat_block_dense = bfMatBlockDenseNewFromBlocks(len(I), numColBlocks, &blocks, policy.value)
        _.mat = bfMatBlockDenseToMat(_.mat_block_dense)

        return _

    def get_col_blocks(self, J, policy=Policy.View):
        cdef BfSize numRowBlocks = bfMatBlockDenseGetNumRowBlocks(self.mat_block_dense)

        cdef BfPtrArray blocks
        bfInitPtrArray(&blocks, numRowBlocks*len(J))

        cdef BfMat *block = NULL
        cdef BfSize i
        for i in range(numRowBlocks):
            for j in J:
                block = bfMatBlockDenseGetBlock(self.mat_block_dense, i, j)
                bfPtrArrayAppend(&blocks, block)

        cdef MatBlockDense _ = MatBlockDense.__new__(MatBlockDense)
        _.mat_block_dense = bfMatBlockDenseNewFromBlocks(numRowBlocks, len(J), &blocks, policy.value)
        _.mat = bfMatBlockDenseToMat(_.mat_block_dense)

        return _

cdef class MatBlockDiag(Mat):
    cdef BfMatBlockDiag *matBlockDiag

    @staticmethod
    def from_blocks(blocks, policy):
        cdef MatBlockDiag _ = MatBlockDiag.__new__(MatBlockDiag)

        cdef BfPtrArray blocksPtrArray
        bfInitPtrArray(&blocksPtrArray, len(blocks))

        cdef Mat block
        for block in blocks:
            bfPtrArrayAppend(&blocksPtrArray, block.mat)

        _.matBlockDiag = bfMatBlockDiagNewFromBlocks(&blocksPtrArray, policy.value)
        _.mat = bfMatBlockDiagToMat(_.matBlockDiag)

        bfPtrArrayDeinit(&blocksPtrArray)

        return _

cdef class MatCsrReal(Mat):
    cdef BfMatCsrReal *mat_csr_real

    def to_dense_ndarray(self):
        """
        Densify this CSR matrix into a 2D NumPy float64 array.

        This is O(m*n) and meant for relatively small far-field blocks.
        """
        cdef BfSize m = bfMatGetNumRows(self.mat)
        cdef BfSize n = bfMatGetNumCols(self.mat)
        if m == 0 or n == 0:
            return np.zeros((m, n), dtype=np.float64)

        cdef const BfSize * rp = bfMatCsrRealGetRowptrConstPtr(self.mat_csr_real)
        cdef const BfSize * ci = bfMatCsrRealGetColindConstPtr(self.mat_csr_real)
        cdef const BfReal * da = bfMatCsrRealGetDataConstPtr(self.mat_csr_real)

        cdef cnp.ndarray[double, ndim=2] dense = np.zeros((m, n), dtype=np.float64)

        cdef BfSize i, k, row_start, row_end, j
        for i in range(m):
            row_start = rp[i]
            row_end = rp[i + 1]
            for k in range(row_start, row_end):
                j = ci[k]
                if j >= n:
                    if ff_debug:
                        printf("[ff-debug] MatCsrReal.to_dense_ndarray: "
                               "col index %zu out of range for shape (%zu, %zu) "
                               "(row i=%zu, k=%zu, row_start=%zu, row_end=%zu)\n",
                               j, m, n, i, k, row_start, row_end)
                        fflush(stdout)
                    continue
                dense[i, j] = da[k]

        return dense

    def to_scipy_csr(self):
        """
        Convert a BfMatCsrReal into a SciPy CSR matrix.

        This version is defensive against degenerate/invalid shapes
        coming from the BF side: if m or n are non-positive, we
        return an empty CSR of the appropriate shape instead of
        trying to build a bogus memoryview (which was triggering
        "Invalid shape in axis 0: 0" before).
        """
        import numpy as np
        import scipy.sparse as sp

        cdef BfSize m = bfMatGetNumRows(self.mat)
        cdef BfSize n = bfMatGetNumCols(self.mat)

        # Convert to Py_ssize_t to safely check for non-positive sizes
        cdef Py_ssize_t m_py = <Py_ssize_t>m
        cdef Py_ssize_t n_py = <Py_ssize_t>n

        if m_py <= 0 or n_py <= 0:
            # Degenerate or error state from BF: return an empty CSR.
            # Clamp to >= 0 just in case m/n are negative.
            if m_py < 0:
                m_py = 0
            if n_py < 0:
                n_py = 0
            return sp.csr_matrix((int(m_py), int(n_py)), dtype=np.float64)

        cdef const BfSize* rp = bfMatCsrRealGetRowptrConstPtr(self.mat_csr_real)
        if rp == NULL:
            raise RuntimeError("MatCsrReal.to_scipy_csr: rowptr is NULL")

        cdef BfSize nnz = rp[m]

        cdef const BfSize* ci = bfMatCsrRealGetColindConstPtr(self.mat_csr_real)
        if ci == NULL and nnz != 0:
            raise RuntimeError("MatCsrReal.to_scipy_csr: colind is NULL but nnz != 0")

        cdef const BfReal* da = bfMatCsrRealGetDataConstPtr(self.mat_csr_real)
        if da == NULL and nnz != 0:
            raise RuntimeError("MatCsrReal.to_scipy_csr: data is NULL but nnz != 0")

        # Copy to NumPy so SciPy owns the buffers (safe if BF frees its memory)
        rowptr = np.asarray(<BfSize[:m+1]> rp, dtype=np.uintp).copy()
        colind = np.asarray(<BfSize[:nnz]>  ci, dtype=np.uintp).copy() if nnz > 0 else np.empty(0, dtype=np.uintp)
        data   = np.asarray(<BfReal[:nnz]>  da, dtype=np.float64).copy() if nnz > 0 else np.empty(0, dtype=np.float64)

        return sp.csr_matrix((data, colind, rowptr), shape=(int(m), int(n)))


    @staticmethod
    def new_view_factor_matrix_from_trimesh(Trimesh trimesh, BfSize[::1] I=None, BfSize[::1] J=None):
        if I is None:
            I = np.arange(trimesh.num_faces, dtype=np.uintp)
        cdef BfSizeArray *rowInds = bfSizeArrayNewView(len(I), &I[0])

        if J is None:
            J = np.arange(trimesh.num_faces, dtype=np.uintp)
        cdef BfSizeArray *colInds = bfSizeArrayNewView(len(J), &J[0])

        cdef MatCsrReal _ = MatCsrReal.__new__(MatCsrReal)
        _.mat_csr_real = bfMatCsrRealNewViewFactorMatrixFromTrimesh(trimesh.trimesh, rowInds, colInds)
        _.mat = bfMatCsrRealToMat(_.mat_csr_real)

        bfSizeArrayDeinitAndDealloc(&rowInds)
        bfSizeArrayDeinitAndDealloc(&colInds)

        return _

cdef class MatDenseComplex(Mat):
    cdef BfMatDenseComplex *mat_dense_complex

    cdef size_t _buf_itemsize
    cdef Py_ssize_t[2] _buf_shape
    cdef Py_ssize_t[2] _buf_strides

    cdef void _buf_init(self):
        self._buf_itemsize = sizeof(BfComplex)
        self._buf_shape[0] = self.shape[0]
        self._buf_shape[1] = self.shape[1]
        self._buf_strides[1] = self._buf_itemsize
        self._buf_strides[0] = self.shape[1]*self._buf_strides[1]

    @staticmethod
    cdef MatDenseComplex from_ptr(BfMatDenseComplex *mat_dense_complex):
        _ = MatDenseComplex()
        _.mat = bfMatDenseComplexToMat(mat_dense_complex)
        _.mat_dense_complex = mat_dense_complex
        _._buf_init()
        return _

    @staticmethod
    cdef MatDenseComplex from_ndarray(cnp.ndarray arr):
        # Make sure arr is a packed, 2D, row-major array of complex doubles:
        assert arr.ndim == 2
        assert arr.flags.c_contiguous
        assert arr.itemsize == 16

        # cdef BfSize m = arr.shape[0]
        # cdef BfSize n = arr.shape[1]
        # cdef BfComplex[:, :] data = arr

        # cdef MatDenseComplex _ = MatDenseComplex.__new__(MatDenseComplex)
        # _.mat_dense_complex = bfMatDenseComplexNewViewFromPtr(m, n, &data[0, 0])
        # _.mat = bfMatDenseComplexToMat(_.mat_dense_complex)
        # return _

        cdef MatDenseComplex _ = MatDenseComplex.__new__(MatDenseComplex)
        _.mat_dense_complex = bfMatDenseComplexNewViewFromPyArray(arr)
        _.mat = bfMatDenseComplexToMat(_.mat_dense_complex)
        return _

    def __array__(self):
        cdef BfComplex *data = bfMatDenseComplexGetDataPtr(self.mat_dense_complex)
        return np.asarray(<BfComplex[:self.shape[0], :self.shape[1]]>data)

    def __getbuffer__(self, Py_buffer *buf, int flags):
        buf.buf = <char *>bfMatDenseComplexGetDataPtr(self.mat_dense_complex)
        buf.format = 'Zd'
        buf.internal = <void *>NULL
        buf.itemsize = self._buf_itemsize
        buf.len = self._buf_shape[0]*self._buf_shape[1]*self._buf_itemsize
        buf.ndim = 2
        buf.obj = self
        buf.readonly = 1
        buf.shape = self._buf_shape
        buf.strides = self._buf_strides
        buf.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buf):
        pass

cdef class MatDenseReal(Mat):
    cdef BfMatDenseReal *matDenseReal

    @staticmethod
    def from_ndarray(cnp.ndarray arr):
        """
        Wrap a 2D C-contiguous float64 NumPy array as a MatDenseReal
        using bfMatDenseRealNewViewFromPtr.

        The BF matrix *views* the NumPy data; it does not own it.
        """
        assert arr.ndim == 2
        assert arr.flags.c_contiguous
        assert arr.dtype == np.float64

        cdef BfSize m = arr.shape[0]
        cdef BfSize n = arr.shape[1]
        cdef BfReal[:, :] data = arr

        cdef MatDenseReal _ = MatDenseReal.__new__(MatDenseReal)
        _.matDenseReal = bfMatDenseRealNewViewFromPtr(m, n, &data[0, 0], n, 1)
        _.mat = bfMatDenseRealToMat(_.matDenseReal)
        return _

    @staticmethod
    cdef from_ptr(BfMatDenseReal *matDenseReal):
        """
        Wrap an existing BfMatDenseReal* as MatDenseReal.
        """
        cdef MatDenseReal _ = MatDenseReal.__new__(MatDenseReal)
        _.matDenseReal = matDenseReal
        _.mat = bfMatDenseRealToMat(_.matDenseReal)
        return _

cdef class MatDiagReal(Mat):
    cdef BfMatDiagReal *matDiagReal

    @staticmethod
    cdef MatDiagReal from_constant(BfSize m, BfSize n, BfReal diag_value):
        mat_diag_real = MatDiagReal()
        mat_diag_real.matDiagReal = bfMatDiagRealNewConstant(m, n, diag_value)
        mat_diag_real.mat = bfMatDiagRealToMat(mat_diag_real.matDiagReal)
        return mat_diag_real

    @staticmethod
    cdef from_ptr(BfMatDiagReal *matDiagReal):
        """
        Wrap an existing BfMatDiagReal* as MatDiagReal.
        """
        cdef MatDiagReal _ = MatDiagReal.__new__(MatDiagReal)
        _.matDiagReal = matDiagReal
        _.mat = bfMatDiagRealToMat(matDiagReal)
        return _

cdef class MatDiff(Mat):
    cdef BfMatDiff *matDiff

    def __init__(self, Mat first, Mat second, policy=Policy.View):
        self.matDiff = bfMatDiffNew(first.mat, second.mat, policy.value)
        self.mat = bfMatDiffToMat(self.matDiff)

    @staticmethod
    cdef from_ptr(BfMatDiff *matDiff):
        cdef MatDiff _ = MatDiff.__new__(MatDiff)
        _.matDiff = matDiff
        _.mat = bfMatDiffToMat(_.matDiff)
        return _

    @property
    def first(self):
        return reify_mat(bfMatDiffGetFirst(self.matDiff))

    @property
    def second(self):
        return reify_mat(bfMatDiffGetSecond(self.matDiff))

    def get_blocks(self, I, J, policy=Policy.View):
        firstBlocks = self.first.get_blocks(I, J, policy)
        secondBlocks = self.second.get_blocks(I, J, policy)
        return MatDiff(firstBlocks, secondBlocks, policy)

cdef class MatFunc(Mat):
    cdef BfMatFunc *MatFunc

cdef class MatIdentity(Mat):
    cdef BfMatIdentity *matIdentity

    def __cinit__(self, BfSize n):
        self.matIdentity = bfMatIdentityNew()
        self.mat = bfMatIdentityToMat(self.matIdentity)
        bfMatIdentityInit(self.matIdentity, n)

    def __truediv__(self, BfReal denom):
        m, n = self.shape
        return MatDiagReal.from_constant(m, n, 1/denom)

cdef class MatProduct(Mat):
    cdef BfMatProduct *matProduct
    cdef list _factors

    def __cinit__(self):
        self.matProduct = bfMatProductNew()
        self.mat = bfMatProductToMat(self.matProduct)
        bfMatProductInit(self.matProduct)

    @staticmethod
    cdef from_ptr(BfMatProduct *matProduct):
        cdef MatProduct _ = MatProduct.__new__(MatProduct)
        _.matProduct = matProduct
        _.mat = bfMatProductToMat(_.matProduct)
        return _

    @staticmethod
    def from_factors(*factors):
        matProduct = MatProduct()
        for factor in factors:
            matProduct.post_multiply(factor)
        return matProduct

    @property
    def factors(self):
        if not self._factors:
            self._factors = [
                reify_mat(bfMatProductGetFactor(self.matProduct, i))
                for i in range(bfMatProductNumFactors(self.matProduct))]
        return self._factors

    def get_blocks(self, I, J, policy=Policy.View):
        if len(self.factors) == 0:
            raise RuntimeError()
        elif len(self.factors) == 1:
            newFactors = [self.factors[0].get_blocks(I, J, policy)]
        else:
            newFactors = [self.factors[0].get_row_blocks(I, policy)] \
                + self.factors[1:-1] + [self.factors[-1].get_col_blocks(J, policy)]
        return MatProduct.from_factors(*newFactors)

    cdef post_multiply(self, Mat mat):
        assert mat.mat != NULL
        bfMatProductPostMultiply(self.matProduct, mat.mat)

cdef class NodeSpan:
    cdef BfNodeSpan *nodeSpan

cdef class Perm:
    cdef BfPerm *perm

    def __getitem__(self, i):
        return bfPermGetIndex(self.perm, i)

    def get_reverse(self):
        perm = Perm()
        perm.perm = bfPermGetReversePerm(self.perm)
        return perm

cdef class Points2:
    cdef BfPoints2 *points

    cdef Py_ssize_t[2] shape
    cdef Py_ssize_t[2] strides

    def __init__(self, BfSize capacity=16):
        self.points = bfPoints2NewWithCapacity(capacity)

    @staticmethod
    def from_point(point):
        if len(point) != 2:
            raise ValueError('len(point) != 2')
        cdef Points2 points = Points2.__new__(Points2)
        points.points = bfPoints2NewWithCapacity(1)
        cdef BfPoint2 point_ = [point[0], point[1]]
        bfPoints2Append(points.points, point_)
        return points

    @staticmethod
    def sample_poisson_disk(Bbox2 bbox, BfReal min_dist, BfSize k=30):
        cdef Points2 points = Points2.__new__(Points2)
        points.points = bfPoints2SamplePoissonDisk(&bbox.bbox, min_dist, k)
        return points

    def __getbuffer__(self, Py_buffer *buf, int flags):
        itemsize = sizeof(BfReal)
        num_points = len(self)

        self.shape[0] = num_points
        self.shape[1] = 2

        self.strides[1] = itemsize
        self.strides[0] = 2*self.strides[1]

        buf.buf = <char *>bfPoints2GetDataPtr(self.points)
        buf.format = 'd'
        buf.internal = NULL
        buf.itemsize = itemsize
        buf.len = self.shape[0]*self.shape[1]*itemsize
        buf.ndim = 2
        buf.obj = self
        buf.readonly = 1
        buf.shape = self.shape
        buf.strides = self.strides
        buf.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buf):
        pass

    def __len__(self):
        return bfPoints2GetSize(self.points)

    def __getitem__(self, i):
        cdef BfPoint2 point
        if i > len(self):
            raise IndexError()
        bfPoints2Get(self.points, i, point)
        return Point2(point[0], point[1])

    def extend(self, Points2 points):
        bfPoints2Extend(self.points, points.points)

cdef class Quadtree(Tree):
    cdef BfQuadtree *quadtree

    def __init__(self):
        raise RuntimeError("use factory functions to instantiate Quadtree")

    @staticmethod
    def from_tree(Tree tree):
        cdef Quadtree _ = Quadtree.__new__(Quadtree)
        _.tree = tree.tree
        _.quadtree = bfTreeToQuadtree(_.tree)
        return _

    @staticmethod
    cdef from_ptr(BfQuadtree *quadtree):
        cdef Quadtree _ = Quadtree.__new__(Quadtree)
        _.quadtree = quadtree
        _.tree = bfQuadtreeToTree(_.quadtree)
        return _

    @staticmethod
    def from_points_and_normals(Points2 points, Vectors2 normals):
        cdef Quadtree quadtree = Quadtree.__new__(Quadtree)
        quadtree.quadtree = bfQuadtreeNew()
        bfQuadtreeInit(quadtree.quadtree, points.points, normals.vectors)
        quadtree.tree = bfQuadtreeToTree(quadtree.quadtree)
        return quadtree

    def plot_node_boxes(self, ax=None):
        if ax is None:
            ax = plt.gca()
        rects = []
        for node in self.nodes:
            bbox = node.bbox
            rect = Rectangle(bbox.xy, bbox.dx, bbox.dy)
            rects.append(rect)
        pc = PatchCollection(rects, facecolor='none', edgecolor='k')
        ax.add_collection(pc)

cdef class QuadtreeNode(TreeNode):
    cdef BfQuadtreeNode *quadtreeNode

    @staticmethod
    cdef from_ptr(BfQuadtreeNode *quadtreeNode):
        _ = QuadtreeNode()
        _.quadtreeNode = quadtreeNode
        _.treeNode = bfQuadtreeNodeToTreeNode(quadtreeNode)
        return _

    @property
    def split(self):
        cdef BfPoint2 split
        bfQuadtreeNodeGetSplit(self.quadtreeNode, split)
        return (split[0], split[1])

    @property
    def bbox(self):
        cdef BfBbox2 bbox = bfQuadtreeNodeGetBbox(self.quadtreeNode)
        return Bbox2(bbox.min[0], bbox.max[0], bbox.min[1], bbox.max[1])

    def get_inds(self):
        cdef BfSize i0 = self.get_first_index()
        cdef BfSize i1 = self.get_last_index()
        cdef SizeArray inds = SizeArray.__new__(SizeArray)
        inds.sizeArray = bfSizeArrayNewWithCapacity(i1 - i0)
        cdef Perm perm = self.tree.perm
        cdef BfSize i
        for i in range(i0, i1):
            bfSizeArrayAppend(inds.sizeArray, perm[i])
        return inds

    def get_points(self):
        cdef BfQuadtree *quadtree = bfQuadtreeNodeGetQuadtree(self.quadtreeNode)
        points = Points2()
        points.points = bfQuadtreeNodeGetPoints(self.quadtreeNode, quadtree)
        return points

def sample_uniform(lo=0, hi=1, n=1, dtype=np.float64):
    if dtype == np.float64:
        samples = [(hi - lo)*bfRealUniform1() + lo for _ in range(n)]
    else:
        raise NotImplementedError()
    return samples[0] if n == 1 else samples

cdef class RealArray:
    cdef BfRealArray *real_array

    cdef Py_ssize_t[1] _buf_shape
    cdef Py_ssize_t[1] _buf_strides

    def __cinit__(self):
        self.real_array = bfRealArrayNew()

    def __init__(self):
        bfRealArrayInitWithDefaultCapacity(self.real_array)

    def __len__(self):
        return bfRealArrayGetSize(self.real_array)

    def __getbuffer__(self, Py_buffer *buf, int flags):
        itemsize = sizeof(BfReal)
        n = len(self)
        self._buf_shape[0] = n
        self._buf_strides[0] = itemsize
        buf.buf = <char *>bfRealArrayGetDataPtr(self.real_array)
        buf.format = 'd'
        buf.internal = NULL
        buf.itemsize = itemsize
        buf.len = n*itemsize
        buf.ndim = 1
        buf.obj = self
        buf.readonly = 1
        buf.shape = self._buf_shape
        buf.strides = self._buf_strides
        buf.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buf):
        pass

    def __getitem__(self, index):
        if isinstance(index, Perm):
            copy = self.copy()
            copy.permute(index)
            return copy
        else:
            raise NotImplementedError(f'type {type(index)} currently unsupported')

    def copy(self):
        cdef RealArray _ = RealArray.__new__(RealArray)
        _.real_array = bfRealArrayNew()
        bfRealArrayInitCopy(_.real_array, self.real_array)
        return _

    def extend(self, RealArray arr):
        bfRealArrayExtend(self.real_array, arr.real_array)

    def permute(self, Perm perm):
        bfRealArrayPermute(self.real_array, perm.perm)

cdef class SizeArray:
    cdef BfSizeArray *sizeArray

    cdef Py_ssize_t[1] _buf_shape
    cdef Py_ssize_t[1] _buf_strides

    @staticmethod
    cdef from_ptr(BfSizeArray *sizeArray):
        cdef SizeArray _ = SizeArray.__new__(SizeArray)
        _.sizeArray = sizeArray
        return _

    def __len__(self):
        return bfSizeArrayGetSize(self.sizeArray)

    def __getbuffer__(self, Py_buffer *buf, int flags):
        itemsize = sizeof(BfSize)
        n = len(self)
        self._buf_shape[0] = n
        self._buf_strides[0] = itemsize
        buf.buf = <char *>bfSizeArrayGetDataPtr(self.sizeArray)
        buf.format = 'Q'
        buf.internal = NULL
        buf.itemsize = itemsize
        buf.len = n*itemsize
        buf.ndim = 1
        buf.obj = self
        buf.readonly = 1
        buf.shape = self._buf_shape
        buf.strides = self._buf_strides
        buf.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buf):
        pass

cdef class Tree:
    cdef BfTree *tree

    @staticmethod
    cdef _from_node_span_NodeSpan(NodeSpan nodeSpan):
        cdef Tree tree = Tree.__new__(Tree)
        cdef Perm perm = Perm.__new__(Perm)
        tree.tree = bfTreeNewFromNodeSpan(nodeSpan.nodeSpan, &perm.perm)
        return tree, perm

    @staticmethod
    cdef _from_node_span_list(list nodes):
        if not all(isinstance(_, TreeNode) for _ in nodes):
            raise ValueError()
        cdef BfPtrArray ptrArray
        bfInitPtrArray(&ptrArray, len(nodes))
        cdef TreeNode node
        for node in nodes:
            bfPtrArrayAppend(&ptrArray, node.treeNode)
        cdef BfNodeSpan *nodeSpan = bfNodeSpanNewFromPtrArray(&ptrArray, BF_POLICY_COPY)
        cdef Tree tree = Tree.__new__(Tree)
        cdef Perm perm = Perm.__new__(Perm)
        tree.tree = bfTreeNewFromNodeSpan(nodeSpan, &perm.perm)
        return tree, perm

    @staticmethod
    def from_node_span(nodeSpan):
        if isinstance(nodeSpan, NodeSpan):
            return Tree._from_node_span_NodeSpan(nodeSpan)
        elif isinstance(nodeSpan, list):
            return Tree._from_node_span_list(nodeSpan)
        else:
            raise NotImplementedError()

    @staticmethod
    def from_node(TreeNode node):
        cdef Tree tree = Tree.__new__(Tree)
        cdef Perm perm = Perm.__new__(Perm)
        tree.tree = bfTreeNewFromNode(node.treeNode, &perm.perm)
        return tree, perm

    @staticmethod
    def for_middle_fac(Tree templateTree, BfSize p):
        cdef Tree tree = Tree.__new__(Tree)
        tree.tree = bfTreeNewForMiddleFac(templateTree.tree, p)
        return tree

    @property
    def perm(self):
        perm = Perm()
        perm.perm = bfPermGetView(bfTreeGetPerm(self.tree))
        return perm

    @property
    def root(self):
        return reify_tree_node(<BfTreeNode *>bfTreeGetRootNode(self.tree))

    @property
    def nodes(self):
        return TreeLevelIter.from_tree(self)

    def get_level_nodes(self, BfSize level):
        cdef BfPtrArray *levelPtrArray = bfTreeGetLevelPtrArray(self.tree, level)
        nodes = [reify_tree_node(<BfTreeNode *>bfPtrArrayGet(levelPtrArray, i))
                 for i in range(bfPtrArraySize(levelPtrArray))]
        bfPtrArrayDeinitAndDealloc(&levelPtrArray)
        return nodes

    def __eq__(self, Tree other):
        return self.tree == other.tree

    def get_max_depth(self):
        return bfTreeGetMaxDepth(self.tree)

cdef class TreeLevelIter:
    cdef BfTreeLevelIter *treeLevelIter

    @staticmethod
    def from_tree(Tree tree):
        cdef TreeLevelIter _ = TreeLevelIter.__new__(TreeLevelIter)
        _.treeLevelIter = bfTreeLevelIterNewFromTree(BF_TREE_TRAVERSAL_UNKNOWN, tree.tree)
        return _

    def __iter__(self):
        cdef BfPtrArray *levelNodes = NULL
        cdef BfType treeNodeType
        while not bfTreeLevelIterIsDone(self.treeLevelIter):
            levelNodes = bfTreeLevelIterGetLevelNodes(self.treeLevelIter)
            for i in range(bfPtrArraySize(levelNodes)):
                yield reify_tree_node(<BfTreeNode *>bfPtrArrayGet(levelNodes, i))
            bfTreeLevelIterNext(self.treeLevelIter)

cdef class TreeNode:
    cdef BfTreeNode *treeNode

    def __hash__(self):
        return <Py_hash_t>self.treeNode

    def __eq__(self, TreeNode other):
        return self.treeNode == other.treeNode

    def get_num_points(self):
        return bfTreeNodeGetNumPoints(self.treeNode)

    def get_first_index(self):
        return bfTreeNodeGetFirstIndex(self.treeNode)

    def get_last_index(self):
        return bfTreeNodeGetLastIndex(self.treeNode)

    @property
    def tree(self):
        return reify_tree(bfTreeNodeGetTree(self.treeNode))

    @property
    def parent(self):
        return reify_tree_node(bfTreeNodeGetParent(self.treeNode))

    @property
    def children(self):
        return [
            reify_tree_node(bfTreeNodeGetChild(self.treeNode, i))
            for i in range(self.max_num_children)
            if bfTreeNodeHasChild(self.treeNode, i)
        ]

    @property
    def depth(self):
        return bfTreeNodeGetDepth(self.treeNode)

    @property
    def max_num_children(self):
        return bfTreeNodeGetMaxNumChildren(self.treeNode)

    @property
    def i0(self):
        return self.get_first_index()

    @property
    def i1(self):
        return self.get_last_index()

cdef class Trimesh:
    cdef BfTrimesh *trimesh

    def ensure_face_geometry(self):
        bfTrimeshEnsureFaceNormals(self.trimesh)

    @staticmethod
    def from_obj(path):
        path_byte_string = str(path).encode('UTF-8')
        cdef char *path_c_string = path_byte_string
        cdef Trimesh _ = Trimesh.__new__(Trimesh)
        _.trimesh = bfTrimeshNewFromObjFile(path_c_string)
        return _

    def init_embree(self):
        bfTrimeshInitEmbree(self.trimesh)

    @property
    def num_verts(self):
        return bfTrimeshGetNumVerts(self.trimesh)

    @property
    def num_faces(self):
        return bfTrimeshGetNumFaces(self.trimesh)

    @property
    def verts(self):
        cdef BfReal *data = bfTrimeshGetVertsPtr(self.trimesh)
        return np.asarray(<BfReal[:self.num_verts, :3]>data)

    @property
    def faces(self):
        cdef BfSize *data = bfTrimeshGetFacesPtr(self.trimesh)
        return np.asarray(<BfSize[:self.num_faces, :3]>data)

    @property
    def vertex_normals(self):
        if bfTrimeshHasVertexNormals(self.trimesh):
            return Vectors3.from_ptr(bfTrimeshGetVertexNormalsPtr(self.trimesh))

    @property
    def face_normals(self):
        if bfTrimeshHasFaceNormals(self.trimesh):
            return Vectors3.from_ptr(bfTrimeshGetFaceNormalsPtr(self.trimesh))

    def compute_face_normals_matching_vertex_normals(self):
        bfTrimeshComputeFaceNormalsMatchingVertexNormals(self.trimesh)

    def get_embree_handle(self) -> int:
        """
        Return the opaque Embree RTCScene handle as an integer.
        NOTE: You must call self.init_embree() first; otherwise this may be 0.
        """
        cdef void *h = bfTrimeshGetRTCSceneHandle(self.trimesh)
        return <size_t> h

    def get_native_ptr(self) -> int:
        """Return the native BfTrimesh* as an integer (size_t)."""
        return <size_t> self.trimesh

cdef class Vec:
    cdef BfVec *vec

    @staticmethod
    def from_ndarray(arr):
        if arr.dtype == np.float64:
            return VecReal.from_ndarray(arr)
        else:
            raise NotImplementedError()

    def __len__(self):
        return bfVecGetSize(self.vec)

cdef class VecReal(Vec):
    cdef BfVecReal *vecReal

    @staticmethod
    def from_ndarray(arr):
        # Make sure arr consists of packed 1D doubles
        assert arr.ndim == 1
        assert arr.flags.c_contiguous
        assert arr.itemsize == 8

        cdef BfSize n = arr.shape[0]
        cdef BfReal[:] data = arr

        cdef VecReal _ = VecReal.__new__(VecReal)
        _.vecReal = bfVecRealNewViewFromPtr(n, &data[0], 1)
        _.vec = bfVecRealToVec(_.vecReal)
        return _

    @staticmethod
    cdef from_ptr(BfVecReal *vecReal):
        cdef VecReal _ = VecReal.__new__(VecReal)
        _.vecReal = vecReal
        _.vec = bfVecRealToVec(_.vecReal)
        return _

    def to_array(self):
        cdef BfReal *data = bfVecRealGetDataPtr(self.vecReal)
        return np.asarray(<BfReal[:len(self)]>data)

cdef class Vectors2:
    cdef BfVectors2 *vectors

    def __cinit__(self):
        self.vectors = bfVectors2NewEmpty()

    def __len__(self):
        return bfVectors2GetSize(self.vectors)

    def extend(self, Vectors2 vectors):
        bfVectors2Extend(self.vectors, vectors.vectors)

cdef class Vectors3:
    cdef BfVectors3 *vectors

    @staticmethod
    cdef from_ptr(BfVectors3 *vectors):
        cdef Vectors3 _ = Vectors3.__new__(Vectors3)
        _.vectors = vectors
        return _

############################################################################
# Other stuff...

# - need to set mat for DenseLu
# - need to call bfMatInit and pass BfMatVtable
# - need to set MatMul entry of BfMatMul with a C or Cython function
#   that will look up and call DenseLu._Mul

cdef class MatPython(Mat):
    cdef BfMatPython *matPython

    def __init__(self, m, n):
        self.matPython = bfMatPythonNewFromPyObject(self, m, n)
        self.mat = bfMatPythonToMat(self.matPython)

    @staticmethod
    cdef from_ptr(BfMatPython *matPython):
        cdef MatPython _ = MatPython.__new__(MatPython)
        _.matPython = matPython
        _.mat = bfMatPythonToMat(_.matPython)
        return _

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
    cdef cnp.ndarray V = tm.verts        # (Nv, 3), dtype float64
    cdef cnp.ndarray F = tm.faces        # (Nf, 3), dtype uintp or int64

    cdef cnp.ndarray VF = V[F]           # (Nf, 3, 3)

    cdef cnp.ndarray P3  = VF.mean(axis=1)                          # (Nf, 3)
    cdef cnp.ndarray P2  = P3[:, :2].astype(np.float64, copy=False) # (Nf, 2)

    cdef cnp.ndarray C   = np.cross(VF[:, 1, :] - VF[:, 0, :],
                                    VF[:, 2, :] - VF[:, 0, :])      # (Nf, 3)
    cdef cnp.ndarray N2  = C[:, :2].astype(np.float64, copy=False)  # (Nf, 2)
    cdef cnp.ndarray nrm = np.sqrt((N2 * N2).sum(axis=1))           # (Nf,)

    cdef cnp.ndarray m = nrm > 0
    if m.any():
        N2[m] /= nrm[m][:, None]
    else:
        N2[...] = 0.0

    cdef Points2   pts = Points2(P2.shape[0])
    cdef Vectors2  nn  = Vectors2()
    cdef BfPoint2  p, q
    cdef Py_ssize_t i

    for i in range(P2.shape[0]):
        p[0] = <double>P2[i, 0]
        p[1] = <double>P2[i, 1]
        bfPoints2Append(pts.points, p)

        q[0] = <double>N2[i, 0]
        q[1] = <double>N2[i, 1]
        bfVectors2Append(nn.vectors, q)

    return Quadtree.from_points_and_normals(pts, nn)

cdef inline bint _is_far_bbox(const BfBbox2* br,
                              const BfBbox2* bc,
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
    cdef double diam   = diam_r if diam_r > diam_c else diam_c

    cdef double dx = crx - ccx
    cdef double dy = cry - ccy
    cdef double dist = sqrt(dx*dx + dy*dy)

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
    cdef const BfSize* ptr = bfSizeArrayGetDataPtr(idx.sizeArray)
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
    cdef public cnp.ndarray row_idx   # uintp[.]
    cdef public cnp.ndarray col_idx   # uintp[.]

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
    cdef MatCsrReal A

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
                j_local = ci[k]                   # 0..mj-1
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
    cdef MatProduct P
    cdef BfSize rank  # numerical rank of this block

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
        cdef cnp.ndarray xsub = x[self.col_idx]
        cdef object ysub_vec = self.P @ xsub  # VecReal
        y[self.row_idx] += ysub_vec.to_array()

cdef class FFNode(FFBlock):
    """
    Internal node: just a container of child blocks.
    """
    cdef list children

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
    cdef const BfSize* ptr = bfSizeArrayGetDataPtr(idx.sizeArray)

    for i in range(n):
        v = ptr[i]
        if v >= nFaces:
            if ff_debug:
                printf("[ff-debug] %s index out of range: %zu >= %zu (pos=%zu)\n",
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
        raise RuntimeError("hierarchical block size out of range")

    cdef bint small = (mi <= leaf_max) or (mj <= leaf_max)

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
    if ((leafI and leafJ) or small) and mi >= leaf_min and mj >= leaf_min:
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

    # Filter out any empty children
    children = [c for c in children if c is not None]
    if not children:
        if ff_debug:
            printf("[ff-debug] all children empty -> returning None\n")
            fflush(stdout)
        return None

    if ff_debug:
        printf("[ff-debug] built FFNode with %d children\n", <int>len(children))
        fflush(stdout)

    return FFNode(children)


cdef class HierarchicalFormFactor:
    """
    High-level wrapper for a hierarchical FF operator:

        y = H.apply(x) ≈ F x

    where F is the full form-factor matrix.
    """
    cdef FFBlock root
    cdef BfSize n
    cdef list sparse_leaves
    cdef list svd_leaves

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

    # cpdef cnp.ndarray apply(self, cnp.ndarray x):
    #     cdef cnp.ndarray x_flat = np.asarray(x, dtype=np.float64)
    #     if x_flat.ndim != 1 or x_flat.shape[0] != self.n:
    #         raise ValueError(
    #             f"x must be 1D of length {self.n}, got shape" # {x_flat.shape}"
    #         )
    #
    #     cdef cnp.ndarray y = np.zeros_like(x_flat)
    #     self.root.apply(x_flat, y)
    #
    #     if ff_debug:
    #         import numpy as _np
    #         nan_mask = _np.isnan(y)
    #         inf_mask = _np.isinf(y)
    #         num_nan = int(nan_mask.sum())
    #         num_inf = int(inf_mask.sum())
    #
    #         if num_nan or num_inf:
    #             nan_idx = _np.where(nan_mask)[0]
    #             inf_idx = _np.where(inf_mask)[0]
    #             print(f"[ff-debug] HierarchicalFormFactor.apply: "
    #                   f"{num_nan} NaN, {num_inf} Inf entries in y")
    #             print(f"[ff-debug] first NaN indices: {nan_idx[:20]}")
    #             print(f"[ff-debug] first Inf indices: {inf_idx[:20]}")
    #
    #     return y
    cpdef cnp.ndarray apply(self, cnp.ndarray x):
        cdef cnp.ndarray x_flat = np.asarray(x, dtype=np.float64)
        if x_flat.ndim != 1 or x_flat.shape[0] != self.n:
            raise ValueError(f"x must be 1D of length {self.n}, got shape") # {x_flat.shape}")

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
            dense_mv[i, j] = da[k]

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
    ci = <BfSize*> bfMatCsrRealGetColindConstPtr(A.mat_csr_real)
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

cdef class VfHier:
    cdef BfVfHier *vfHier
    cdef BfSize     n

    def __cinit__(self):
        self.vfHier = NULL
        self.n = 0

    @staticmethod
    def from_trimesh(Trimesh tm,
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

        By default this will:
          - ensure face geometry (normals) is available
          - initialize the Embree scene
        before constructing the quadtree and BfVfHier.
        """
        cdef VfHier H = VfHier.__new__(VfHier)

        if ensure_geometry:
            tm.ensure_face_geometry()
        if init_embree_scene:
            tm.init_embree()

        qt = _quadtree_from_trimesh_xy(tm)       # returns a Quadtree
        H.vfHier = bfVfHierNewFromQuadtree(tm.trimesh,
                                           qt.quadtree,
                                           eta, leaf_max, leaf_min, min_area,
                                           tol, min_svd_size, max_svd_rank_frac)
        if H.vfHier == NULL:
            raise RuntimeError("bfVfHierNewFromQuadtree failed")

        H.n = tm.num_faces
        return H

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

        H.n = tm.num_faces
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
            "num_svd_leaves":    int(s.numSvdLeaves),
            "num_nodes":         int(s.numNodeBlocks),
            "nnz_sparse_total":  int(s.nnzSparseTotal),
            "mem_bytes_sparse":  int(s.memBytesSparseEst),
            "mem_bytes_svd":     int(s.memBytesSvdEst),
            "mem_bytes_total":   int(s.memBytesSparseEst + s.memBytesSvdEst),
            "rank_total":        int(s.rankTotal),
        }


    cpdef cnp.ndarray apply(self, cnp.ndarray x):
        cdef cnp.ndarray x_flat = np.ascontiguousarray(x, dtype=np.float64)
        if x_flat.ndim != 1 or x_flat.shape[0] != self.n:
            raise ValueError(f"x must be 1D of length {self.n}")

        cdef cnp.ndarray y = np.zeros_like(x_flat)
        bfVfHierApply(self.vfHier, <BfReal*>x_flat.data, <BfReal*>y.data)
        return y

    cpdef apply_inplace(self, cnp.ndarray x, cnp.ndarray y):
        """
        Apply the hierarchical FF operator y = F @ x into a preallocated y.

        - x: 1D ndarray of length n (will be coerced to float64 view)
        - y: 1D ndarray of length n, float64 (output is written in-place)

        This avoids repeated allocations in tight loops (thermal time stepping).
        """
        cdef cnp.ndarray x_flat = np.ascontiguousarray(x, dtype=np.float64)
        if x_flat.ndim != 1 or x_flat.shape[0] != self.n:
            raise ValueError(f"x must be 1D of length {self.n}")

        if y.dtype != np.float64 or not y.flags.c_contiguous:
            raise ValueError("y must be float64 and C-contiguous")
        if y.ndim != 1 or y.shape[0] != self.n:
            raise ValueError(f"y must be 1D of length {self.n}")

        bfVfHierApply(self.vfHier,
                      <BfReal*>x_flat.data,
                      <BfReal*>y.data)
        return y

    @staticmethod
    cdef VfHier from_ptr(BfVfHier *ptr):
        cdef VfHier obj = VfHier.__new__(VfHier)
        obj.vfHier = ptr
        obj.n = bfVfHierGetNumFaces(ptr)
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
        if self.vfHier == NULL:
            return 0
        return <Py_ssize_t> bfVfHierGetNumFaces(self.vfHier)

### tests

def build_hierarchical_form_factor(Trimesh tm,
                                   double tol=1e-3,
                                   double eta=2.0,
                                   BfSize leaf_max=256,
                                   BfSize leaf_min=1,
                                   BfSize min_svd_size=5000, #16384,
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
    Ax_vec = A_mat @ x          # VecReal
    Ax = Ax_vec.to_array()      # numpy 1D

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
            nnz * 8      # data
            + nnz * 8    # colind
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
    cdef MatCsrReal FF = FormFactorMat.from_trimesh(tm) #.mat_csr_real
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
    y_full_vec = FF @ x        # VecReal
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
    print(f"[bench] full FF memory: {mem_full/1e6:.3f} MB")

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
    print(f"[bench] hier FF memory est: {mem_hier/1e6:.3f} MB "
          f"(sparse ~{hstats['mem_bytes_sparse_est']/1e6:.3f} MB, "
          f"SVD ~{hstats['mem_bytes_svd_est']/1e6:.3f} MB)")

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
    print(f"[bench] full FF MVP: {t_full_mvp*1e3:.2f} ms (avg over {mvp_repeats})")

    # Hierarchical FF MVP
    t0 = time.perf_counter()
    for _i in range(mvp_repeats):
        y_hier = H.apply(x)
    t_hier_mvp = (time.perf_counter() - t0) / mvp_repeats
    print(f"[bench] hier FF MVP: {t_hier_mvp*1e3:.2f} ms (avg over {mvp_repeats})")

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
        tm.trimesh,
        qt.quadtree,
        eta,
        leaf_max,
        leaf_min,
        min_area,
        0.0,   # tol (unused when minSvdSize == 0)
        0,     # minSvdSize == 0 => disable SVD
        0.0)   # maxSvdRankFrac (ignored when SVD disabled)

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
    H = VfHier.from_trimesh(tm,
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

    print(f"[vf-block] VfHier MVP: {t_hier_mvp*1e3:.2f} ms")
    print(f"[vf-block] Block MVP:  {t_block_mvp*1e3:.2f} ms")
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
    y_ref_vec = F_csr @ x          # VecReal
    y_ref = y_ref_vec.to_array()   # 1D numpy
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

    print(f"[vf-hier] full MVP:  {t_full_mvp*1e3:.2f} ms")
    print(f"[vf-hier] hier MVP:  {t_hier_mvp*1e3:.2f} ms")
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
    cdef VfHier H_comp

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
    print(f"[bench] full FF memory: {mem_full/1e6:.3f} MB")

    # --- compressed C hierarchical FF (with SVD) ---
    t0 = time.perf_counter()
    H_comp = VfHier.from_trimesh(tm,
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
    print(f"[vf-hier-SVD] mem est: sparse={stats.memBytesSparseEst/1e6:.3f} MB, "
          f"SVD={stats.memBytesSvdEst/1e6:.3f} MB, "
          f"total={(stats.memBytesSparseEst+stats.memBytesSvdEst)/1e6:.3f} MB")
    print(f"[vf-hier-SVD] total SVD rank sum={stats.rankTotal}")

    # --- random test vector ---
    _np.random.seed(seed_val)
    x = _np.random.rand(tm.num_faces)

    # full MVP
    t0 = time.perf_counter()
    y_ref_vec = F_csr @ x          # VecReal
    y_ref = y_ref_vec.to_array()   # 1D numpy
    t_full_mvp = time.perf_counter() - t0

    # compressed MVP
    t0 = time.perf_counter()
    y_comp = H_comp.apply(x)
    t_comp_mvp = time.perf_counter() - t0

    print(f"[vf-hier-SVD] full MVP:  {t_full_mvp*1e3:.2f} ms")
    print(f"[vf-hier-SVD] comp MVP:  {t_comp_mvp*1e3:.2f} ms")

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
    cdef VfHier H_old
    cdef VfHier H_new

    t0 = time.perf_counter()
    H_old = VfHier.from_trimesh(tm,
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
    mem_old_svd    = stats_old.memBytesSvdEst
    mem_old_total  = mem_old_sparse + mem_old_svd

    print(f"[cmp] OLD VfHier memory: total={mem_old_total/1e6:.3f} MB "
          f"(sparse={mem_old_sparse/1e6:.3f} MB, "
          f"SVD={mem_old_svd/1e6:.3f} MB)")
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
    print(f"[cmp] OLD VfHier MVP: {t_old_mvp*1e3:.2f} ms (avg over {mvp_repeats})")

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
