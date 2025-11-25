from size_array cimport BfSizeArray
# from types cimport BfMat, BfMatCsrReal, BfTrimesh, BfVec
from defs cimport BfSize, BfReal, BfPolicy

# Forward declarations that match the C headers exactly:
cdef extern from "bf/types.h":
    cdef struct BfMat        # opaque
    cdef struct BfVec        # opaque
    cdef struct BfTrimesh    # opaque

cdef extern from "bf/mat_csr_real.h":

    cdef struct BfMatCsrReal:   # opaque to Cython (we don't list fields)
        pass

    BfMatCsrReal *bfMatCsrRealNewViewFactorMatrixFromTrimesh(const BfTrimesh *trimesh, const BfSizeArray *rowInds, const BfSizeArray *colInds)
    BfMatCsrReal *bfMatCsrRealNewViewFactorMatrixFromTrimeshTol(const BfTrimesh *trimesh, const BfSizeArray *rowInds, const BfSizeArray *colInds, BfReal eps)
    BfMat *bfMatCsrRealToMat(BfMatCsrReal *matCsrReal)
    BfMatCsrReal *bfMatToMatCsrReal(BfMat *mat)

    const BfSize* bfMatCsrRealGetRowptrConstPtr(const BfMatCsrReal* A)
    const BfSize* bfMatCsrRealGetColindConstPtr(const BfMatCsrReal* A)
    const BfReal* bfMatCsrRealGetDataConstPtr  (const BfMatCsrReal* A)

    # CSR creation from arrays (present in your lib)
    BfMatCsrReal *bfMatCsrRealNewFromPtrs(
        BfSize numRows, BfSize numCols,
        const BfSize *rowptr,
        const BfSize *colind,
        const BfReal *data)
