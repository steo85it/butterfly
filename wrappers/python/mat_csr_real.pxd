from size_array cimport BfSizeArray
from types cimport BfMat, BfMatCsrReal, BfTrimesh, BfVec
from defs cimport BfSize, BfReal

cdef extern from "bf/mat_csr_real.h":
    BfMatCsrReal *bfMatCsrRealNewViewFactorMatrixFromTrimesh(const BfTrimesh *trimesh, const BfSizeArray *rowInds, const BfSizeArray *colInds)
    BfMat *bfMatCsrRealToMat(BfMatCsrReal *matCsrReal)

    const BfSize* bfMatCsrRealGetRowptrConstPtr(const BfMatCsrReal* A)
    const BfSize* bfMatCsrRealGetColindConstPtr(const BfMatCsrReal* A)
    const BfReal* bfMatCsrRealGetDataConstPtr  (const BfMatCsrReal* A)

