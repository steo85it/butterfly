# from defs cimport BfReal, BfSize
# from types cimport BfMat
#
# cdef extern from "bf/linalg.h":
#     BfMat *bfSolveGMRES(const BfMat *A, const BfMat *B, BfMat *X0, BfReal tol, BfSize maxNumIter, BfSize *numIter, const BfMat *M)

from defs cimport BfReal, BfSize
from types cimport BfMat
from mat_diag_real cimport BfMatDiagReal

cdef extern from "bf/linalg.h":
    ctypedef enum BfBackend:
        BF_BACKEND_LAPACK
        BF_BACKEND_SVDS

    cdef struct BfTruncSpec:
        bint usingTol
        BfReal tol
        BfSize k

    BfSize bfTruncSpecGetNumTerms(const BfTruncSpec *truncSpec,
                                  const BfMatDiagReal *S)

    BfMat *bfSolveGMRES(const BfMat *A,
                        const BfMat *B,
                        BfMat *X0,
                        BfReal tol,
                        BfSize maxNumIter,
                        BfSize *numIter,
                        const BfMat *M)

    bint bfGetTruncatedSvd(const BfMat *mat,
                           BfMat **UPtr,
                           BfMatDiagReal **SPtr,
                           BfMat **VTPtr,
                           const BfTruncSpec *truncSpec,
                           BfBackend backend)

    void bfSparseSvdPrintStats(const char *tag)
    void bfSparseSvdResetStats()