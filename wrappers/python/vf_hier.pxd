from quadtree cimport BfQuadtree
from types cimport BfTrimesh, BfMatCsrReal
from defs cimport BfReal, BfSize

cdef extern from "bf/vf_hier.h":
    cdef struct BfVfHier:
        pass

    cdef struct BfVfHierStats:
        BfSize numSparseLeaves
        BfSize numSvdLeaves
        BfSize numNodeBlocks
        unsigned long long nnzSparseTotal
        unsigned long long memBytesSparseEst
        unsigned long long memBytesSvdEst
        unsigned long long rankTotal

    void bfVfHierCollectStats(const BfVfHier *vfHier,
                              BfVfHierStats *stats)

    BfVfHier *bfVfHierNewFromTrimesh(const BfTrimesh *trimesh,
                                     BfReal eta,
                                     BfSize leafMax,
                                     BfSize leafMin)

    BfVfHier *bfVfHierNewFromQuadtree(const BfTrimesh *trimesh,
                                      BfQuadtree *quadtree,
                                      BfReal eta,
                                      BfSize leafMax,
                                      BfSize leafMin,
                                      BfReal        minArea,
                                      BfReal tol,
                                      BfSize minSvdSize,
                                      BfReal maxSvdRankFrac)

    void bfVfHierInitFromCsrAndQuadtree(BfVfHier *vfHier,
                                        BfMatCsrReal *Afull,
                                        BfQuadtree *quadtree,
                                        BfReal eta,
                                        BfSize leafMax,
                                        BfSize leafMin,
                                        BfReal        minArea,
                                        BfReal tol,
                                        BfSize minSvdSize,
                                        BfReal maxSvdRankFrac)

    BfVfHier *bfVfHierNewFromCsrAndQuadtree(BfMatCsrReal *Afull,
                                            BfQuadtree *quadtree,
                                            BfReal eta,
                                            BfSize leafMax,
                                            BfSize leafMin,
                                            BfReal        minArea,
                                            BfReal tol,
                                            BfSize minSvdSize,
                                            BfReal maxSvdRankFrac)

    void bfVfHierApply(const BfVfHier *vfHier,
                       const BfReal *x,
                       BfReal *y) nogil

    void bfVfHierApplyMany(const BfVfHier *vfHier,
                           const BfReal   *X, BfSize ldX,
                           BfReal         *Y, BfSize ldY,
                           BfSize nrhs) nogil

    BfSize bfVfHierGetNumFaces(const BfVfHier *vfHier)

    void bfVfHierDeinitAndDealloc(BfVfHier **vfHierPtr)

    bint bfVfHierSave(const BfVfHier *vfHier, const char *path)
    BfVfHier *bfVfHierLoad(const char *path)
    BfSize bfVfHierGetNumFaces(const BfVfHier *vfHier)

    BfSize bfVfHierGetNumLeafBlocks(const BfVfHier *vfHier)
    void bfVfHierDumpLeafBlocks(const BfVfHier *vfHier,
                                BfSize *row_i0,
                                BfSize *row_i1,
                                BfSize *col_j0,
                                BfSize *col_j1,
                                unsigned char *kind,
                                BfSize *rank,
                                unsigned long long *nnz)
