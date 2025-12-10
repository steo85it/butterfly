from quadtree cimport BfQuadtree
from types cimport BfTrimesh, BfMat
from defs cimport BfReal, BfSize

cdef extern from "bf/mat_vf_hier.h":

    BfMat *bfMatVfHierNewFromQuadtree(
        const BfTrimesh *trimesh,
        BfQuadtree      *quadtree,
        BfReal           eta,
        BfSize           leafMax,
        BfSize           leafMin,
        BfReal           minArea,
        BfReal           tol,
        BfSize           minSvdSize,
        BfReal           maxSvdRankFrac)
