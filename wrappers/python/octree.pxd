# wrappers/python/octree.pxd
# cython: language_level=3

from defs cimport BfSize
from types cimport BfType
from tree cimport BfTree
from points cimport BfPoints3
from vectors cimport BfVectors3
from geom cimport BfPoint3
from size_array cimport BfSizeArray

cdef extern from "bf/octree.h":
    ctypedef struct BfOctree:
        pass

    BfType bfOctreeGetType(const BfOctree *octree)

    BfTree *bfOctreeToTree(BfOctree *octree)
    const BfTree *bfOctreeConstToTreeConst(const BfOctree *octree)

    BfOctree *bfTreeToOctree(BfTree *tree)

    BfOctree *bfOctreeNew()
    BfOctree *bfOctreeNewFromPoints(const BfPoints3 *points, BfSize maxLeafSize)
    void bfOctreeInit(BfOctree *octree,
                      const BfPoints3 *points,
                      const BfVectors3 *unitNormals,
                      BfSize maxLeafSize)

    void bfOctreeDeinit(BfOctree *octree)
    void bfOctreeDealloc(BfOctree **octree)
    void bfOctreeDelete(BfOctree **octree)

    BfSizeArray *bfOctreeGetNearestNeighbors(const BfOctree *octree,
                                             BfPoint3 point,
                                             BfSize numNbs)
