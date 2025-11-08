from defs cimport BfReal, BfSize
from types cimport BfTrimesh
from vectors cimport BfVectors3

cdef extern from "bf/trimesh.h":
    BfTrimesh *bfTrimeshNewFromObjFile(const char *objPath)
    void bfTrimeshInitEmbree(BfTrimesh *trimesh)
    BfSize bfTrimeshGetNumVerts(const BfTrimesh *trimesh)
    BfSize bfTrimeshGetNumFaces(const BfTrimesh *trimesh)
    BfReal *bfTrimeshGetVertsPtr(BfTrimesh *trimesh)
    BfSize *bfTrimeshGetFacesPtr(BfTrimesh *trimesh)
    bint bfTrimeshHasFaceNormals(const BfTrimesh *trimesh)
    bint bfTrimeshHasVertexNormals(const BfTrimesh *trimesh)
    BfVectors3 *bfTrimeshGetFaceNormalsPtr(BfTrimesh *trimesh)
    BfVectors3 *bfTrimeshGetVertexNormalsPtr(BfTrimesh *trimesh)
    void bfTrimeshComputeFaceNormalsMatchingVertexNormals(BfTrimesh *trimesh)
    void *bfTrimeshGetRTCSceneHandle(const BfTrimesh *trimesh)

