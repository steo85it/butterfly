# circle.pxd
from defs cimport BfReal
from geom cimport BfPoint2

cdef extern from "bf/circle.h":
    ctypedef struct BfCircle:
        BfPoint2 center
        BfReal   r
