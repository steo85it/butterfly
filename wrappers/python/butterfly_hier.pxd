# wrappers/python/butterfly_hier.pxd
# cython: language_level=3

#
import numpy as np
cimport numpy as cnp

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

from butterfly cimport MatCsrReal, MatProduct, Quadtree, QuadtreeNode, Trimesh, reify_tree_node

cdef class FFBlock:
    cdef public cnp.ndarray row_idx
    cdef public cnp.ndarray col_idx
    cpdef apply(self, cnp.ndarray x, cnp.ndarray y)

cdef class FFSparseLeaf(FFBlock):
    cdef MatCsrReal A

cdef class FFSvdLeaf(FFBlock):
    cdef MatProduct P
    cdef BfSize rank

cdef class FFNode(FFBlock):
    cdef list children

cdef class HierarchicalFormFactor:
    cdef FFBlock root
    cdef BfSize n
    cdef list sparse_leaves
    cdef list svd_leaves
    cpdef cnp.ndarray apply(self, cnp.ndarray x)

cdef class VfHier:
    cdef BfVfHier *vfHier
    cdef BfSize     _n
    cdef object     _qt_owner  # keep Quadtree alive if C stores pointer
    cdef inline void _apply_ptr(self,
                                const double *x,
                                double *y) noexcept nogil
    cpdef dump_leaf_blocks(self)
    cpdef cnp.ndarray apply_vec(self, cnp.ndarray x)
    cpdef apply_inplace(self, cnp.ndarray x, cnp.ndarray y)
    cdef inline void _apply_mat_ptr(self,
                                    const double *X,
                                    double *Y,
                                    BfSize k) noexcept nogil
    cpdef cnp.ndarray apply_mat(self, cnp.ndarray X)
    cpdef apply_mat_inplace(self, cnp.ndarray X, cnp.ndarray Y)
    cpdef apply(self, object x)
    @staticmethod
    cdef VfHier from_ptr(BfVfHier *ptr)

# If butterfly_dev.pyx calls these cdef helpers, they must be declared here too:
cdef class FormFactorMat(MatCsrReal):
    pass

cdef Quadtree _quadtree_from_trimesh_xy(Trimesh tm)
cdef FFBlock _build_ff_block(Trimesh tm,
                             QuadtreeNode rowNode,
                             QuadtreeNode colNode,
                             double tol,
                             double eta,
                             BfSize leaf_max,
                             BfSize leaf_min,
                             BfSize min_svd_size,
                             double max_svd_rank_frac)
