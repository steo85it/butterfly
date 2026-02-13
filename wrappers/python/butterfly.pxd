# cython: language_level=3
from bbox cimport *
from defs cimport *
#
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

import numpy as np
cimport numpy as cnp

cdef class Mat:
    cdef BfMat *mat
    @staticmethod
    cdef Mat from_ndarray(cnp.ndarray arr)


cdef class MatCsrReal(Mat):
    cdef BfMatCsrReal *mat_csr_real

cdef class MatDiagReal(Mat):
    @staticmethod
    cdef MatDiagReal from_constant(BfSize m, BfSize n, BfReal diag_value)
    @staticmethod
    cdef from_ptr(BfMatDiagReal *matDiagReal)
    cdef BfMatDiagReal *matDiagReal

cdef class MatProduct(Mat):
    cdef BfMatProduct *matProduct
    cdef list _factors
    @staticmethod
    cdef from_ptr(BfMatProduct *matProduct)
    cdef post_multiply(self, Mat mat)

cdef class Vec:
    cdef BfVec *vec

cdef class VecReal(Vec):
    cdef BfVecReal *vecReal
    cdef object _owner
    @staticmethod
    cdef from_ptr(BfVecReal *vecReal)

cdef class Tree:
    cdef BfTree *tree
    @staticmethod
    cdef _from_node_span_NodeSpan(NodeSpan nodeSpan)
    @staticmethod
    cdef _from_node_span_list(list nodes)

cdef class NodeSpan:
    cdef BfNodeSpan *nodeSpan

cdef class TreeNode:
    cdef BfTreeNode *treeNode

cdef class Quadtree(Tree):
    cdef BfQuadtree *quadtree
    @staticmethod
    cdef from_ptr(BfQuadtree *quadtree)

cdef class QuadtreeNode(TreeNode):
    cdef BfQuadtreeNode *quadtreeNode
    @staticmethod
    cdef from_ptr(BfQuadtreeNode *quadtreeNode)

cdef class Trimesh:
    cdef BfTrimesh *trimesh

cdef class Points2:
    cdef BfPoints2 *points     # was *pts
    cdef Py_ssize_t[2] shape
    cdef Py_ssize_t[2] strides

cdef class Vectors2:
    cdef BfVectors2 *vectors   # was *vecs

cdef class SizeArray:
    cdef BfSizeArray *sizeArray

    cdef Py_ssize_t[1] _buf_shape
    cdef Py_ssize_t[1] _buf_strides

    @staticmethod
    cdef from_ptr(BfSizeArray *sizeArray)

cdef object reify_tree_node(BfTreeNode *treeNode)
cdef object reify_tree(BfTree *tree)
cdef object reify_mat(BfMat *mat)
cdef object reify_vec(BfVec *vec)