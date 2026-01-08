#pragma once

#ifndef BF_VF_HIER_H
#define BF_VF_HIER_H

#include <bf/def.h>
#include <bf/size_array.h>
#include <bf/ptr_array.h>
#include <bf/quadtree.h>
#include <bf/trimesh.h>
#include <bf/mat.h>
#include <bf/mat_csr_real.h>
#include <bf/vec_real.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum BfVfHierBlockKind {
  BF_VF_HIER_BLOCK_NONE  = 0,
  BF_VF_HIER_BLOCK_SPARSE,
  BF_VF_HIER_BLOCK_SVD,
  BF_VF_HIER_BLOCK_NODE
} BfVfHierBlockKind;

/* Sparse leaf: explicit CSR sub-block with row/col index sets */
typedef struct BfVfSparseLeaf {
  BfSizeArray  rowInds;      /* global row indices */
  BfSizeArray  colInds;      /* global col indices */
  BfMatCsrReal *mat;         /* |rowInds| x |colInds| view-factor block */
  bool         colsAreLocal; /* true if mat->colind is 0..nA-1 (local) */
} BfVfSparseLeaf;

/* SVD leaf: explicit SVD sub-block with row/col index sets */
typedef struct BfVfSvdLeaf {
  BfSizeArray rowInds;       /* global row indices */
  BfSizeArray colInds;       /* global col indices */
  BfMat      *mat;           /* MatProduct U S V^T, stored as generic BfMat* */
  BfSize      rank;          /* numerical rank */
  BfReal     *work;
  BfSize      workLen;
} BfVfSvdLeaf;

/* Internal node: just a list of child blocks */
typedef struct BfVfNode {
  BfPtrArray children;       /* array of BfVfHierBlock* */
} BfVfNode;

/* Hierarchical block */
typedef struct BfVfHierBlock {
  BfVfHierBlockKind kind;
  union {
    BfVfSparseLeaf sparse;
    BfVfSvdLeaf    svd;
    BfVfNode       node;
  } data;
} BfVfHierBlock;

/* Top-level hierarchical operator */
typedef struct BfVfHier {
  BfTrimesh const *trimesh; /* mesh is owned externally */
  BfVfHierBlock   *root;    /* root block of the hierarchy */
  BfSize           n;       /* number of faces */
} BfVfHier;

/* Stats for inspecting the hierarchy (for debugging / profiling) */
typedef struct BfVfHierStats {
  BfSize numSparseLeaves;
  BfSize numSvdLeaves;
  BfSize numNodeBlocks;

  unsigned long long nnzSparseTotal;

  unsigned long long memBytesSparseEst;
  unsigned long long memBytesSvdEst;
  unsigned long long rankTotal;
} BfVfHierStats;

/* Construction */
BfVfHier *bfVfHierNew(void);

void bfVfHierCollectStats(BfVfHier const *vfHier,
                          BfVfHierStats *stats);

void bfVfHierInitFromTrimesh(BfVfHier        *vfHier,
                             BfTrimesh const *trimesh,
                             BfReal           eta,
                             BfSize           leafMax,
                             BfSize           leafMin);

BfVfHier *bfVfHierNewFromTrimesh(BfTrimesh const *trimesh,
                                 BfReal           eta,
                                 BfSize           leafMax,
                                 BfSize           leafMin);

void bfVfHierInitFromQuadtree(BfVfHier        *vfHier,
                              BfTrimesh const *trimesh,
                              BfQuadtree      *quadtree,
                              BfReal           eta,
                              BfSize           leafMax,
                              BfSize           leafMin,
                              BfReal        minArea,
                              BfReal           tol,
                              BfSize           minSvdSize,
                              BfReal           maxSvdRankFrac);

BfVfHier *bfVfHierNewFromQuadtree(BfTrimesh const *trimesh,
                                  BfQuadtree      *quadtree,
                                  BfReal           eta,
                                  BfSize           leafMax,
                                  BfSize           leafMin,
                                  BfReal        minArea,
                                  BfReal           tol,
                                  BfSize           minSvdSize,
                                  BfReal           maxSvdRankFrac);

/* New: CSR + quadtree constructors */

void bfVfHierInitFromCsrAndQuadtree(BfVfHier        *vfHier,
                                    BfMatCsrReal    *Afull,
                                    BfQuadtree      *quadtree,
                                    BfReal           eta,
                                    BfSize           leafMax,
                                    BfSize           leafMin,
                                    BfReal        minArea,
                                    BfReal           tol,
                                    BfSize           minSvdSize,
                                    BfReal           maxSvdRankFrac);

BfVfHier *bfVfHierNewFromCsrAndQuadtree(BfMatCsrReal *Afull,
                                         BfQuadtree   *quadtree,
                                         BfReal        eta,
                                         BfSize        leafMax,
                                         BfSize        leafMin,
                                         BfReal        minArea,
                                         BfReal        tol,
                                         BfSize        minSvdSize,
                                         BfReal        maxSvdRankFrac);

/* Application: y <- y + F x  (size n) */
void bfVfHierApply(BfVfHier const *vfHier,
                   BfReal const   *x,
                   BfReal         *y);

/* Destruction */
void bfVfHierDeinit(BfVfHier *vfHier);
void bfVfHierDealloc(BfVfHier **vfHierPtr);
void bfVfHierDeinitAndDealloc(BfVfHier **vfHierPtr);

/* ============================================================
 * I/O: save/load a VfHier to/from a single binary file.
 *
 * Notes:
 *  - This stores ONLY the hierarchy (blocks, indices, matrices).
 *  - trimesh/quadtree are NOT serialized.
 *  - Loaded vfHier has vfHier->trimesh == NULL and vfHier->n set.
 * ============================================================ */

BfBool  bfVfHierSave(BfVfHier const *vfHier, char const *path);
BfVfHier *bfVfHierLoad(char const *path);
BfSize bfVfHierGetNumFaces(const BfVfHier *vfHier);

#ifdef __cplusplus
}
#endif

#endif /* BF_VF_HIER_H */
