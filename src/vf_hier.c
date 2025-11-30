#include <bf/def.h>
#include <bf/vf_hier.h>

#include <bf/mem.h>
#include <bf/error.h>
#include <bf/tree.h>
#include <bf/tree_node.h>
#include <bf/quadtree_node.h>
#include <bf/vec_real.h>
#include <bf/mat_dense_real.h>
#include <bf/points.h>
#include <bf/vectors.h>
#include <bf/mat.h>
#include <bf/vec.h>
#include <bf/perm.h>
#include <bf/assert.h>
#include <bf/mat_csr_real.h>
#include <bf/size_array.h>      /* NEW: for bfSizeArrayNewWithCapacity, etc */
//#include <bf/indexed_mat.h>     /* NEW: for BfIndexedMat, bfIndexedMatNewFromMat */
//#include <bf/mat_block_coo.h>   /* NEW: for BfMatBlockCoo, bfMatBlockCooNewFromIndexedBlocks */
#include <bf/mat_product.h>
#include <bf/mat_diag_real.h>
#include <bf/linalg.h>  /* for bfGetTruncatedSvd, BfTruncSpec, BfBackend */

#include <stdio.h>
#include <math.h>
#include <time.h>  /* for timing instrumentation */
#include <string.h>         /* for memset */
#include <bf/real_array.h>  /* for BfRealArray, bfRealArrayNewWithDefaultCapacity, etc */

#ifndef BF_UNUSED
#  define BF_UNUSED(x) (void)(x)
#endif

#ifndef BF_TRUE
#  define BF_TRUE true
#endif

#ifndef BF_FALSE
#  define BF_FALSE false
#endif

#ifndef BfBool
#  define BfBool bool
#endif

/* Enable/disable SVD timing (set to 0 to compile out) */
#define BF_VF_HIER_TIME_SVD 1

#if BF_VF_HIER_TIME_SVD
static double bfVfHierNowSecs(void) {
  return (double)clock() / (double)CLOCKS_PER_SEC;
}

/* Accumulate total CSR->dense and SVD times over one hierarchy build */
static double g_vf_csr_to_dense_time = 0.0;
static double g_vf_svd_time          = 0.0;
#endif

/* Local helpers to avoid BF's max/sqrt macros */
static BfReal bf_local_max(BfReal a, BfReal b) {
  return a > b ? a : b;
}

static BfReal bf_local_sqrt(BfReal x) {
  if (x <= 0) return 0;
  return (BfReal)sqrt((double)x);
}

typedef struct {
  BfSize i0, i1;
  BfSize j0, j1;
  BfSize mi, mj;
  bool leafI, leafJ;
  bool small;
  bool empty;
} BfVfBlockMeta;

static BfVfBlockMeta getBlockMeta(BfQuadtreeNode *rowNode,
                                  BfQuadtreeNode *colNode,
                                  BfSize          leafMax);

static BfVfHierBlock *bfVfHierBlockNew(void);
static void           bfVfHierBlockDeinit(BfVfHierBlock *block);
static void           bfVfHierBlockDealloc(BfVfHierBlock **blockPtr);
static void           bfVfHierBlockDeinitAndDealloc(BfVfHierBlock **blockPtr); /* NEW */
static void           bfVfHierBlockApply(BfVfHierBlock const *block,
                                         BfReal const        *x,
                                         BfReal              *y,
                                         BfSize               n);

/* --- Utilities --------------------------------------------------------- */
static void bfVfHierStatsInit(BfVfHierStats *stats) {
  memset(stats, 0, sizeof(*stats));
}

static void bfVfHierBlockCollectStats(BfVfHierBlock const *block,
                                      BfVfHierStats       *stats) {
  if (block == NULL) return;

  switch (block->kind) {
  case BF_VF_HIER_BLOCK_SPARSE: {
    BfVfSparseLeaf const *leaf = &block->data.sparse;

    BfMat *A = bfMatCsrRealToMat(leaf->mat);
    BfSize mA = bfMatGetNumRows(A);
    BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(leaf->mat);
    BF_ASSERT(rp != NULL);
    BfSize nnz = rp[mA];

    stats->numSparseLeaves += 1;
    stats->nnzSparseTotal  += (unsigned long long)nnz;

    /* crude CSR memory estimate */
    stats->memBytesSparseEst +=
      (unsigned long long)nnz * 8ull      /* data */
      + (unsigned long long)nnz * 8ull    /* colind */
      + (unsigned long long)(mA + 1) * 8ull; /* rowptr */
  } break;

  case BF_VF_HIER_BLOCK_SVD: {
    BfVfSvdLeaf const *leaf = &block->data.svd;
    BfSize m = bfSizeArrayGetSize((BfSizeArray *)&leaf->rowInds);
    BfSize n = bfSizeArrayGetSize((BfSizeArray *)&leaf->colInds);
    BfSize r = leaf->rank;

    stats->numSvdLeaves += 1;
    stats->rankTotal    += (unsigned long long)r;

    /* memory for U (m×r), S (r), VT (r×n) */
    stats->memBytesSvdEst +=
      (unsigned long long)(m*r + r + r*n) * 8ull;
  } break;

  case BF_VF_HIER_BLOCK_NODE: {
    stats->numNodeBlocks += 1;
    BfPtrArray const *children = &block->data.node.children;
    for (BfSize i = 0; i < bfPtrArraySize(children); ++i) {
      BfVfHierBlock *child = bfPtrArrayGet(children, i);
      bfVfHierBlockCollectStats(child, stats);
    }
  } break;

  case BF_VF_HIER_BLOCK_NONE:
  default:
    break;
  }
}

void bfVfHierCollectStats(BfVfHier const *vfHier,
                          BfVfHierStats *stats) {
  BF_ASSERT(vfHier != NULL);
  BF_ASSERT(stats != NULL);

  bfVfHierStatsInit(stats);

  if (vfHier->root != NULL)
    bfVfHierBlockCollectStats(vfHier->root, stats);
}

static BfVfSparseLeaf *bfVfSparseLeafNew(void) {
  BfVfSparseLeaf *leaf = bfMemAlloc(1, sizeof(BfVfSparseLeaf));
  bfSizeArrayInitWithDefaultCapacity(&leaf->rowInds);
  bfSizeArrayInitWithDefaultCapacity(&leaf->colInds);
  leaf->mat = NULL;
  return leaf;
}

static void bfVfSparseLeafDeinit(BfVfSparseLeaf *leaf) {
  if (leaf == NULL) return;
  bfSizeArrayDeinit(&leaf->rowInds);
  bfSizeArrayDeinit(&leaf->colInds);
  if (leaf->mat != NULL)
    bfMatCsrRealDeinitAndDealloc(&leaf->mat);
}

/* --- HierBlock helpers ------------------------------------------------- */

static BfVfHierBlock *bfVfHierBlockNew(void) {
  BfVfHierBlock *block = bfMemAlloc(1, sizeof(BfVfHierBlock));
  memset(block, 0, sizeof(BfVfHierBlock));
  block->kind = BF_VF_HIER_BLOCK_NONE;
  return block;
}

static void bfVfSvdLeafDeinit(BfVfSvdLeaf *leaf) {
  if (leaf == NULL) return;
  bfSizeArrayDeinit(&leaf->rowInds);
  bfSizeArrayDeinit(&leaf->colInds);
  if (leaf->mat != NULL) {
    /* mat is a generic Mat (actually MatProduct); use BF's Mat destructor */
    bfMatDelete(&leaf->mat);  /* or whatever your Mat destroy function is */
  }
  if (leaf->work != NULL) {
    bfMemFree(leaf->work);
    leaf->work = NULL;
    leaf->workLen = 0;
  }
}

static void bfVfHierBlockDeinit(BfVfHierBlock *block) {
  if (block == NULL) return;

  switch (block->kind) {
  case BF_VF_HIER_BLOCK_SPARSE:
    bfVfSparseLeafDeinit(&block->data.sparse);
    break;

  case BF_VF_HIER_BLOCK_SVD:
    bfVfSvdLeafDeinit(&block->data.svd);
    break;

  case BF_VF_HIER_BLOCK_NODE:
    /* Deinit all children and free the array */
    for (BfSize i = 0; i < bfPtrArraySize(&block->data.node.children); ++i) {
      BfVfHierBlock *child = bfPtrArrayGet(&block->data.node.children, i);
      bfVfHierBlockDeinitAndDealloc(&child);
    }
    bfPtrArrayDeinit(&block->data.node.children);
    break;

  case BF_VF_HIER_BLOCK_NONE:
  default:
    break;
  }

  block->kind = BF_VF_HIER_BLOCK_NONE;
}

static void bfVfHierBlockDealloc(BfVfHierBlock **blockPtr) {
  if (blockPtr == NULL || *blockPtr == NULL) return;
  bfVfHierBlockDeinit(*blockPtr);
  bfMemFree(*blockPtr);
  *blockPtr = NULL;
}

static void bfVfHierBlockDeinitAndDealloc(BfVfHierBlock **blockPtr) {
  bfVfHierBlockDealloc(blockPtr);
}

static BfVfHierBlock *makeLeafWithOptionalSvd(
    BfTrimesh const *tm,
    BfQuadtreeNode const *rowNode,
    BfQuadtreeNode const *colNode,
    BfVfBlockMeta const *meta,
    BfReal eta,
    BfReal tol,
    BfSize leafMax,
    BfSize leafMin,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac);

static void bfVfSvdLeafApply(BfVfSvdLeaf const *leaf,
                             BfReal const      *x,
                             BfReal            *y,
                             BfSize             n);

/* --- Geometry / admissibility ----------------------------------------- */

///* Simple admissibility test in the stereographic plane, similar to your Cython _is_far_bbox */
static BfBool isFar(BfQuadtreeNode const *rowNode,
                    BfQuadtreeNode const *colNode,
                    BfReal                eta) {
  BfBbox2 br = bfQuadtreeNodeGetBbox(rowNode);
  BfBbox2 bc = bfQuadtreeNodeGetBbox(colNode);

  BfReal br_xmin = br.min[0], br_xmax = br.max[0];
  BfReal br_ymin = br.min[1], br_ymax = br.max[1];
  BfReal bc_xmin = bc.min[0], bc_xmax = bc.max[0];
  BfReal bc_ymin = bc.min[1], bc_ymax = bc.max[1];

  BfReal crx = 0.5*(br_xmin + br_xmax);
  BfReal cry = 0.5*(br_ymin + br_ymax);
  BfReal ccx = 0.5*(bc_xmin + bc_xmax);
  BfReal ccy = 0.5*(bc_ymin + bc_ymax);

  BfReal dxr = br_xmax - br_xmin;
  BfReal dyr = br_ymax - br_ymin;
  BfReal dxc = bc_xmax - bc_xmin;
  BfReal dyc = bc_ymax - bc_ymin;

  BfReal diam_r = bf_local_max(dxr, dyr);
  BfReal diam_c = bf_local_max(dxc, dyc);
  BfReal diam   = bf_local_max(diam_r, diam_c);

  BfReal dx = crx - ccx;
  BfReal dy = cry - ccy;
  BfReal dist = bf_local_sqrt(dx*dx + dy*dy);

  if (dist == 0) return BF_FALSE;

  return diam <= eta*dist;
}

/* Extract global face indices for a node into a BfSizeArray */
static void getNodeInds(BfQuadtreeNode *node,
                        BfQuadtree          const *qt,
                        BfSizeArray         *inds) {
//  bfSizeArrayClear(inds);

  BfSize i0 = bfTreeNodeGetFirstIndex(bfQuadtreeNodeToTreeNode(node));
  BfSize i1 = bfTreeNodeGetLastIndex(bfQuadtreeNodeToTreeNode(node));

  /* Perm lives in the underlying tree */
  BfPerm const *perm = bfTreeGetPerm(bfQuadtreeToTree(qt));

  for (BfSize i = i0; i < i1; ++i) {
    BfSize idx = bfPermGetIndex(perm, i);
    bfSizeArrayAppend(inds, idx);
  }
}

/* --- Leaf / block creation -------------------------------------------- */

/* Compute the basic geometric/size info for a (rowNode, colNode) pair.
 *
 *  - leafMax is only used to set `small`
 *  - if the block is empty (no rows or no cols), `empty` is true and mi=mj=0.
 */
static BfVfBlockMeta getBlockMeta(BfQuadtreeNode *rowNode,
                                  BfQuadtreeNode *colNode,
                                  BfSize          leafMax)
{
  BfVfBlockMeta meta;

  BfTreeNode const *ni = bfQuadtreeNodeToTreeNode(rowNode);
  BfTreeNode const *nj = bfQuadtreeNodeToTreeNode(colNode);

  meta.i0 = bfTreeNodeGetFirstIndex(ni);
  meta.i1 = bfTreeNodeGetLastIndex(ni);
  meta.j0 = bfTreeNodeGetFirstIndex(nj);
  meta.j1 = bfTreeNodeGetLastIndex(nj);

  if (meta.i1 <= meta.i0 || meta.j1 <= meta.j0) {
    meta.empty = true;
    meta.mi = meta.mj = 0;
    meta.leafI = meta.leafJ = false;
    meta.small = false;
    return meta;
  }

  meta.mi = meta.i1 - meta.i0;
  meta.mj = meta.j1 - meta.j0;

  meta.leafI = bfTreeNodeIsLeaf(ni);
  meta.leafJ = bfTreeNodeIsLeaf(nj);

  meta.small = (meta.mi <= leafMax) || (meta.mj <= leafMax);
  meta.empty = false;

  return meta;
}

static BfVfHierBlock *makeSparseLeaf(BfTrimesh const *tm,
                                     BfQuadtreeNode const *rowNode,
                                     BfQuadtreeNode const *colNode) {
  BfQuadtree const *qt = bfQuadtreeNodeGetQuadtree((BfQuadtreeNode *)rowNode);

  BfVfHierBlock *block = bfVfHierBlockNew();
  block->kind = BF_VF_HIER_BLOCK_SPARSE;

  BfVfSparseLeaf *leaf = &block->data.sparse;
  bfSizeArrayInitWithDefaultCapacity(&leaf->rowInds);
  bfSizeArrayInitWithDefaultCapacity(&leaf->colInds);

  /* Fill row/col index sets */
  getNodeInds(rowNode, qt, &leaf->rowInds);
  getNodeInds(colNode, qt, &leaf->colInds);

  /* Build sub-block via existing BF builder */
  leaf->mat = bfMatCsrRealNewViewFactorMatrixFromTrimesh(
      tm, &leaf->rowInds, &leaf->colInds);

  /* Decide once if CSR colind is already local 0..nA-1 or global faces */
  BfSize mA = bfMatGetNumRows(bfMatCsrRealToMat(leaf->mat));
  BfSize nA = bfMatGetNumCols(bfMatCsrRealToMat(leaf->mat));

  BF_ASSERT(mA == bfSizeArrayGetSize(&leaf->rowInds));
  BF_ASSERT(nA == bfSizeArrayGetSize(&leaf->colInds));

  BfSize const *rowptr = bfMatCsrRealGetRowptrConstPtr(leaf->mat);
  BfSize const *colind = bfMatCsrRealGetColindConstPtr(leaf->mat);

  BF_ASSERT(rowptr != NULL);
  BF_ASSERT(colind != NULL);

  BfSize nnz = rowptr[mA];
  BfSize maxCol = 0;
  for (BfSize k = 0; k < nnz; ++k) {
    if (colind[k] > maxCol)
      maxCol = colind[k];
  }

  /* If maxCol < nA, interpret as local 0..nA-1 */
  leaf->colsAreLocal = BF_FALSE;
  if (nA > 0 && maxCol < nA)
    leaf->colsAreLocal = BF_TRUE;

  return block;
}

/* Recursive builder */
static BfVfHierBlock *buildBlock(BfTrimesh const *tm,
                                 BfQuadtreeNode *rowNode,
                                 BfQuadtreeNode *colNode,
                                 BfReal          eta,
                                 BfSize          leafMax,
                                 BfSize          leafMin,
                                 BfReal          tol,
                                 BfSize          minSvdSize,
                                 BfReal          maxSvdRankFrac)
{
  BfVfBlockMeta meta = getBlockMeta(rowNode, colNode, leafMax);
  if (meta.empty)
    return NULL;

  BfSize mi = meta.mi;
  BfSize mj = meta.mj;
  bool leafI = meta.leafI;
  bool leafJ = meta.leafJ;
  bool small = meta.small;

  /* Leaf candidate */
  if ((leafI && leafJ) || (small && mi >= leafMin && mj >= leafMin)) {
    return makeLeafWithOptionalSvd(tm, rowNode, colNode,
                                   &meta, eta, tol,
                                   leafMax, leafMin,
                                   minSvdSize, maxSvdRankFrac);
  }

  /* Otherwise, split at least one side */
  BfVfHierBlock *nodeBlock = bfVfHierBlockNew();
  nodeBlock->kind = BF_VF_HIER_BLOCK_NODE;
  bfInitPtrArray(&nodeBlock->data.node.children, 4);

  BfTreeNode const *ni = bfQuadtreeNodeToTreeNode(rowNode);
  BfTreeNode const *nj = bfQuadtreeNodeToTreeNode(colNode);

  if (!leafI && !leafJ) {
    for (BfSize a = 0; a < bfTreeNodeGetMaxNumChildren(ni); ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfTreeNode *child_i = bfTreeNodeGetChild(ni, a);
      BfQuadtreeNode *nai = bfTreeNodeToQuadtreeNode(child_i);

      for (BfSize b = 0; b < bfTreeNodeGetMaxNumChildren(nj); ++b) {
        if (!bfTreeNodeHasChild(nj, b)) continue;
        BfTreeNode *child_j = bfTreeNodeGetChild(nj, b);
        BfQuadtreeNode *nbj = bfTreeNodeToQuadtreeNode(child_j);

        BfVfHierBlock *child =
          buildBlock(tm, nai, nbj, eta, leafMax, leafMin, tol, minSvdSize, maxSvdRankFrac);

        if (child != NULL)
          bfPtrArrayAppend(&nodeBlock->data.node.children, child);
      }
    }

  } else if (!leafI) {
    for (BfSize a = 0; a < bfTreeNodeGetMaxNumChildren(ni); ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfTreeNode *child_i = bfTreeNodeGetChild(ni, a);
      BfQuadtreeNode *nai = bfTreeNodeToQuadtreeNode(child_i);

      BfVfHierBlock *child =
        buildBlock(tm, nai, colNode, eta, leafMax, leafMin, tol, minSvdSize, maxSvdRankFrac);

      if (child != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, child);
    }

  } else { /* !leafJ */
    for (BfSize b = 0; b < bfTreeNodeGetMaxNumChildren(nj); ++b) {
      if (!bfTreeNodeHasChild(nj, b)) continue;
      BfTreeNode *child_j = bfTreeNodeGetChild(nj, b);
      BfQuadtreeNode *nbj = bfTreeNodeToQuadtreeNode(child_j);

      BfVfHierBlock *child =
        buildBlock(tm, rowNode, nbj, eta, leafMax, leafMin, tol, minSvdSize, maxSvdRankFrac);

      if (child != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, child);
    }
  }

  if (bfPtrArraySize(&nodeBlock->data.node.children) == 0) {
    bfVfHierBlockDeinitAndDealloc(&nodeBlock);
    return NULL;
  }

  return nodeBlock;
}

/* --- BfVfHier API ------------------------------------------------------ */

BfVfHier *bfVfHierNew(void) {
  BfVfHier *vfHier = bfMemAlloc(1, sizeof(BfVfHier));
  vfHier->trimesh = NULL;
  vfHier->root    = NULL;
  vfHier->n       = 0;
  return vfHier;
}

void bfVfHierInitFromTrimesh(BfVfHier       *vfHier,
                             BfTrimesh const *trimesh,
                             BfReal          eta,
                             BfSize          leafMax,
                             BfSize          leafMin)
{
  (void)vfHier;
  (void)trimesh;
  (void)eta;
  (void)leafMax;
  (void)leafMin;

  /* For now, this path is intentionally disabled.
   * Build the quadtree in Python/Cython and call bfVfHierInitFromQuadtree.
   */
  BF_ASSERT(0 && "bfVfHierInitFromTrimesh is not implemented; use bfVfHierInitFromQuadtree instead");
}

BfVfHier *bfVfHierNewFromTrimesh(BfTrimesh const *trimesh,
                                 BfReal          eta,
                                 BfSize          leafMax,
                                 BfSize          leafMin)
{
  (void)trimesh;
  (void)eta;
  (void)leafMax;
  (void)leafMin;

  BF_ASSERT(0 && "bfVfHierNewFromTrimesh is not implemented; use bfVfHierNewFromQuadtree instead");
  return NULL;
}

void bfVfHierInitFromQuadtree(BfVfHier        *vfHier,
                              BfTrimesh const *trimesh,
                              BfQuadtree      *quadtree,
                              BfReal           eta,
                              BfSize           leafMax,
                              BfSize           leafMin,
                              BfReal           tol,
                              BfSize           minSvdSize,
                              BfReal           maxSvdRankFrac)
{
  BF_ASSERT(vfHier != NULL);
  BF_ASSERT(trimesh != NULL);
  BF_ASSERT(quadtree != NULL);

  vfHier->trimesh = trimesh;
  vfHier->n       = bfTrimeshGetNumFaces(trimesh);

  BfTree         *tree     = bfQuadtreeToTree(quadtree);
  BfTreeNode     *rootNode = bfTreeGetRootNode(tree);
  BfQuadtreeNode *rootQt   = bfTreeNodeToQuadtreeNode(rootNode);

  vfHier->root = buildBlock(trimesh, rootQt, rootQt,
                            eta, leafMax, leafMin,
                            tol, minSvdSize, maxSvdRankFrac);

  #if BF_VF_HIER_TIME_SVD
    fprintf(stderr,
          "[vf_hier] total CSR->dense time: %.6f s, SVD time: %.6f s\n",
          g_vf_csr_to_dense_time, g_vf_svd_time);

    /* reset accumulators so multiple builds don't add up across runs */
    g_vf_csr_to_dense_time = 0.0;
    g_vf_svd_time          = 0.0;
  #endif
}

BfVfHier *bfVfHierNewFromQuadtree(BfTrimesh const *trimesh,
                        BfQuadtree      *quadtree,
                        BfReal           eta,
                        BfSize           leafMax,
                        BfSize           leafMin,
                        BfReal           tol,
                        BfSize           minSvdSize,
                        BfReal           maxSvdRankFrac)
{
  BfVfHier *vfHier = bfVfHierNew();
  bfVfHierInitFromQuadtree(vfHier, trimesh, quadtree,
                           eta, leafMax, leafMin, tol, minSvdSize, maxSvdRankFrac);
  return vfHier;
}

/* ============================================================
 * CSR-based hierarchical view-factor hierarchy
 * ============================================================ */

/* Forward declaration for local CSR submatrix builder */
static BfMatCsrReal *
bfMatCsrRealNewSubmatrixFromFaces(BfMatCsrReal const *A_par,
                                  BfSizeArray  const *rowFaces,
                                  BfSizeArray  const *colFaces);

/* Leaf builder using CSR submatrix instead of geometry */
static BfVfHierBlock *makeLeafFromCsrWithOptionalSvd(
    BfMatCsrReal const *A_par,
    BfQuadtreeNode const *rowNode,
    BfQuadtreeNode const *colNode,
    BfVfBlockMeta const *meta,
    BfReal eta,
    BfReal tol,
    BfSize leafMax,
    BfSize leafMin,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac)
{
  BfQuadtree const *qt = bfQuadtreeNodeGetQuadtree((BfQuadtreeNode *)rowNode);

  if (meta->empty)
    return NULL;

  BfSize mi = meta->mi;
  BfSize mj = meta->mj;

  if (mi < leafMin || mj < leafMin)
    return NULL;

  /* Allocate the block and build row/col index arrays */
  BfVfHierBlock *block = bfVfHierBlockNew();

  BfSizeArray rowInds, colInds;
  bfSizeArrayInitWithDefaultCapacity(&rowInds);
  bfSizeArrayInitWithDefaultCapacity(&colInds);

  getNodeInds((BfQuadtreeNode *)rowNode, qt, &rowInds);
  getNodeInds((BfQuadtreeNode *)colNode, qt, &colInds);

  /* Build CSR sub-block using the helper above */
  BfMatCsrReal *Acsr =
    bfMatCsrRealNewSubmatrixFromFaces(A_par, &rowInds, &colInds);

  if (Acsr == NULL) {
    bfSizeArrayDeinit(&rowInds);
    bfSizeArrayDeinit(&colInds);
    bfVfHierBlockDeinitAndDealloc(&block);
    return NULL;
  }

  BfMat *A = bfMatCsrRealToMat(Acsr);
  BfSize mA = bfMatGetNumRows(A);
  BfSize nA = bfMatGetNumCols(A);

  /* Decide far vs near using geometry of nodes, like the trimesh path */
  BfBool far = isFar(rowNode, colNode, eta);

  /* NEAR or SVD disabled: keep CSR leaf (cols are LOCAL by construction) */
  if (!far || minSvdSize == 0) {
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    return block;
  }

  /* --- NEW: skip SVD on small far blocks, like trimesh path --- */
  unsigned long long blockSize =
    (unsigned long long)mA * (unsigned long long)nA;

  if (blockSize < (unsigned long long)minSvdSize) {
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    return block;
  }

  /* FAR + SVD enabled: decide whether SVD is worthwhile */

  /* 1) Memory estimate for CSR (same model as makeLeafWithOptionalSvd) */
  BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(Acsr);
  BfSize const *ci = bfMatCsrRealGetColindConstPtr(Acsr);
  BfReal const *da = bfMatCsrRealGetDataConstPtr(Acsr);

  BF_ASSERT(rp && ci && da);

  BfSize nnz = rp[mA];
  double bytesCsr = 8.0*nnz          /* data */
                  + 8.0*nnz          /* colind */
                  + 8.0*(mA + 1);    /* rowptr */

  /* SVD on CSR via ARPACK */
  BfTruncSpec truncSpec;
  truncSpec.usingTol = 1;
  truncSpec.tol = tol;

  #if BF_VF_HIER_TIME_SVD
    double t_svd_start = bfVfHierNowSecs();
  #endif

  BfMat *U = NULL, *VT = NULL;
  BfMatDiagReal *S = NULL;

  BfBackend backend = BF_BACKEND_ARPACK;
  BfBool truncated =
    bfGetTruncatedSvd(bfMatCsrRealToMat(Acsr), &U, &S, &VT,
                      &truncSpec, backend);

  #if BF_VF_HIER_TIME_SVD
    double t_svd_end = bfVfHierNowSecs();
    g_vf_svd_time += t_svd_end - t_svd_start;
  #endif

  (void)truncated; /* optional for logging */

  if (U == NULL || S == NULL || VT == NULL) {
    /* Fall back to sparse leaf on failure */
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    return block;
  }

  BfSize k = bfMatGetNumCols(U);  /* rank */
  BfSize minDim = mA < nA ? mA : nA;
  double rankFrac = minDim > 0 ? (double)k / (double)minDim : 1.0;

  /* 3) reject SVD if rank is too high */
  if (rankFrac > maxSvdRankFrac) {
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;

    bfMatDelete(&U);
    bfMatDiagRealDeinitAndDealloc(&S);
    bfMatDelete(&VT);
    return block;
  }

  /* 4) SVD memory estimate vs CSR memory */
  double bytesSvd =
    8.0*((double)mA*k + (double)nA*k + (double)k); /* U + VT + diag(S) */

  if (bytesSvd >= bytesCsr) {
    /* SVD not worth it; keep sparse leaf */
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;

    bfMatDelete(&U);
    bfMatDiagRealDeinitAndDealloc(&S);
    bfMatDelete(&VT);
    return block;
  }

  /* 5) Build MatProduct P = U S VT (like in makeLeafWithOptionalSvd) */
  BfMatProduct *Pprod = bfMatProductNew();
  bfMatProductInit(Pprod);
  bfMatProductPostMultiply(Pprod, U);
  bfMatProductPostMultiply(Pprod, bfMatDiagRealToMat(S));
  bfMatProductPostMultiply(Pprod, VT);

  BfMat *P = bfMatProductToMat(Pprod);

  /* SVD leaf: store row/col index arrays and P */
  block->kind = BF_VF_HIER_BLOCK_SVD;
  block->data.svd.rowInds = rowInds;  /* move ownership */
  block->data.svd.colInds = colInds;  /* move ownership */
  block->data.svd.mat     = P;
  block->data.svd.rank    = k;
  block->data.svd.work    = NULL;
  block->data.svd.workLen = 0;

  bfMatCsrRealDeinitAndDealloc(&Acsr);

  return block;
}

/* Recursive builder from parent CSR */
static BfVfHierBlock *buildBlockFromCsr(
    BfMatCsrReal const *A_par,
    BfQuadtreeNode *rowNode,
    BfQuadtreeNode *colNode,
    BfReal eta,
    BfSize leafMax,
    BfSize leafMin,
    BfReal tol,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac)
{
  BfVfBlockMeta meta = getBlockMeta(rowNode, colNode, leafMax);
  if (meta.empty)
    return NULL;

  BfSize mi = meta.mi, mj = meta.mj;
  bool leafI = meta.leafI, leafJ = meta.leafJ, small = meta.small;

  if ((leafI && leafJ) || (small && mi >= leafMin && mj >= leafMin)) {
    return makeLeafFromCsrWithOptionalSvd(A_par, rowNode, colNode, &meta,
                                          eta, tol, leafMax, leafMin,
                                          minSvdSize, maxSvdRankFrac);
  }

  BfVfHierBlock *nodeBlock = bfVfHierBlockNew();
  nodeBlock->kind = BF_VF_HIER_BLOCK_NODE;
  bfInitPtrArray(&nodeBlock->data.node.children, 4);

  BfTreeNode const *ni = bfQuadtreeNodeToTreeNode(rowNode);
  BfTreeNode const *nj = bfQuadtreeNodeToTreeNode(colNode);

  if (!leafI && !leafJ) {
    for (BfSize a = 0; a < bfTreeNodeGetMaxNumChildren(ni); ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfTreeNode *ci = bfTreeNodeGetChild(ni, a);
      BfQuadtreeNode *nai = bfTreeNodeToQuadtreeNode(ci);

      for (BfSize b = 0; b < bfTreeNodeGetMaxNumChildren(nj); ++b) {
        if (!bfTreeNodeHasChild(nj, b)) continue;
        BfTreeNode *cj = bfTreeNodeGetChild(nj, b);
        BfQuadtreeNode *nbj = bfTreeNodeToQuadtreeNode(cj);

        BfVfHierBlock *child =
          buildBlockFromCsr(A_par, nai, nbj, eta,
                            leafMax, leafMin, tol,
                            minSvdSize, maxSvdRankFrac);
        if (child != NULL)
          bfPtrArrayAppend(&nodeBlock->data.node.children, child);
      }
    }

  } else if (!leafI) {
    for (BfSize a = 0; a < bfTreeNodeGetMaxNumChildren(ni); ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfTreeNode *ci = bfTreeNodeGetChild(ni, a);
      BfQuadtreeNode *nai = bfTreeNodeToQuadtreeNode(ci);

      BfVfHierBlock *child =
        buildBlockFromCsr(A_par, nai, colNode,
                          eta, leafMax, leafMin,
                          tol, minSvdSize, maxSvdRankFrac);
      if (child != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, child);
    }

  } else { /* !leafJ */
    for (BfSize b = 0; b < bfTreeNodeGetMaxNumChildren(nj); ++b) {
      if (!bfTreeNodeHasChild(nj, b)) continue;
      BfTreeNode *cj = bfTreeNodeGetChild(nj, b);
      BfQuadtreeNode *nbj = bfTreeNodeToQuadtreeNode(cj);

      BfVfHierBlock *child =
        buildBlockFromCsr(A_par, rowNode, nbj,
                          eta, leafMax, leafMin,
                          tol, minSvdSize, maxSvdRankFrac);
      if (child != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, child);
    }
  }

  if (bfPtrArraySize(&nodeBlock->data.node.children) == 0) {
    bfVfHierBlockDeinitAndDealloc(&nodeBlock);
    return NULL;
  }

  return nodeBlock;
}

/* Top-level initializer from CSR + quadtree */
void bfVfHierInitFromCsrAndQuadtree(BfVfHier     *vfHier,
                                    BfMatCsrReal *Afull,
                                    BfQuadtree   *quadtree,
                                    BfReal        eta,
                                    BfSize        leafMax,
                                    BfSize        leafMin,
                                    BfReal        tol,
                                    BfSize        minSvdSize,
                                    BfReal        maxSvdRankFrac)
{
  BF_ASSERT(vfHier && Afull && quadtree);

  BfMat *A_base = bfMatCsrRealToMat(Afull);
  BfSize nRows  = bfMatGetNumRows(A_base);
  BfSize nCols  = bfMatGetNumCols(A_base);

  /* Basic consistency: FF should be square and compatible with quadtree */
  if (nRows != nCols) {
    fprintf(stderr,
            "[vf_hier] bfVfHierInitFromCsrAndQuadtree: Afull not square (%lu x %lu)\n",
            (unsigned long)nRows, (unsigned long)nCols);
  }

  vfHier->trimesh = NULL;  /* no geometry in this path */
  vfHier->n       = nRows;

  BfTree *tree = bfQuadtreeToTree(quadtree);
  BfTreeNode *rootNode = bfTreeGetRootNode(tree);
  BfQuadtreeNode *rootQt = bfTreeNodeToQuadtreeNode(rootNode);

  /* Optional debug: print root index range */
#if 1
  {
    BfVfBlockMeta rootMeta = getBlockMeta(rootQt, rootQt, leafMax);
    fprintf(stderr,
            "[vf_hier] CSR init: nRows=%lu, root i=[%lu,%lu), j=[%lu,%lu)\n",
            (unsigned long)nRows,
            (unsigned long)rootMeta.i0, (unsigned long)rootMeta.i1,
            (unsigned long)rootMeta.j0, (unsigned long)rootMeta.j1);
  }
#endif

  vfHier->root = buildBlockFromCsr(Afull, rootQt, rootQt,
                                   eta, leafMax, leafMin,
                                   tol, minSvdSize, maxSvdRankFrac);

  if (vfHier->root == NULL) {
    fprintf(stderr,
            "[vf_hier] bfVfHierInitFromCsrAndQuadtree: root block is NULL "
            "(maybe all leaves too small or CSR submatrix failed)\n");
  }

#if BF_VF_HIER_TIME_SVD
  fprintf(stderr,
          "[vf_hier] total CSR->dense time: %.6f s, SVD time: %.6f s\n",
          g_vf_csr_to_dense_time, g_vf_svd_time);
  g_vf_csr_to_dense_time = 0.0;
  g_vf_svd_time          = 0.0;
#endif
}

/* Convenience constructor */
BfVfHier *bfVfHierNewFromCsrAndQuadtree(BfMatCsrReal *Afull,
                                         BfQuadtree   *quadtree,
                                         BfReal        eta,
                                         BfSize        leafMax,
                                         BfSize        leafMin,
                                         BfReal        tol,
                                         BfSize        minSvdSize,
                                         BfReal        maxSvdRankFrac)
{
  BfVfHier *vf = bfVfHierNew();
  bfVfHierInitFromCsrAndQuadtree(vf, Afull, quadtree,
                                 eta, leafMax, leafMin,
                                 tol, minSvdSize, maxSvdRankFrac);
  return vf;
}

/* Sparse leaf MVP using explicit CSR kernel, robust to local/global col indices */
static void bfVfSparseLeafApply(BfVfSparseLeaf const *leaf,
                                BfReal const         *x,
                                BfReal               *y,
                                BfSize                n)
{
  BfMat const *A_mat = bfMatCsrRealToMat(leaf->mat);
  BfSize mA = bfMatGetNumRows(A_mat);
  BfSize nA = bfMatGetNumCols(A_mat);

  BF_ASSERT(mA == bfSizeArrayGetSize((BfSizeArray *)&leaf->rowInds));
  BF_ASSERT(nA == bfSizeArrayGetSize((BfSizeArray *)&leaf->colInds));

  BfSize const *rowptr = bfMatCsrRealGetRowptrConstPtr(leaf->mat);
  BfSize const *colind = bfMatCsrRealGetColindConstPtr(leaf->mat);
  BfReal const *data   = bfMatCsrRealGetDataConstPtr(leaf->mat);

  BF_ASSERT(rowptr != NULL);
  BF_ASSERT(colind != NULL);
  BF_ASSERT(data   != NULL);

  /* fast pointer to row/col index arrays */
  BfSizeArray const *rowInds = &leaf->rowInds;
  BfSizeArray const *colInds = &leaf->colInds;

  if (leaf->colsAreLocal) {
    /* Local-column case: colind[k] indexes into colInds */
    for (BfSize i = 0; i < mA; ++i) {
      BfSize row_start = rowptr[i];
      BfSize row_end   = rowptr[i + 1];

      BF_ASSERT(row_end >= row_start);
      BF_ASSERT(row_end <= rowptr[mA]);

      BfReal acc = 0;

      for (BfSize k = row_start; k < row_end; ++k) {
        BfSize cLocal = colind[k];
        BF_ASSERT(cLocal < bfSizeArrayGetSize((BfSizeArray *)colInds));

        BfSize globalCol = bfSizeArrayGet((BfSizeArray *)colInds, cLocal);
        BF_ASSERT(globalCol < n);

        acc += data[k] * x[globalCol];
      }

      BfSize globalRow = bfSizeArrayGet((BfSizeArray *)rowInds, i);
      BF_ASSERT(globalRow < n);

      y[globalRow] += acc;
    }
  } else {
    /* Global-column case: colind[k] is already a global column index */
    for (BfSize i = 0; i < mA; ++i) {
      BfSize row_start = rowptr[i];
      BfSize row_end   = rowptr[i + 1];

      BF_ASSERT(row_end >= row_start);
      BF_ASSERT(row_end <= rowptr[mA]);

      BfReal acc = 0;

      for (BfSize k = row_start; k < row_end; ++k) {
        BfSize globalCol = colind[k];
        BF_ASSERT(globalCol < n);
        acc += data[k] * x[globalCol];
      }

      BfSize globalRow = bfSizeArrayGet((BfSizeArray *)rowInds, i);
      BF_ASSERT(globalRow < n);

      y[globalRow] += acc;
    }
  }
}

static void bfVfHierBlockApply(BfVfHierBlock const *block,
                               BfReal const        *x,
                               BfReal              *y,
                               BfSize               n) {
  if (block == NULL) return;

  switch (block->kind) {
  case BF_VF_HIER_BLOCK_SPARSE:
    bfVfSparseLeafApply(&block->data.sparse, x, y, n);
    break;

  case BF_VF_HIER_BLOCK_SVD:
    bfVfSvdLeafApply(&block->data.svd, x, y, n);
    break;

  case BF_VF_HIER_BLOCK_NODE:
    for (BfSize i = 0; i < bfPtrArraySize(&block->data.node.children); ++i) {
      BfVfHierBlock *child = bfPtrArrayGet(&block->data.node.children, i);
      bfVfHierBlockApply(child, x, y, n);
    }
    break;

  case BF_VF_HIER_BLOCK_NONE:
  default:
    break;
  }
}

static void bfVfSvdLeafApply(BfVfSvdLeaf const *leaf,
                             BfReal const      *x,
                             BfReal            *y,
                             BfSize             n)
{
  BfSize m  = bfSizeArrayGetSize((BfSizeArray *)&leaf->rowInds);
  BfSize mj = bfSizeArrayGetSize((BfSizeArray *)&leaf->colInds);

  /* Ensure workspace is large enough */
  BfReal *xSub;
  if (leaf->work != NULL && leaf->workLen >= mj) {
    xSub = leaf->work;
  } else {
    /* cast away const just for realloc; logically this is mutable state */
    BfVfSvdLeaf *leaf_mut = (BfVfSvdLeaf *)leaf;
    if (leaf_mut->work != NULL)
      bfMemFree(leaf_mut->work);
    leaf_mut->work = bfMemAlloc(mj, sizeof(BfReal));
    leaf_mut->workLen = mj;
    xSub = leaf_mut->work;
  }

  /* Gather x_sub into workspace */
  for (BfSize j = 0; j < mj; ++j) {
    BfSize col = bfSizeArrayGet((BfSizeArray *)&leaf->colInds, j);
    BF_ASSERT(col < n);
    xSub[j] = x[col];
  }

  /* Wrap xSub in a temporary VecReal view */
  BfVecReal *vx = bfVecRealNewViewFromPtr(mj, xSub, 1);

  /* ySub = (U S V^T) * xSub via MatProduct */
  BfVec *vyBase = bfMatMulVec(leaf->mat, bfVecRealToVec(vx));
  BfVecReal *vy = bfVecToVecReal(vyBase);
  BfReal const *yData = bfVecRealGetDataPtr(vy);

  /* Scatter-add into global y */
  for (BfSize i = 0; i < m; ++i) {
    BfSize row = bfSizeArrayGet((BfSizeArray *)&leaf->rowInds, i);
    BF_ASSERT(row < n);
    y[row] += yData[i];
  }

  bfVecRealDeinitAndDealloc(&vx);
  bfVecRealDeinitAndDealloc(&vy);
}

void bfVfHierApply(BfVfHier const *vfHier,
                   BfReal const   *x,
                   BfReal         *y) {
  BF_ASSERT(vfHier != NULL);
  BF_ASSERT(vfHier->root != NULL);

  /* y is assumed allocated and initialized by caller */
  bfVfHierBlockApply(vfHier->root, x, y, vfHier->n);
}

void bfVfHierDeinit(BfVfHier *vfHier) {
  if (vfHier == NULL) return;
  if (vfHier->root != NULL)
    bfVfHierBlockDeinitAndDealloc(&vfHier->root);
  vfHier->trimesh = NULL;
  vfHier->n = 0;
}

void bfVfHierDealloc(BfVfHier **vfHierPtr) {
  if (vfHierPtr == NULL || *vfHierPtr == NULL) return;
  bfVfHierDeinit(*vfHierPtr);
  bfMemFree(*vfHierPtr);
  *vfHierPtr = NULL;
}

void bfVfHierDeinitAndDealloc(BfVfHier **vfHierPtr) {
  bfVfHierDealloc(vfHierPtr);
}

static void reindexCsrColsToLocal(BfMatCsrReal *Acsr,
                                  BfSizeArray const *colFaces,
                                  BfSize nFaces) {
  BfMat *A = bfMatCsrRealToMat(Acsr);
  BfSize mA = bfMatGetNumRows(A);
  BfSize nA = bfMatGetNumCols(A);

  BfSize const *rowptr = bfMatCsrRealGetRowptrConstPtr(Acsr);
  BfSize const *colind = bfMatCsrRealGetColindConstPtr(Acsr);

  BF_ASSERT(rowptr && colind);

  BfSize nnz = rowptr[mA];
  BfSize maxCol = 0;
  for (BfSize k = 0; k < nnz; ++k)
    if (colind[k] > maxCol) maxCol = colind[k];

  if (maxCol < nA)
    return;  /* already local */

  BfSize *globalToLocal = bfMemAlloc(nFaces, sizeof(BfSize));
  for (BfSize t = 0; t < nFaces; ++t)
    globalToLocal[t] = BF_SIZE_BAD_VALUE;

  for (BfSize j = 0; j < nA; ++j) {
    BfSize g = bfSizeArrayGet((BfSizeArray *)colFaces, j);
    globalToLocal[g] = j;
  }

  for (BfSize k = 0; k < nnz; ++k) {
    BfSize g = colind[k];
    BF_ASSERT(g < nFaces);
    BfSize local = globalToLocal[g];
    BF_ASSERT(local < nA);
    ((BfSize *)colind)[k] = local;  /* const-cast ok here */
  }

  bfMemFree(globalToLocal);
}

static BfVfHierBlock *makeLeafWithOptionalSvd(
    BfTrimesh const *tm,
    BfQuadtreeNode const *rowNode,
    BfQuadtreeNode const *colNode,
    BfVfBlockMeta const *meta,
    BfReal eta,
    BfReal tol,
    BfSize leafMax,
    BfSize leafMin,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac)
{
  BfQuadtree const *qt = bfQuadtreeNodeGetQuadtree(rowNode);

//  /* Compute index ranges and sizes exactly as getBlockMeta does */
//  BfVfBlockMeta meta = getBlockMeta((BfQuadtreeNode *)rowNode,
//                                    (BfQuadtreeNode *)colNode,
//                                    leafMax);
  if (meta->empty)
    return NULL;

  BfSize mi = meta->mi;
  BfSize mj = meta->mj;

  if (mi < leafMin || mj < leafMin)
    return NULL;

  /* Build row/col index arrays */
  BfVfHierBlock *block = bfVfHierBlockNew();

  /* Common row/col index sets (global) */
  BfSizeArray rowInds, colInds;
  bfSizeArrayInitWithDefaultCapacity(&rowInds);
  bfSizeArrayInitWithDefaultCapacity(&colInds);

  getNodeInds((BfQuadtreeNode *)rowNode, qt, &rowInds);
  getNodeInds((BfQuadtreeNode *)colNode, qt, &colInds);

  /* Build CSR block */
  BfMatCsrReal *Acsr =
    bfMatCsrRealNewViewFactorMatrixFromTrimesh(tm, &rowInds, &colInds);
  if (Acsr == NULL) {
    bfSizeArrayDeinit(&rowInds);
    bfSizeArrayDeinit(&colInds);
    bfVfHierBlockDeinitAndDealloc(&block);
    return NULL;
  }

  BfMat *A = bfMatCsrRealToMat(Acsr);
  BfSize mA = bfMatGetNumRows(A);
  BfSize nA = bfMatGetNumCols(A);

  /* Decide far vs near */
  BfBool far = isFar(rowNode, colNode, eta);

  /* NEAR or SVD disabled: keep CSR leaf with whatever col indexing we got */
  if (!far || minSvdSize == 0) {
    block->kind                         = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds          = rowInds;
    block->data.sparse.colInds          = colInds;
    block->data.sparse.mat              = Acsr;
    /* detect local vs global once, like makeSparseLeaf */
    BfSize const *rp     = bfMatCsrRealGetRowptrConstPtr(Acsr);
    BfSize const *colind = bfMatCsrRealGetColindConstPtr(Acsr);
    BfSize nnz           = rp[mA];
    BfSize maxCol = 0;
    for (BfSize k = 0; k < nnz; ++k)
      if (colind[k] > maxCol) maxCol = colind[k];
    block->data.sparse.colsAreLocal = (nA > 0 && maxCol < nA) ? BF_TRUE : BF_FALSE;
    return block;
  }

  /* FAR + SVD enabled: now you may need local columns */
  BfSize nFaces = bfTrimeshGetNumFaces(tm);
  reindexCsrColsToLocal(Acsr, &colInds, nFaces);

  /* FAR and SVD enabled: decide whether to SVD-compress */
  unsigned long long blockSize =
    (unsigned long long)mA * (unsigned long long)nA;

  if (blockSize < (unsigned long long)minSvdSize) {
    /* too small for SVD: keep CSR leaf */
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds    = rowInds;
    block->data.sparse.colInds    = colInds;
    block->data.sparse.mat        = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    return block;
  }

  /* 1) Memory estimate for CSR */
  BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(Acsr);
  BfSize nnz = rp[mA];

  double bytesCsr = 8.0*nnz + 8.0*nnz + 8.0*(mA + 1);  /* data + colind + rowptr */

//  fprintf(stderr, "[makeLeaf] A=%p type=%d vtbl=%p ToType=%p\n",
//          (void*)A, (int)bfMatGetType(A),
//          (void*)A->vtbl,
//          A->vtbl ? (void*)A->vtbl->ToType : NULL);

  /* 2) Dense + truncated SVD: explicitly convert CSR -> dense
   * (MatCsrReal does not implement ->ToType, so bfMatToType would segfault).
   */
  #if BF_VF_HIER_TIME_SVD
    double t_csr_start = bfVfHierNowSecs();
  #endif

  BfMatDenseReal *A_dense_real = bfMatDenseRealNew();
  bfMatDenseRealInitZeros(A_dense_real, mA, nA);

  BfReal *denseData = A_dense_real->data;

  BfSize const *colind  = bfMatCsrRealGetColindConstPtr(Acsr);
  BfReal const *csrData = bfMatCsrRealGetDataConstPtr(Acsr);

  BF_ASSERT(rp != NULL);
  BF_ASSERT(colind != NULL);
  BF_ASSERT(csrData != NULL);

  /* Fill dense matrix in row-major, consistent with bfMatDenseRealInit */
  for (BfSize i = 0; i < mA; ++i) {
    for (BfSize k = rp[i]; k < rp[i + 1]; ++k) {
      BfSize j = colind[k];
      BF_ASSERT(j < nA);
      denseData[i*nA + j] = csrData[k];
    }
  }

  #if BF_VF_HIER_TIME_SVD
    double t_csr_end = bfVfHierNowSecs();
    g_vf_csr_to_dense_time += t_csr_end - t_csr_start;
  #endif

  BfMat *A_dense = bfMatDenseRealToMat(A_dense_real);

  BfTruncSpec truncSpec;
  truncSpec.usingTol = 1;
  truncSpec.tol = tol;

  BfMat *U = NULL, *VT = NULL;
  BfMatDiagReal *S = NULL;

  #if BF_VF_HIER_TIME_SVD
    double t_svd_start = bfVfHierNowSecs();
  #endif

//  BfBackend backend = BF_BACKEND_LAPACK;
//  BfBool truncated =
//    bfGetTruncatedSvd((BfMat const *)A_dense, &U, &S, &VT,
//                      &truncSpec, backend);
  BfBackend backend = BF_BACKEND_ARPACK;
  BfBool truncated =
    bfGetTruncatedSvd(bfMatCsrRealToMat(Acsr), &U, &S, &VT,
                      &truncSpec, backend);

  #if BF_VF_HIER_TIME_SVD
    double t_svd_end = bfVfHierNowSecs();
    g_vf_svd_time += t_svd_end - t_svd_start;
  #endif

  (void)truncated; /* you can print/log if desired */

  if (U == NULL || S == NULL || VT == NULL) {
    /* Fall back to sparse leaf on failure */
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    bfMatDelete(&A_dense);
    return block;
  }

  BfSize k = bfMatGetNumCols(U);  /* rank */

  BfSize minDim = mA < nA ? mA : nA;
  double rankFrac = minDim > 0 ? (double)k / (double)minDim : 1.0;

  /* 3) reject SVD if rank is too high */
  if (rankFrac > maxSvdRankFrac) {
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;

    /* free SVD pieces */
    bfMatDelete(&U);
    bfMatDiagRealDeinitAndDealloc(&S);
    bfMatDelete(&VT);
    bfMatDelete(&A_dense);
    return block;
  }

  /* 4) SVD memory estimate */
  double bytesSvd =
    8.0*((double)mA*k + (double)nA*k + (double)k); /* U + VT + diag(S) */

  if (bytesSvd >= bytesCsr) {
    /* SVD not worth it; keep sparse leaf */
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;

    bfMatDelete(&U);
    bfMatDiagRealDeinitAndDealloc(&S);
    bfMatDelete(&VT);
    bfMatDelete(&A_dense);
    return block;
  }

  /* 5) Build MatProduct P = U S VT */
  BfMatProduct *Pprod = bfMatProductNew();
  bfMatProductInit(Pprod);
  bfMatProductPostMultiply(Pprod, U);
  bfMatProductPostMultiply(Pprod, bfMatDiagRealToMat(S));
  bfMatProductPostMultiply(Pprod, VT);

  BfMat *P = bfMatProductToMat(Pprod);

  /* SVD leaf: store row/col index arrays and P */
  block->kind = BF_VF_HIER_BLOCK_SVD;
  block->data.svd.rowInds = rowInds;  /* move ownership */
  block->data.svd.colInds = colInds;  /* move ownership */
  block->data.svd.mat     = P;
  block->data.svd.rank    = k;
  block->data.svd.work    = NULL;
  block->data.svd.workLen = 0;

  /* rowInds/colInds are now owned by the SVD leaf; don't deinit them here */

  bfMatCsrRealDeinitAndDealloc(&Acsr);
  bfMatDelete(&A_dense);

  return block;
}

/* ============================================================
 * CSR SUBMATRIX BUILDER (from an existing parent CSR)
 * ============================================================ */

/* Build a CSR submatrix A_sub = A_par[rowFaces, colFaces], with
 * LOCAL column indexing 0..|colFaces|-1.
 *
 * NOTE: this assumes the usual BF CSR construction helpers
 *   - bfMatCsrRealNew
 *   - bfMatCsrRealInitEmpty(A, m, n, nnz_hint)
 *   - bfMatCsrRealAppendEntry(A, i, j, value)
 *   - bfMatCsrRealCompress(A)
 * If your names differ slightly, just adapt them.
 */
/* Build a CSR submatrix A_sub = A_par[rowFaces, colFaces], with
 * LOCAL column indexing 0..|colFaces|-1.
 *
 * This version uses BfSizeArray / BfRealArray and bfMatCsrRealNewFromArrays,
 * matching the pattern used in bfMatCsrRealNewViewFactorMatrixFromTrimesh.
 */
static BfMatCsrReal *
bfMatCsrRealNewSubmatrixFromFaces(BfMatCsrReal const *A_par,
                                  BfSizeArray  const *rowFaces,
                                  BfSizeArray  const *colFaces)
{
  BF_ASSERT(A_par    != NULL);
  BF_ASSERT(rowFaces != NULL);
  BF_ASSERT(colFaces != NULL);

  /* Block dimensions: how many selected rows/cols */
  BfSize m = bfSizeArrayGetSize((BfSizeArray *)rowFaces);
  BfSize n = bfSizeArrayGetSize((BfSizeArray *)colFaces);

  /* Parent matrix info */
  BfMat *A_base = bfMatCsrRealToMat((BfMatCsrReal *)A_par);
  BfSize nRows  = bfMatGetNumRows(A_base);
  BfSize nCols  = bfMatGetNumCols(A_base);  /* assume square FF */

  /* Trivial empty block */
  if (m == 0 || n == 0)
    return bfMatCsrRealNewFromArrays(0, 0,
                                     bfSizeArrayNewWithDefaultCapacity(),
                                     bfSizeArrayNewWithDefaultCapacity(),
                                     bfRealArrayNewWithDefaultCapacity(),
                                     BF_POLICY_STEAL);

  /* Quick sanity: faces indices must be < dims */
  for (BfSize i = 0; i < m; ++i) {
    BfSize gRow = bfSizeArrayGet((BfSizeArray *)rowFaces, i);
    if (gRow >= nRows) {
      fprintf(stderr,
              "[bf] bfMatCsrRealNewSubmatrixFromFaces: BAD row index %lu >= %lu\n",
              (unsigned long)gRow, (unsigned long)nRows);
      return NULL;
    }
  }
  for (BfSize j = 0; j < n; ++j) {
    BfSize gCol = bfSizeArrayGet((BfSizeArray *)colFaces, j);
    if (gCol >= nCols) {
      fprintf(stderr,
              "[bf] bfMatCsrRealNewSubmatrixFromFaces: BAD col index %lu >= %lu\n",
              (unsigned long)gCol, (unsigned long)nCols);
      return NULL;
    }
  }

  /* Build global->local column map: size = nCols, init to BAD */
  BfSize *globalToLocal = bfMemAlloc(nCols, sizeof(BfSize));
  if (globalToLocal == NULL)
    return NULL;

  for (BfSize i = 0; i < nCols; ++i)
    globalToLocal[i] = BF_SIZE_BAD_VALUE;

  for (BfSize j = 0; j < n; ++j) {
    BfSize g = bfSizeArrayGet((BfSizeArray *)colFaces, j);
    BF_ASSERT(g < nCols);
    globalToLocal[g] = j;  /* global col index -> local col index */
  }

  /* Parent CSR data */
  BfSize const *rp_par = bfMatCsrRealGetRowptrConstPtr(A_par);
  BfSize const *ci_par = bfMatCsrRealGetColindConstPtr(A_par);
  BfReal const *da_par = bfMatCsrRealGetDataConstPtr(A_par);

  BF_ASSERT(rp_par != NULL);
  BF_ASSERT(ci_par != NULL);
  BF_ASSERT(da_par != NULL);

  /* Dynamic CSR arrays for the submatrix */
  BfSizeArray *rowptr = bfSizeArrayNewWithDefaultCapacity();
  BfSizeArray *colind = bfSizeArrayNewWithDefaultCapacity();
  BfRealArray *data   = bfRealArrayNewWithDefaultCapacity();

  /* rowptr[0] = 0 */
  bfSizeArrayAppend(rowptr, 0);

  BfSize nnz_so_far = 0;

  /* Build each sub-row */
  for (BfSize i = 0; i < m; ++i) {
    /* Global row index in parent CSR */
    BfSize gRow = bfSizeArrayGet((BfSizeArray *)rowFaces, i);
    if (gRow + 1 > nRows) {
      fprintf(stderr,
              "[bf] bfMatCsrRealNewSubmatrixFromFaces: gRow %lu + 1 > nRows %lu\n",
              (unsigned long)gRow, (unsigned long)nRows);
      bfMemFree(globalToLocal);
      bfSizeArrayDeinitAndDealloc(&rowptr);
      bfSizeArrayDeinitAndDealloc(&colind);
      bfRealArrayDeinitAndDealloc(&data);
      return NULL;
    }

    BfSize r0 = rp_par[gRow];
    BfSize r1 = rp_par[gRow + 1];

    for (BfSize k = r0; k < r1; ++k) {
      BfSize gCol = ci_par[k];
      if (gCol >= nCols) {
        fprintf(stderr,
                "[bf] bfMatCsrRealNewSubmatrixFromFaces: BAD gCol %lu >= %lu\n",
                (unsigned long)gCol, (unsigned long)nCols);
        bfMemFree(globalToLocal);
        bfSizeArrayDeinitAndDealloc(&rowptr);
        bfSizeArrayDeinitAndDealloc(&colind);
        bfRealArrayDeinitAndDealloc(&data);
        return NULL;
      }

      BfSize lCol = globalToLocal[gCol];
      if (lCol == BF_SIZE_BAD_VALUE)
        continue;  /* column not in colFaces: skip */

      /* Append entry (i, lCol, value) */
      bfSizeArrayAppend(colind, lCol);
      bfRealArrayAppend(data, da_par[k]);
      ++nnz_so_far;
    }

    /* rowptr[i+1] = current nnz count */
    bfSizeArrayAppend(rowptr, nnz_so_far);
  }

  bfMemFree(globalToLocal);

  /* Convert dynamic arrays to a proper CSR matrix (takes ownership) */
  BfMatCsrReal *A_sub =
    bfMatCsrRealNewFromArrays(m, n, rowptr, colind, data, BF_POLICY_STEAL);

  return A_sub;
}




