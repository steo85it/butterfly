/*
 * vf_hier_build.c
 * --------------
 * Construction of the VfHier block tree from either:
 *   - direct geometric traversal / raytracing, and/or
 *   - mid-level CSR blocks (parent CSR -> recursively sliced children)
 *
 * Responsibilities:
 *   - all hierarchy building logic and heuristics (SVD gating, flattening, etc.)
 *   - progress instrumentation for SVD attempts (g_svd_tries_total / done)
 *
 * Legacy:
 *   - older builders are kept under BF_VF_HIER_ENABLE_LEGACY_BUILD and are
 *     contained in this file (not exported via public headers).
 */

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

#include <stdint.h>
#include <errno.h>

#include <stdio.h>
#include <math.h>
#include <time.h>  /* for timing instrumentation */
#include <string.h>         /* for memset */
#include <bf/real_array.h>  /* for BfRealArray, bfRealArrayNewWithDefaultCapacity, etc */
#include <bf/ptr_array.h>

#include <bf/vf_hier_internal.h>

#if BF_VF_HIER_TIME_SVD
double g_vf_csr_to_dense_time = 0.0;
double g_vf_svd_time          = 0.0;
double g_vf_csr_slice_time    = 0.0;
double g_vf_leaf_map_time     = 0.0;
#endif


/* ============================================================
 * Optional SVD progress reporting (CSR-midlevel only)
 * ============================================================ */

unsigned long long g_svd_tries_total = 0ull;  /* computed by dry-run */
unsigned long long g_svd_tries_done  = 0ull;  /* incremented before bfGetTruncatedSvd */

/* “Final leaf” progress: counts how many leaf blocks have been materialized.
 * This tracks overall build completion better than SVD tries alone, since
 * most leaves may remain sparse (no SVD attempt). Total is computed by an
 * optional dry-run (countLeavesFromCsrMidlevel). */
unsigned long long g_vf_leaf_total = 0ull;
unsigned long long g_vf_leaf_done  = 0ull;

static int g_svd_progress_enabled = -1;  /* -1 unknown, 0 off, 1 on */
static unsigned long long g_svd_progress_every = 1ull; /* print every N tries */
static int g_progress_printed_any = 0;   /* used to cleanly end '\r' progress lines */

void vfHierInitProgressFromEnv_(void) {
  if (g_svd_progress_enabled != -1) return;

  const char *e = getenv("BF_VF_HIER_PROGRESS");
  g_svd_progress_enabled = (e && atoi(e) != 0) ? 1 : 0;

  const char *k = getenv("BF_VF_HIER_PROGRESS_EVERY");
  if (k && atoll(k) > 0) g_svd_progress_every = (unsigned long long)atoll(k);
}

/* If we've been printing carriage-return progress, ensure we end on a clean line
 * before any other stderr logging. Safe to call multiple times. */
static void vfHierProgressFinalizeLine_(void) {
  vfHierInitProgressFromEnv_();
  if (!g_svd_progress_enabled) return;
  if (!g_progress_printed_any) return;
  fprintf(stderr, "\n");
  fflush(stderr);
  g_progress_printed_any = 0;
}

void vfHierMaybePrintProgress_(unsigned long long done, unsigned long long total) {
  vfHierInitProgressFromEnv_();
  if (!g_svd_progress_enabled) return;
  if (total == 0ull) return;

  if (g_svd_progress_every == 0ull) g_svd_progress_every = 1ull;
  if (done % g_svd_progress_every != 0ull && done != total) return;

  double pct = 100.0 * (double)done / (double)total;
  fprintf(stderr, "\r[vf_hier] SVD tries: %llu / %llu (%.1f%%)",
          (unsigned long long)done,
          (unsigned long long)total,
          pct);
  g_progress_printed_any = 1;
  if (done == total) fprintf(stderr, "\n");
  fflush(stderr);
}


///* Local helpers to avoid BF's max/sqrt macros */
//static BfReal bf_local_max(BfReal a, BfReal b) {
//  return a > b ? a : b;
//}
//
//static BfReal bf_local_sqrt(BfReal x) {
//  if (x <= 0) return 0;
//  return (BfReal)sqrt((double)x);
//}

//typedef struct {
//  BfQuadtreeNode *row;
//  BfQuadtreeNode *col;
//} ChildPair;

//typedef struct {
//  BfSize *globalToRow;  /* size = mapSize, maps global face -> parent row index */
//  BfSize *globalToCol;  /* size = mapSize, maps global face -> parent col index */
//  BfSize  mapSize;      /* == maxFace + 1 in parent block */
//} BfVfFaceMap;


/* Return true iff every leaf in this subtree is a sparse (CSR) leaf.
 * Any SVD leaf, or unexpected kind, disables flattening.
 */
#if BF_VF_HIER_ENABLE_LEGACY_BUILD
static BfBool
bfVfHierBlockAllSparseLeaves(BfVfHierBlock const *block)
{
  if (block == NULL)
    return BF_TRUE; /* you can also choose FALSE; we only call on non-NULL roots */

  switch (block->kind) {
  case BF_VF_HIER_BLOCK_SPARSE:
    return BF_TRUE;

  case BF_VF_HIER_BLOCK_SVD:
    return BF_FALSE;

  case BF_VF_HIER_BLOCK_NODE: {
    BfPtrArray const *children = &block->data.node.children;
    for (BfSize i = 0; i < bfPtrArraySize(children); ++i) {
      BfVfHierBlock *child = bfPtrArrayGet(children, i);
      if (!bfVfHierBlockAllSparseLeaves(child))
        return BF_FALSE;
    }
    return BF_TRUE;
  }

  case BF_VF_HIER_BLOCK_NONE:
  default:
    return BF_FALSE;
  }
}
#endif

void vfHierMaybePrintLeafProgress_(unsigned long long done,
                                  unsigned long long total)
{
  vfHierInitProgressFromEnv_();
  if (!g_svd_progress_enabled) return;

  if (g_svd_progress_every == 0ull) g_svd_progress_every = 1ull;
  /* If we don't know the total (no dry-run), still print a single global
   * monotonically increasing counter. */
  if (total == 0ull) {
    if (done % g_svd_progress_every != 0ull) return;
    fprintf(stderr, "\r[vf_hier] leaves: %llu",
            (unsigned long long)done);
    g_progress_printed_any = 1;
    fflush(stderr);
    return;
  }

  if (done % g_svd_progress_every != 0ull && done != total) return;

  double pct = 100.0 * (double)done / (double)total;
  fprintf(stderr, "\r[vf_hier] leaves: %llu / %llu (%.1f%%)",
          (unsigned long long)done,
          (unsigned long long)total,
          pct);
  g_progress_printed_any = 1;
  if (done == total) fprintf(stderr, "\n");
  fflush(stderr);
}

void bfVfFaceMapInitFromParentFaces(
    BfVfFaceMap      *map,
    BfSizeArray const *rowFaces_par,
    BfSizeArray const *colFaces_par)
{
  BF_ASSERT(map != NULL);
  memset(map, 0, sizeof(*map));

  BfSize mPar = bfSizeArrayGetSize((BfSizeArray *)rowFaces_par);
  BfSize nPar = bfSizeArrayGetSize((BfSizeArray *)colFaces_par);

  /* Find max global face id used by this midlevel block */
  BfSize maxFace = 0;
  for (BfSize i = 0; i < mPar; ++i) {
    BfSize g = bfSizeArrayGet((BfSizeArray *)rowFaces_par, i);
    if (g > maxFace) maxFace = g;
  }
  for (BfSize j = 0; j < nPar; ++j) {
    BfSize g = bfSizeArrayGet((BfSizeArray *)colFaces_par, j);
    if (g > maxFace) maxFace = g;
  }

  map->mapSize = maxFace + 1;

  map->globalToRow = bfMemAlloc(map->mapSize, sizeof(BfSize));
  map->globalToCol = bfMemAlloc(map->mapSize, sizeof(BfSize));

  if (map->globalToRow == NULL || map->globalToCol == NULL) {
    if (map->globalToRow) bfMemFree(map->globalToRow);
    if (map->globalToCol) bfMemFree(map->globalToCol);
    map->globalToRow = map->globalToCol = NULL;
    map->mapSize = 0;
    return;
  }

  /* Initialize to BAD and fill from parent face lists */
  for (BfSize t = 0; t < map->mapSize; ++t) {
    map->globalToRow[t] = BF_SIZE_BAD_VALUE;
    map->globalToCol[t] = BF_SIZE_BAD_VALUE;
  }

  for (BfSize i = 0; i < mPar; ++i) {
    BfSize g = bfSizeArrayGet((BfSizeArray *)rowFaces_par, i);
    BF_ASSERT(g < map->mapSize);
    map->globalToRow[g] = i;
  }

  for (BfSize j = 0; j < nPar; ++j) {
    BfSize g = bfSizeArrayGet((BfSizeArray *)colFaces_par, j);
    BF_ASSERT(g < map->mapSize);
    map->globalToCol[g] = j;
  }
}

void bfVfFaceMapDeinit(BfVfFaceMap *map)
{
  if (map == NULL) return;
  if (map->globalToRow) bfMemFree(map->globalToRow);
  if (map->globalToCol) bfMemFree(map->globalToCol);
  map->globalToRow = map->globalToCol = NULL;
  map->mapSize = 0;
}

//typedef struct {
//  BfSize i0, i1;
//  BfSize j0, j1;
//  BfSize mi, mj;
//  bool leafI, leafJ;
//  bool small;
//  bool empty;
//} BfVfBlockMeta;















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


double bfVfHierBlockMemBytes(BfVfHierBlock const *block) {
  if (block == NULL)
    return 0.0;

  BfVfHierStats stats;
  bfVfHierStatsInit(&stats);
  bfVfHierBlockCollectStats(block, &stats);

  /* Same model you use for stats: sparse bytes + SVD bytes */
  return (double)stats.memBytesSparseEst + (double)stats.memBytesSvdEst;
}


void bfVfHierCollectStats(BfVfHier const *vfHier,
                          BfVfHierStats *stats) {
  BF_ASSERT(vfHier != NULL);
  BF_ASSERT(stats != NULL);

  bfVfHierStatsInit(stats);

  if (vfHier->root != NULL)
    bfVfHierBlockCollectStats(vfHier->root, stats);
}

#if BF_VF_HIER_ENABLE_LEGACY_BUILD
static BfVfSparseLeaf *bfVfSparseLeafNew(void) {
  BfVfSparseLeaf *leaf = bfMemAlloc(1, sizeof(BfVfSparseLeaf));
  bfSizeArrayInitWithDefaultCapacity(&leaf->rowInds);
  bfSizeArrayInitWithDefaultCapacity(&leaf->colInds);
  leaf->mat = NULL;
  return leaf;
}
#endif

static void bfVfSparseLeafDeinit(BfVfSparseLeaf *leaf) {
  if (leaf == NULL) return;
  bfSizeArrayDeinit(&leaf->rowInds);
  bfSizeArrayDeinit(&leaf->colInds);
  if (leaf->mat != NULL)
    bfMatCsrRealDeinitAndDealloc(&leaf->mat);
}

/* --- HierBlock helpers ------------------------------------------------- */

BfVfHierBlock *bfVfHierBlockNew(void) {
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
    bfMatDelete(&leaf->mat);  /* or whatever your Mat destroy function is */  }
  if (leaf->work != NULL) {
    bfMemFree(leaf->work);
    leaf->work = NULL;
    leaf->workLen = 0;
  }
}

void bfVfHierBlockDeinit(BfVfHierBlock *block) {
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

void bfVfHierBlockDealloc(BfVfHierBlock **blockPtr) {
  if (blockPtr == NULL || *blockPtr == NULL) return;
  bfVfHierBlockDeinit(*blockPtr);
  bfMemFree(*blockPtr);
  *blockPtr = NULL;
}

void bfVfHierBlockDeinitAndDealloc(BfVfHierBlock **blockPtr) {
  bfVfHierBlockDealloc(blockPtr);
}





/* --- Geometry / admissibility ----------------------------------------- */

///* Simple admissibility test in the stereographic plane, similar to your Cython _is_far_bbox */
BfBool isFar(BfQuadtreeNode const *rowNode,
                    BfQuadtreeNode const *colNode,
                    BfReal                eta) {

  if (eta < 0) return BF_TRUE;    /* flux-like: ignore geometry for SVD */

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
void getNodeInds(BfQuadtreeNode *node,
                        BfQuadtree          const *qt,
                        BfSizeArray         *inds) {
//  bfSizeArrayClear(inds);

  BfSize i0 = bfTreeNodeGetFirstIndex(bfQuadtreeNodeToTreeNode(node));
  BfSize i1 = bfTreeNodeGetLastIndex(bfQuadtreeNodeToTreeNode(node));

  /* Perm lives in the underlying tree */
  BfPerm const *perm = bfTreeGetPerm(bfQuadtreeToTree((BfQuadtree *)qt));

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
BfVfBlockMeta getBlockMeta(BfQuadtreeNode *rowNode,
                                  BfQuadtreeNode *colNode,
                                  BfSize          leafMax)
{
  BfVfBlockMeta meta;

  BfTreeNode *ni = bfQuadtreeNodeToTreeNode(rowNode);
  BfTreeNode *nj = bfQuadtreeNodeToTreeNode(colNode);

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

#if BF_VF_HIER_ENABLE_LEGACY_BUILD
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
  getNodeInds((BfQuadtreeNode *)rowNode, qt, &leaf->rowInds);
  getNodeInds((BfQuadtreeNode *)colNode, qt, &leaf->colInds);

  /* Build sub-block via existing BF builder */
  leaf->mat = bfMatCsrRealNewViewFactorMatrixFromTrimesh(
      tm, &leaf->rowInds, &leaf->colInds);

  /* Enforce invariant */
  BfSize nFaces = bfTrimeshGetNumFaces(tm);
  reindexCsrColsToLocal(leaf->mat, &leaf->colInds, nFaces);
  leaf->colsAreLocal = BF_TRUE;

  return block;
}
#endif

/* LEGACY: geometric builder using direct trimesh leaves.
 * Public API paths now use buildBlockHybrid + makeLeafFromCsrMidlevel.
 * Kept for debugging / comparison.
 */
#if BF_VF_HIER_ENABLE_LEGACY_BUILD
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

  BfTreeNode *ni = bfQuadtreeNodeToTreeNode(rowNode);
  BfTreeNode *nj = bfQuadtreeNodeToTreeNode(colNode);

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
#endif

/* --- BfVfHier API ------------------------------------------------------ */

BfVfHier *bfVfHierNew(void) {
  BfVfHier *vfHier = bfMemAlloc(1, sizeof(BfVfHier));
  vfHier->trimesh = NULL;
  vfHier->root    = NULL;
  vfHier->n       = 0;
  vfHier->applyPlan = NULL;
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
                              BfReal           minArea,
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

  vfHier->root = buildBlockHybrid(trimesh, rootQt, rootQt,
                                  eta, leafMax, leafMin, minArea,
                                  tol, minSvdSize, maxSvdRankFrac);

  /* If build printed '\r' progress, end on a clean line before any other logs. */
  vfHierProgressFinalizeLine_();

  #if BF_VF_HIER_TIME_SVD
fprintf(stderr,
        "[vf_hier] timing (sum over leaves): CSR->dense=%.6f s, SVD=%.6f s\n",
        g_vf_csr_to_dense_time, g_vf_svd_time);
fprintf(stderr,
        "[vf_hier] timing (sum over leaves): CSR slice=%.6f s, leaf map=%.6f s\n",
        g_vf_csr_slice_time, g_vf_leaf_map_time);


    /* reset accumulators so multiple builds don't add up across runs */
    g_vf_csr_to_dense_time = 0.0;
    g_vf_svd_time          = 0.0;
    g_vf_csr_slice_time    = 0.0;
    g_vf_leaf_map_time     = 0.0;
  #endif

#if BF_VF_HIER_DEBUG_FLATTEN
  fprintf(stderr,
          "[vf_hier] flatten stats: attempts=%llu success=%llu "
          "bytes_children=%.3e bytes_flat=%.3e\n",
          g_vf_flatten_attempts,
          g_vf_flatten_success,
          (double)g_vf_flatten_bytes_before,
          (double)g_vf_flatten_bytes_after);

  g_vf_flatten_attempts      = 0ull;
  g_vf_flatten_success      = 0ull;
  g_vf_flatten_bytes_before = 0ull;
  g_vf_flatten_bytes_after  = 0ull;
#endif

#if BF_VF_HIER_DEBUG_SVD_FILTER
  fprintf(stderr,
          "[vf_hier] svd stats: near_or_disabled=%llu too_small=%llu "
          "fail=%llu rank_reject=%llu "
          "mem_pre_reject=%llu mem_reject=%llu accept=%llu\n",
          g_svd_near_or_disabled,
          g_svd_too_small,
          g_svd_fail,
          g_svd_rank_reject,
          g_svd_mem_pre_reject,
          g_svd_mem_reject,
          g_svd_accept);

  g_svd_near_or_disabled = 0ull;
  g_svd_too_small        = 0ull;
  g_svd_fail             = 0ull;
  g_svd_rank_reject      = 0ull;
  g_svd_mem_pre_reject   = 0ull;
  g_svd_mem_reject       = 0ull;
  g_svd_accept           = 0ull;
#endif

#ifndef BF_VF_HIER_APPLY_TILE_SIZE
#define BF_VF_HIER_APPLY_TILE_SIZE 4096
#endif

bfVfHierBuildApplyPlan(vfHier, BF_VF_HIER_APPLY_TILE_SIZE);

}

BfVfHier *bfVfHierNewFromQuadtree(BfTrimesh const *trimesh,
                        BfQuadtree      *quadtree,
                        BfReal           eta,
                        BfSize           leafMax,
                        BfSize           leafMin,
                        BfReal           minArea,
                        BfReal           tol,
                        BfSize           minSvdSize,
                        BfReal           maxSvdRankFrac)
{
  BfVfHier *vfHier = bfVfHierNew();
  bfVfHierInitFromQuadtree(vfHier, trimesh, quadtree,
                           eta, leafMax, leafMin, minArea, tol, minSvdSize, maxSvdRankFrac);
  return vfHier;
}

/* ============================================================
 * CSR-based hierarchical view-factor hierarchy
 * ============================================================ */

#if BF_VF_HIER_ENABLE_LEGACY_BUILD
/* Forward declaration for local CSR submatrix builder */
static BfMatCsrReal *
bfMatCsrRealNewSubmatrixFromFaces(BfMatCsrReal const *A_par,
                                  BfSizeArray  const *rowFaces,
                                  BfSizeArray  const *colFaces);

/* LEGACY: CSR leaf builder (pre-midlevel).
 * Not used by current public APIs; unified leaf policy lives in
 * makeLeafFromCsrMidlevel.
 */
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

  BfBackend backend = BF_BACKEND_SVDS;
  BfBool truncated;

  truncated =
    bfGetTruncatedSvd(bfMatCsrRealToMat(Acsr), &U, &S, &VT,
                        &truncSpec, backend);

  if (!truncated) {
    /* Not an error by itself, but often indicates “k==maxRank”.
       With the new gates in linalg.c, this is usually safe. */
    BF_VF_HIER_LOG("SVD leaf: not truncated (k hit maxRank or tol loose)\n");
  }

  #if BF_VF_HIER_TIME_SVD
    double t_svd_end = bfVfHierNowSecs();
    #pragma omp atomic
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
  block->data.svd.U_cache  = NULL;
  block->data.svd.S_cache  = NULL;
  block->data.svd.VT_cache = NULL;

  bfMatCsrRealDeinitAndDealloc(&Acsr);

  return block;
}
#endif

#if BF_VF_HIER_ENABLE_LEGACY_BUILD
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

  BfTreeNode *ni = bfQuadtreeNodeToTreeNode(rowNode);
  BfTreeNode *nj = bfQuadtreeNodeToTreeNode(colNode);

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
#endif

/* Top-level initializer from CSR + quadtree */
void bfVfHierInitFromCsrAndQuadtree(BfVfHier     *vfHier,
                                    BfMatCsrReal *Afull,
                                    BfQuadtree   *quadtree,
                                    BfReal        eta,
                                    BfSize        leafMax,
                                    BfSize        leafMin,
                                    BfReal        minArea,
                                    BfReal        tol,
                                    BfSize        minSvdSize,
                                    BfReal        maxSvdRankFrac)
{
  BF_ASSERT(vfHier && Afull && quadtree);

  /* Reset progress counters for this build invocation (avoid >100% across runs). */
  g_svd_tries_total = 0ull;
  g_svd_tries_done  = 0ull;
  g_vf_leaf_total   = 0ull;
  g_vf_leaf_done    = 0ull;

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

  /* Parent face lists: here rows/cols of Afull already index global faces
   * 0..nRows-1, so we just build identity mappings.
   */
  BfSizeArray rowFaces_par;
  BfSizeArray colFaces_par;
  bfSizeArrayInitWithDefaultCapacity(&rowFaces_par);
  bfSizeArrayInitWithDefaultCapacity(&colFaces_par);

  for (BfSize i = 0; i < nRows; ++i)
    bfSizeArrayAppend(&rowFaces_par, i);
  for (BfSize j = 0; j < nCols; ++j)
    bfSizeArrayAppend(&colFaces_par, j);

  BfVfFaceMap faceMap;
  bfVfFaceMapInitFromParentFaces(&faceMap,
                                 (BfSizeArray const *)&rowFaces_par,
                                 (BfSizeArray const *)&colFaces_par);

#if 1
  {
    BfVfBlockMeta rootMeta = getBlockMeta(rootQt, rootQt, leafMax);
    fprintf(stderr,
            "[vf_hier] CSR init (midlevel): nRows=%lu, root i=[%lu,%lu), j=[%lu,%lu)\n",
            (unsigned long)nRows,
            (unsigned long)rootMeta.i0, (unsigned long)rootMeta.i1,
            (unsigned long)rootMeta.j0, (unsigned long)rootMeta.j1);
  }
#endif

  /* Optional dry-run counting pass */
  vfHierInitProgressFromEnv_();
  const char *doCount = getenv("BF_VF_HIER_COUNT_SVD_TRIES");
  if (doCount && atoi(doCount) != 0) {
    unsigned long long total = 0ull;
    countSvdTriesFromCsrMidlevel(
        (BfMatCsrReal const *)Afull,
        (BfVfFaceMap  const *)&faceMap,
        rootQt, rootQt,
        eta, leafMax, leafMin, (BfSize)minArea,
        tol, minSvdSize, maxSvdRankFrac,
        0, &total);

    g_svd_tries_total = total;
    g_svd_tries_done  = 0ull;
    /* Gate dry-run printing separately so you can suppress it while still
     * keeping runtime leaf progress. Default: OFF. */
    const char *p = getenv("BF_VF_HIER_PROGRESS_DRYRUN");
    if (p && atoi(p) != 0) {
      fprintf(stderr, "[vf_hier] dry-run: total SVD tries (by gates) = %llu\n",
              (unsigned long long)g_svd_tries_total);
    }
  }


  /* Optional dry-run: count how many “final leaves” will be materialized.
   * This gives a single monotonic progress counter for the overall build. */
  const char *doLeafCount = getenv("BF_VF_HIER_COUNT_LEAVES");
  if (doLeafCount && atoi(doLeafCount) != 0) {
    unsigned long long totalLeaves = 0ull;
    countLeavesFromCsrMidlevel(
        (BfMatCsrReal const *)Afull,
        (BfVfFaceMap  const *)&faceMap,
        rootQt, rootQt,
        eta, leafMax, leafMin, (BfSize)minArea,
        tol, minSvdSize, maxSvdRankFrac,
        0, &totalLeaves);
    g_vf_leaf_total = totalLeaves;
    g_vf_leaf_done  = 0ull;
    {
      const char *p = getenv("BF_VF_HIER_PROGRESS_DRYRUN");
      if (p && atoi(p) != 0) {
        fprintf(stderr, "[vf_hier] dry-run: total leaves (final blocks) = %llu\n",
                (unsigned long long)g_vf_leaf_total);
      }
    }
  }

  /* Use the same mid-level CSR + unified leaf policy as the hybrid path. */
  vfHier->root = buildBlockFromCsrMidlevel(
      (BfMatCsrReal const *)Afull,
      (BfSizeArray  const *)&rowFaces_par,
      (BfSizeArray  const *)&colFaces_par,
      (BfVfFaceMap  const *)&faceMap,   /* NEW */
      rootQt,
      rootQt,
      eta,
      leafMax,
      leafMin,
      minArea,
      tol,
      minSvdSize,
      maxSvdRankFrac,
      0);  /* depth = 0 */

  /* If build printed '\r' progress, end on a clean line before any other logs. */
  vfHierProgressFinalizeLine_();

  /* In case no total was known (no dry-run), ensure we still finish cleanly
   * even if the last leaf printed wasn't "done==total". */
  vfHierProgressFinalizeLine_();

  if (vfHier->root == NULL) {
    fprintf(stderr,
            "[vf_hier] bfVfHierInitFromCsrAndQuadtree: root block is NULL "
            "(maybe all leaves too small or CSR submatrix failed)\n");
  }

  bfVfFaceMapDeinit(&faceMap);          /* NEW */
  bfSizeArrayDeinit(&rowFaces_par);
  bfSizeArrayDeinit(&colFaces_par);

#if BF_VF_HIER_TIME_SVD
fprintf(stderr,
        "[vf_hier] timing (sum over leaves): CSR->dense=%.6f s, SVD=%.6f s\n",
        g_vf_csr_to_dense_time, g_vf_svd_time);
fprintf(stderr,
        "[vf_hier] timing (sum over leaves): CSR slice=%.6f s, leaf map=%.6f s\n",
        g_vf_csr_slice_time, g_vf_leaf_map_time);

  g_vf_csr_to_dense_time = 0.0;
  g_vf_svd_time          = 0.0;
  g_vf_csr_slice_time    = 0.0;
  g_vf_leaf_map_time     = 0.0;
#endif

#ifndef BF_VF_HIER_APPLY_TILE_SIZE
#define BF_VF_HIER_APPLY_TILE_SIZE 4096
#endif

bfVfHierBuildApplyPlan(vfHier, BF_VF_HIER_APPLY_TILE_SIZE);

}

/* Convenience constructor */
BfVfHier *bfVfHierNewFromCsrAndQuadtree(BfMatCsrReal *Afull,
                                         BfQuadtree   *quadtree,
                                         BfReal        eta,
                                         BfSize        leafMax,
                                         BfSize        leafMin,
                                         BfReal        minArea,
                                         BfReal        tol,
                                         BfSize        minSvdSize,
                                         BfReal        maxSvdRankFrac)
{
  BfVfHier *vf = bfVfHierNew();
  bfVfHierInitFromCsrAndQuadtree(vf, Afull, quadtree,
                                 eta, leafMax, leafMin, minArea,
                                 tol, minSvdSize, maxSvdRankFrac);
  return vf;
}



