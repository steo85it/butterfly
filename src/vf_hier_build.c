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
 */

#include <bf/def.h>
#include <bf/vf_hier.h>

#include <bf/mem.h>
#include <bf/error.h>
#include <bf/tree.h>
#include <bf/tree_node.h>
#include <bf/quadtree_node.h>
#include <bf/octree_node.h>
#include <bf/bbox.h>
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

///* Called exactly when recursion chooses a FINAL leaf representation (or a zero block).
// * Thread-safe. */
//void vfHierOnLeafMaterialized_(void) {
//  vfHierInitProgressFromEnv_();
//  if (!g_svd_progress_enabled) return;
//
//  unsigned long long done;
//  #pragma omp atomic capture
//  done = ++g_vf_leaf_done;
//
//  vfHierMaybePrintLeafProgress_(done, g_vf_leaf_total);
//}


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

/* Return true iff every leaf in this subtree is a sparse (CSR) leaf.
 * Any SVD leaf, or unexpected kind, disables flattening.
 */
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

  return bfVfIsFarTreeNodes_(
    bfQuadtreeNodeConstToTreeNodeConst(rowNode),
    bfQuadtreeNodeConstToTreeNodeConst(colNode),
    eta);
}

/* Extract global face indices for a node into a BfSizeArray */
void getNodeInds(BfQuadtreeNode *node,
                        BfQuadtree          const *qt,
                        BfSizeArray         *inds) {

  bfVfGetNodeIndsFromTreeNode_(
    bfQuadtreeNodeToTreeNode(node),
    bfQuadtreeToTree((BfQuadtree *)qt),
    inds);
}

/* ============================================================
 * Generic TreeNode-based geometry helpers
 * ============================================================ */

static inline BfReal bfVfMax2_(BfReal a, BfReal b) { return a > b ? a : b; }
static inline BfReal bfVfMax3_(BfReal a, BfReal b, BfReal c) { return bfVfMax2_(a, bfVfMax2_(b, c)); }

/* Helper: get a “diameter” compatible with your existing quadtree gating:
 * for 2D we used max(side lengths); for 3D use max(dx,dy,dz). */
static inline BfReal bfVfBbox2Diam_(BfBbox2 const *b) {
  return bfVfMax2_(b->max[0] - b->min[0], b->max[1] - b->min[1]);
}
static inline BfReal bfVfBbox3Diam_(BfBoundingBox3 const *b) {
  return bfVfMax3_(b->max[0] - b->min[0], b->max[1] - b->min[1], b->max[2] - b->min[2]);
}

BfBool bfVfIsFarTreeNodes_(BfTreeNode const *rowNode,
                           BfTreeNode const *colNode,
                           BfReal eta)
{
  if (eta < 0) return BF_TRUE; /* preserve existing behavior */

  BF_ASSERT(rowNode && colNode);

  /* Dispatch by concrete node type (Tree abstraction + downcasts). */
  BfType tr = bfTreeNodeGetType(rowNode);
  BfType tc = bfTreeNodeGetType(colNode);
  BF_ASSERT(tr == tc && "row/col trees must have same node type");

  /* --- Quadtree (2D bbox) --- */
  if (bfTreeNodeInstanceOf(rowNode, BF_TYPE_QUADTREE_NODE)) {
    BfQuadtreeNode const *qr = bfTreeNodeConstToQuadtreeNodeConst(rowNode);
    BfQuadtreeNode const *qc = bfTreeNodeConstToQuadtreeNodeConst(colNode);
    BfBbox2 br = bfQuadtreeNodeGetBbox(qr);
    BfBbox2 bc = bfQuadtreeNodeGetBbox(qc);

    BfPoint2 cr, cc;
    bfBbox2GetCenter(&br, cr);
    bfBbox2GetCenter(&bc, cc);

    BfReal diam = bfVfMax2_(bfVfBbox2Diam_(&br), bfVfBbox2Diam_(&bc));
    BfReal dx = cr[0] - cc[0];
    BfReal dy = cr[1] - cc[1];
    BfReal dist = bf_local_sqrt(dx*dx + dy*dy);
    if (dist == 0) return BF_FALSE;
    return diam <= eta*dist;
  }

  /* --- Octree (3D bbox) --- */
  if (bfTreeNodeInstanceOf(rowNode, BF_TYPE_OCTREE_NODE)) {
    BfOctreeNode const *orow = bfTreeNodeConstToOctreeNodeConst(rowNode);
    BfOctreeNode const *ocol = bfTreeNodeConstToOctreeNodeConst(colNode);
    BfBoundingBox3 br = bfOctreeNodeGetBoundingBox(orow);
    BfBoundingBox3 bc = bfOctreeNodeGetBoundingBox(ocol);

    BfPoint3 cr, cc;
    bfBoundingBox3GetCenter(&br, cr);
    bfBoundingBox3GetCenter(&bc, cc);

    BfReal diam = bfVfMax2_(bfVfBbox3Diam_(&br), bfVfBbox3Diam_(&bc));
    BfReal dx = cr[0] - cc[0];
    BfReal dy = cr[1] - cc[1];
    BfReal dz = cr[2] - cc[2];
    BfReal dist = bf_local_sqrt(dx*dx + dy*dy + dz*dz);
    if (dist == 0) return BF_FALSE;
    return diam <= eta*dist;
  }

  BF_ASSERT(0 && "unsupported TreeNode type for vf_hier admissibility");
  return BF_FALSE;
}

void bfVfGetNodeIndsFromTreeNode_(BfTreeNode *node,
                                 BfTree const *tree,
                                 BfSizeArray *inds)
{
  BF_ASSERT(node && tree && inds);

  BfSize i0 = bfTreeNodeGetFirstIndex(node);
  BfSize i1 = bfTreeNodeGetLastIndex(node);

  BfPerm const *perm = bfTreeGetPermConst((BfTree *)tree);
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
BfVfBlockMeta getBlockMeta(BfTreeNode *rowNode,
                                  BfTreeNode *colNode,
                                  BfSize          leafMax)
{
  BfVfBlockMeta meta;

  BfTreeNode *ni = rowNode;
  BfTreeNode *nj = colNode;

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

/* Build an octree over FACE centroids (and FACE unit normals). */
static BfOctree *bfVfOctreeNewFromTrimeshFaces_(BfTrimesh const *tm, BfSize maxLeafSize,
                                               BfPoints3 **pointsOwned,
                                               BfVectors3 **normalsOwned) {
  BF_ASSERT(tm != NULL);
  BfSize n = bfTrimeshGetNumFaces(tm);

  /* Allocate points/normals we own (freed after build). */
  BfPoints3 *points = bfPoints3NewWithDefaultCapacity();
  BfVectors3 *normals = bfVectors3NewWithCapacity(n);

  for (BfSize i = 0; i < n; ++i) {
    BfPoint3 p;
    BfVector3 u;

    BfReal const *pc = bfTrimeshGetFaceCentroidConstPtr(tm, i);
    BfReal const *nc = bfTrimeshGetFaceUnitNormalConstPtr(tm, i);

    p[0] = pc[0]; p[1] = pc[1]; p[2] = pc[2];
    u[0] = nc[0]; u[1] = nc[1]; u[2] = nc[2];

    bfPoints3Append(points, p);
    bfVectors3Append(normals, u);
  }

  BfOctree *octree = bfOctreeNew();
  bfOctreeInit(octree, points, normals, maxLeafSize);

  *pointsOwned  = points;
  *normalsOwned = normals;
  return octree;
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

  /* Clear any previous contents */
  bfVfHierDeinit(vfHier);

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


void bfVfHierInitFromOctree(BfVfHier        *vfHier,
                            BfTrimesh const *trimesh,
                            BfOctree        *octree,
                            BfReal           eta,
                            BfSize           leafMax,
                            BfSize           leafMin,
                            BfReal           minArea,          /* ignored for octree */
                            BfReal           tol,
                            BfSize           minSvdSize,
                            BfReal           maxSvdRankFrac) {
  BF_ASSERT(vfHier != NULL);
  BF_ASSERT(trimesh != NULL);
  BF_ASSERT(octree != NULL);

  /* Clear any previous contents */
  bfVfHierDeinit(vfHier);

  /* Build using generic Tree interface */
  BfTree *tree = bfOctreeToTree(octree);
  BfTreeNode *root = bfTreeGetRootNode(tree);

  (void)minArea; /* octree ignores this but keep signature stable */

  vfHier->trimesh = trimesh;
  vfHier->n = bfTrimeshGetNumFaces(trimesh);

  vfHier->root = buildBlockHybridFromTreeNodes(
      trimesh, tree, root, root,
      eta, leafMax, leafMin, minArea,
      tol, minSvdSize, maxSvdRankFrac);

  #ifndef BF_VF_HIER_APPLY_TILE_SIZE
  #define BF_VF_HIER_APPLY_TILE_SIZE 4096
  #endif
  bfVfHierBuildApplyPlan(vfHier, BF_VF_HIER_APPLY_TILE_SIZE);

  /* Reset apply cache */
  vfHier->applyPlan = NULL;
  vfHier->applyTileSize = 0;
}

BfVfHier *bfVfHierNewFromOctree(BfTrimesh const *trimesh,
                                BfOctree        *octree,
                                BfReal           eta,
                                BfSize           leafMax,
                                BfSize           leafMin,
                                BfReal           minArea,
                                BfReal           tol,
                                BfSize           minSvdSize,
                                BfReal           maxSvdRankFrac) {
  BfVfHier *vfHier = bfVfHierNew();
  bfVfHierInitFromOctree(vfHier, trimesh, octree,
                         eta, leafMax, leafMin, minArea,
                         tol, minSvdSize, maxSvdRankFrac);
  return vfHier;
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

/* ============================================================
 * Shared CSR-midlevel initializer (tree-agnostic)
 * ============================================================ */
static void bfVfHierInitFromCsrAndTree_(
    BfVfHier     *vfHier,
    BfMatCsrReal *Afull,
    BfTree       *tree,
    BfTreeNode   *rootNode,
    BfReal        eta,
    BfSize        leafMax,
    BfSize        leafMin,
    BfReal        minArea,
    BfReal        tol,
    BfSize        minSvdSize,
    BfReal        maxSvdRankFrac,
    char const   *tag) /* e.g. "quadtree" or "octree" */
{
  BF_ASSERT(vfHier && Afull && tree && rootNode);

  /* Reset progress counters for this build invocation (avoid >100% across runs). */
  g_svd_tries_total = 0ull;
  g_svd_tries_done  = 0ull;
  g_vf_leaf_total   = 0ull;
  g_vf_leaf_done    = 0ull;

  BfMat *A_base = bfMatCsrRealToMat(Afull);
  BfSize nRows  = bfMatGetNumRows(A_base);
  BfSize nCols  = bfMatGetNumCols(A_base);

  if (nRows != nCols) {
    fprintf(stderr,
            "[vf_hier] bfVfHierInitFromCsrAnd%s: Afull not square (%lu x %lu)\n",
            tag ? tag : "Tree",
            (unsigned long)nRows, (unsigned long)nCols);
  }

  vfHier->trimesh = NULL;
  vfHier->n       = nRows;

  /* Parent face lists: identity */
  BfSizeArray rowFaces_par;
  BfSizeArray colFaces_par;
  bfSizeArrayInitWithDefaultCapacity(&rowFaces_par);
  bfSizeArrayInitWithDefaultCapacity(&colFaces_par);

  for (BfSize i = 0; i < nRows; ++i) bfSizeArrayAppend(&rowFaces_par, i);
  for (BfSize j = 0; j < nCols; ++j) bfSizeArrayAppend(&colFaces_par, j);

  /* Build faceMap once */
  BfVfFaceMap faceMap;
  bfVfFaceMapInitFromParentFaces(&faceMap,
                                 (BfSizeArray const *)&rowFaces_par,
                                 (BfSizeArray const *)&colFaces_par);

#if 1
  {
    BfVfBlockMeta rootMeta = getBlockMeta(rootNode, rootNode, leafMax);
    fprintf(stderr,
            "[vf_hier] CSR init (%s, midlevel): nRows=%lu, root i=[%lu,%lu), j=[%lu,%lu)\n",
            tag ? tag : "tree",
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
        rootNode, rootNode,
        eta, leafMax, leafMin, (BfSize)minArea,
        tol, minSvdSize, maxSvdRankFrac,
        0, &total);

    g_svd_tries_total = total;
    g_svd_tries_done  = 0ull;

    const char *p = getenv("BF_VF_HIER_PROGRESS_DRYRUN");
    if (p && atoi(p) != 0) {
      fprintf(stderr, "[vf_hier] dry-run: total SVD tries (by gates) = %llu\n",
              (unsigned long long)g_svd_tries_total);
    }
  }

  /* Optional dry-run: count total leaves */
  const char *doLeafCount = getenv("BF_VF_HIER_COUNT_LEAVES");
  if (doLeafCount && atoi(doLeafCount) != 0) {
    unsigned long long totalLeaves = 0ull;
    countLeavesFromCsrMidlevel(
        (BfMatCsrReal const *)Afull,
        (BfVfFaceMap  const *)&faceMap,
        rootNode, rootNode,
        eta, leafMax, leafMin, (BfSize)minArea,
        tol, minSvdSize, maxSvdRankFrac,
        0, &totalLeaves);

    g_vf_leaf_total = totalLeaves;
    g_vf_leaf_done  = 0ull;

    const char *p = getenv("BF_VF_HIER_PROGRESS_DRYRUN");
    if (p && atoi(p) != 0) {
      fprintf(stderr, "[vf_hier] dry-run: total leaves (final blocks) = %llu\n",
              (unsigned long long)g_vf_leaf_total);
    }
  }

  /* Actual build */
  vfHier->root = buildBlockFromCsrMidlevel(
      (BfMatCsrReal const *)Afull,
      (BfSizeArray  const *)&rowFaces_par,
      (BfSizeArray  const *)&colFaces_par,
      (BfVfFaceMap  const *)&faceMap,
      tree,
      rootNode,
      rootNode,
      eta,
      leafMax,
      leafMin,
      minArea,
      tol,
      minSvdSize,
      maxSvdRankFrac,
      0);

  vfHierProgressFinalizeLine_();

  if (vfHier->root == NULL) {
    fprintf(stderr,
            "[vf_hier] bfVfHierInitFromCsrAnd%s: root block is NULL\n",
            tag ? tag : "Tree");
  }

  bfVfFaceMapDeinit(&faceMap);
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

  BfTree *tree = bfQuadtreeToTree(quadtree);
  BfTreeNode *root = bfTreeGetRootNode(tree);

  bfVfHierInitFromCsrAndTree_(
      vfHier, Afull,
      tree, root,
      eta, leafMax, leafMin,
      minArea, tol, minSvdSize, maxSvdRankFrac,
      "Quadtree");
}


/* Top-level initializer from CSR + quadtree */
void bfVfHierInitFromCsrAndOctree(BfVfHier     *vfHier,
                                  BfMatCsrReal *Afull,
                                  BfOctree     *octree,
                                  BfReal        eta,
                                  BfSize        leafMax,
                                  BfSize        leafMin,
                                  BfReal        minArea,
                                  BfReal        tol,
                                  BfSize        minSvdSize,
                                  BfReal        maxSvd_rank_frac)
{
  BF_ASSERT(vfHier && Afull && octree);

  BfTree *tree = bfOctreeToTree(octree);
  BfTreeNode *root = bfTreeGetRootNode(tree);

  bfVfHierInitFromCsrAndTree_(
      vfHier, Afull,
      tree, root,
      eta, leafMax, leafMin,
      minArea, tol, minSvdSize, maxSvd_rank_frac,
      "Octree");
}


void bfVfHierInitFromCsrAndAutoTree(
    BfVfHier     *vfHier,
    BfMatCsrReal *Afull,
    BfQuadtree   *quadtree,
    BfOctree     *octree,
    BfReal        eta,
    BfSize        leafMax,
    BfSize        leafMin,
    BfReal        minArea,
    BfReal        tol,
    BfSize        minSvdSize,
    BfReal        maxSvdRankFrac)
{
  BF_ASSERT(vfHier != NULL);
  BF_ASSERT(Afull != NULL);

  if (quadtree != NULL) {
    bfVfHierInitFromCsrAndQuadtree(vfHier, Afull, quadtree,
                                  eta, leafMax, leafMin, minArea,
                                  tol, minSvdSize, maxSvdRankFrac);
    return;
  }

  if (octree != NULL) {
    bfVfHierInitFromCsrAndOctree(vfHier, Afull, octree,
                                eta, leafMax, leafMin, minArea,
                                tol, minSvdSize, maxSvdRankFrac);
    return;
  }

  BF_ASSERT(0 && "bfVfHierInitFromCsrAndAutoTree: both quadtree and octree are NULL");
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

BfVfHier *bfVfHierNewFromCsrAndOctree(BfMatCsrReal *Afull,
                                         BfOctree   *octree,
                                         BfReal        eta,
                                         BfSize        leafMax,
                                         BfSize        leafMin,
                                         BfReal        minArea,
                                         BfReal        tol,
                                         BfSize        minSvdSize,
                                         BfReal        maxSvdRankFrac)
{
  BfVfHier *vf = bfVfHierNew();
  bfVfHierInitFromCsrAndOctree(vf, Afull, octree,
                                 eta, leafMax, leafMin, minArea,
                                 tol, minSvdSize, maxSvdRankFrac);
  return vf;
}

void bfVfHierInitFromTrimeshAndAutoTree(BfVfHier        *vfHier,
                                        BfTrimesh const *trimesh,
                                        BfVfTopology     topo,
                                        BfQuadtree      *quadtree, /* nullable */
                                        BfOctree        *octree,   /* nullable */
                                        BfReal           eta,
                                        BfSize           leafMax,
                                        BfSize           leafMin,
                                        BfReal           minArea,
                                        BfReal           tol,
                                        BfSize           minSvdSize,
                                        BfReal           maxSvdRankFrac) {
  BF_ASSERT(vfHier != NULL);
  BF_ASSERT(trimesh != NULL);

  bfVfHierDeinit(vfHier);

  /* Resolve topology + tree */
  BfTree *tree = NULL;
  BfTreeNode *root = NULL;

  /* Track ownership for internally-created octree inputs */
  BfOctree *octreeOwned = NULL;
  BfPoints3 *pointsOwned = NULL;
  BfVectors3 *normalsOwned = NULL;

  /* Track ownership for internally-created quadtree if you support it */
  BfQuadtree *quadtreeOwned = NULL;

  BfVfTopology topoResolved = topo;

  if (quadtree != NULL) {
    topoResolved = BF_VF_TOPO_QUADTREE;
    tree = bfQuadtreeToTree(quadtree);
  }
  else if (octree != NULL) {
    topoResolved = BF_VF_TOPO_OCTREE;
    tree = bfOctreeToTree(octree);
  }
  else {
    /* Nothing provided: decide what to build internally */
    if (topoResolved == BF_VF_TOPO_AUTO) {
      topoResolved = BF_VF_TOPO_OCTREE; /* your desired default */
    }

    if (topoResolved == BF_VF_TOPO_OCTREE) {
      octreeOwned = bfVfOctreeNewFromTrimeshFaces_(trimesh, leafMax,
                                                   &pointsOwned, &normalsOwned);
      tree = bfOctreeToTree(octreeOwned);
    }
    else if (topoResolved == BF_VF_TOPO_QUADTREE) {
      /* TODO: build a quadtree internally here.
       *
       * You likely already have something like:
       *   quadtreeOwned = bfQuadtreeNewFromTrimesh(...);
       * or building from projected coords elsewhere.
       *
       * Once you have it:
       *   tree = bfQuadtreeToTree(quadtreeOwned);
       */
      bfSetError(BF_ERROR_INVALID_ARGUMENTS);
      printf("Internal quadtree build not wired yet (need quadtree-from-trimesh constructor)");
      exit(EXIT_FAILURE);
    }
    else {
      bfSetError(BF_ERROR_INVALID_ARGUMENTS);
      printf("Unknown vf topology");
      exit(EXIT_FAILURE);
    }
  }

  BF_ASSERT(tree != NULL);

  root = bfTreeGetRootNode(tree);
  BF_ASSERT(root != NULL);

  vfHier->trimesh = trimesh;
  vfHier->n = bfTrimeshGetNumFaces(trimesh);

  vfHier->root = buildBlockHybridFromTreeNodes(
      trimesh, tree, root, root,
      eta, leafMax, leafMin, minArea,
      tol, minSvdSize, maxSvdRankFrac);

  vfHier->applyPlan = NULL;
  vfHier->applyTileSize = 0;

  /* Cleanup internals created only for building */
  if (octreeOwned != NULL) {
    bfOctreeDelete(&octreeOwned);
  }
  if (pointsOwned != NULL) {
    bfPoints3DeinitAndDealloc(&pointsOwned);
  }
  if (normalsOwned != NULL) {
    bfVectors3DeinitAndDealloc(&normalsOwned);
  }
  if (quadtreeOwned != NULL) {
    bfQuadtreeDeinitAndDealloc(&quadtreeOwned);
  }
}

BfVfHier *bfVfHierNewFromTrimeshAndAutoTree(BfTrimesh const *trimesh,
                                            BfVfTopology     topo,
                                            BfQuadtree      *quadtree,
                                            BfOctree        *octree,
                                            BfReal           eta,
                                            BfSize           leafMax,
                                            BfSize           leafMin,
                                            BfReal           minArea,
                                            BfReal           tol,
                                            BfSize           minSvdSize,
                                            BfReal           maxSvdRankFrac) {
  BfVfHier *vfHier = bfVfHierNew();
  bfVfHierInitFromTrimeshAndAutoTree(vfHier, trimesh, topo, quadtree, octree,
                                     eta, leafMax, leafMin, minArea,
                                     tol, minSvdSize, maxSvdRankFrac);
  return vfHier;
}


/* Raytrace one CSR block for (rowNode, colNode), then build subtree using CSR slicing */
static BfVfHierBlock *buildSubtreeFromTrimeshUsingCsrTreeNodes_(
    BfTrimesh const *tm,
    BfTree const    *tree,
    BfTreeNode      *rowNode,
    BfTreeNode      *colNode,
    BfReal           eta,
    BfSize           leafMax,
    BfSize           leafMin,
    BfReal           minArea,
    BfReal           tol,
    BfSize           minSvdSize,
    BfReal           maxSvdRankFrac)
{
  /* Parent face lists in GLOBAL face indices for this node pair */
  BfSizeArray rowFaces_par, colFaces_par;
  bfSizeArrayInitWithDefaultCapacity(&rowFaces_par);
  bfSizeArrayInitWithDefaultCapacity(&colFaces_par);

  bfVfGetNodeIndsFromTreeNode_(rowNode, tree, &rowFaces_par);
  bfVfGetNodeIndsFromTreeNode_(colNode, tree, &colFaces_par);

  /* If either side is empty, no block */
  if (bfSizeArrayGetSize(&rowFaces_par) == 0 || bfSizeArrayGetSize(&colFaces_par) == 0) {
    bfSizeArrayDeinit(&rowFaces_par);
    bfSizeArrayDeinit(&colFaces_par);
    return NULL;
  }

  /* Build one CSR block by raytracing from trimesh */
  BfMatCsrReal *A_par = bfMatCsrRealNewViewFactorMatrixFromTrimesh(
      tm, (BfSizeArray const *)&rowFaces_par, (BfSizeArray const *)&colFaces_par);

  if (A_par == NULL) {
    bfSizeArrayDeinit(&rowFaces_par);
    bfSizeArrayDeinit(&colFaces_par);
    return NULL;
  }

  /* Build faceMap for global->parent row/col positions */
  BfVfFaceMap faceMap;
  bfVfFaceMapInitFromParentFaces(&faceMap,
                                 (BfSizeArray const *)&rowFaces_par,
                                 (BfSizeArray const *)&colFaces_par);

  /* Recurse using CSR slicing */
  BfVfHierBlock *subtree = buildBlockFromCsrMidlevel(
      (BfMatCsrReal const *)A_par,
      (BfSizeArray  const *)&rowFaces_par,
      (BfSizeArray  const *)&colFaces_par,
      (BfVfFaceMap  const *)&faceMap,
      tree,
      rowNode, colNode,
      eta, leafMax, leafMin,
      (BfSize)minArea,
      tol, minSvdSize, maxSvdRankFrac,
      0);

  bfVfFaceMapDeinit(&faceMap);
  bfMatCsrRealDeinitAndDealloc(&A_par);
  bfSizeArrayDeinit(&rowFaces_par);
  bfSizeArrayDeinit(&colFaces_par);

  return subtree;
}


/* Generic TreeNode hybrid builder:
 * - split geometrically until area is small enough
 * - then raytrace one CSR block and recurse via CSR slicing
 */
BfVfHierBlock *buildBlockHybridFromTreeNodes(
    BfTrimesh const *tm,
    BfTree const    *tree,
    BfTreeNode      *rowNode,
    BfTreeNode      *colNode,
    BfReal           eta,
    BfSize           leafMax,
    BfSize           leafMin,
    BfReal           minArea,
    BfReal           tol,
    BfSize           minSvdSize,
    BfReal           maxSvdRankFrac)
{
  BfVfBlockMeta meta = getBlockMeta(rowNode, colNode, leafMax);
  if (meta.empty) return NULL;

  unsigned long long area =
    (unsigned long long)meta.mi * (unsigned long long)meta.mj;

  /* Optional early-stop by area: just treat as “small enough for CSR-midlevel subtree”.
   * (This is consistent with your CSR-midlevel policy deciding CSR vs SVD.) */
  if (area <= (unsigned long long)minArea) {
    BfVfHierBlock *sub =
      buildSubtreeFromTrimeshUsingCsrTreeNodes_(
        tm, tree, rowNode, colNode,
        eta, leafMax, leafMin, minArea,
        tol, minSvdSize, maxSvdRankFrac);
    if (sub != NULL) return sub;
    /* fall through to normal logic if something failed */
  }

  if (area <= BF_VF_HIER_MIDLEVEL_MAX_NNZ) {
    return buildSubtreeFromTrimeshUsingCsrTreeNodes_(
      tm, tree, rowNode, colNode,
      eta, leafMax, leafMin, minArea,
      tol, minSvdSize, maxSvdRankFrac);
  }

  /* Too large: split geometrically */
  BfVfHierBlock *nodeBlock = bfVfHierBlockNew();
  nodeBlock->kind = BF_VF_HIER_BLOCK_NODE;
  bfInitPtrArray(&nodeBlock->data.node.children, 4);

  bool leafI = bfTreeNodeIsLeaf(rowNode);
  bool leafJ = bfTreeNodeIsLeaf(colNode);

  if (!leafI && !leafJ) {
    BfSize maxI = bfTreeNodeGetMaxNumChildren(rowNode);
    BfSize maxJ = bfTreeNodeGetMaxNumChildren(colNode);

    for (BfSize a = 0; a < maxI; ++a) {
      if (!bfTreeNodeHasChild(rowNode, a)) continue;
      BfTreeNode *ci = bfTreeNodeGetChild(rowNode, a);

      for (BfSize b = 0; b < maxJ; ++b) {
        if (!bfTreeNodeHasChild(colNode, b)) continue;
        BfTreeNode *cj = bfTreeNodeGetChild(colNode, b);

        BfVfHierBlock *child =
          buildBlockHybridFromTreeNodes(
            tm, tree, ci, cj,
            eta, leafMax, leafMin, minArea,
            tol, minSvdSize, maxSvdRankFrac);
        if (child != NULL)
          bfPtrArrayAppend(&nodeBlock->data.node.children, child);
      }
    }
  }
  else if (!leafI) {
    BfSize maxI = bfTreeNodeGetMaxNumChildren(rowNode);
    for (BfSize a = 0; a < maxI; ++a) {
      if (!bfTreeNodeHasChild(rowNode, a)) continue;
      BfTreeNode *ci = bfTreeNodeGetChild(rowNode, a);

      BfVfHierBlock *child =
        buildBlockHybridFromTreeNodes(
          tm, tree, ci, colNode,
          eta, leafMax, leafMin, minArea,
          tol, minSvdSize, maxSvdRankFrac);
      if (child != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, child);
    }
  }
  else { /* !leafJ */
    BfSize maxJ = bfTreeNodeGetMaxNumChildren(colNode);
    for (BfSize b = 0; b < maxJ; ++b) {
      if (!bfTreeNodeHasChild(colNode, b)) continue;
      BfTreeNode *cj = bfTreeNodeGetChild(colNode, b);

      BfVfHierBlock *child =
        buildBlockHybridFromTreeNodes(
          tm, tree, rowNode, cj,
          eta, leafMax, leafMin, minArea,
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

