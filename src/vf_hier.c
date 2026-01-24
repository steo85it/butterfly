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

#ifndef BF_UNUSED
#  define BF_UNUSED(x) (void)(x)
#endif

#ifndef BF_TRUE
#  define BF_TRUE true
#endif

#ifndef BF_FALSE
#  define BF_FALSE false
#endif

#ifndef BF_VF_HIER_ENABLE_LEGACY_BUILD
#  define BF_VF_HIER_ENABLE_LEGACY_BUILD 0
#endif

/* Enable/disable SVD/timing (set to 0 to compile out).
 * If defined on the compiler command line, don't override it here.
 */
#ifndef BF_VF_HIER_TIME_SVD
#  define BF_VF_HIER_TIME_SVD 1
#endif

#if BF_VF_HIER_TIME_SVD
#  ifdef _OPENMP
#    include <omp.h>
#  endif
#  include <sys/time.h>  /* gettimeofday for wall-clock timing */
#endif

#if BF_VF_HIER_TIME_SVD
/* Return wall-clock time in seconds.
 *
 * - With OpenMP: use omp_get_wtime().
 * - Without OpenMP: fall back to gettimeofday().
 *
 * NOTE: we still accumulate per-leaf timings over all threads, so the
 * totals are "sum of leaf wall-times", which can exceed the overall
 * wall-clock build time when parallelism is used.
 */
static double bfVfHierNowSecs(void) {
#  ifdef _OPENMP
  return omp_get_wtime();
#  else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1e-6*(double)tv.tv_usec;
#  endif
}

/* Accumulate total CSR->dense and SVD times over one hierarchy build
 * (sum of per-leaf wall-times, not global wall time).
 */
static double g_vf_csr_to_dense_time = 0.0;
static double g_vf_svd_time          = 0.0;
static double g_vf_csr_slice_time    = 0.0;  /* bfMatCsrRealNewSubmatrixFromIndices */
static double g_vf_leaf_map_time     = 0.0;  /* build childRowFaces + global→parent maps */
#endif

/* ------------------------------------------------------------
 * Optional lightweight logging for vf_hier.
 * If BF_VF_HIER_LOG is not defined as a macro, the compiler will
 * assume it's an external function -> link error.
 * ------------------------------------------------------------ */
#ifndef BF_VF_HIER_ENABLE_LOG
#define BF_VF_HIER_ENABLE_LOG 0
#endif

#if BF_VF_HIER_ENABLE_LOG
  #define BF_VF_HIER_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
  #define BF_VF_HIER_LOG(...) ((void)0)
#endif

/* Approximate max nnz (= mi*mj) for a block we are willing to
 * raytrace in one shot and then reuse via CSR slicing.
 *
 * You can override at compile time with -DBF_VF_HIER_MIDLEVEL_MAX_NNZ=...
 */
#ifndef BF_VF_HIER_MIDLEVEL_MAX_NNZ
#  define BF_VF_HIER_MIDLEVEL_MAX_NNZ (10000000ull)
#endif

/* NEW: enable a cheap heuristic to keep the parent CSR block
 * as a single leaf instead of building a whole subtree.
 * 1 = enabled (default), 0 = disabled.
 */
#ifndef BF_VF_HIER_ENABLE_PARENT_CSR_HEURISTIC
#  define BF_VF_HIER_ENABLE_PARENT_CSR_HEURISTIC 1
#endif

/* NEW: “how much more expensive” children are allowed to be before
 * we decide to keep the parent as a single CSR leaf.
 * E.g. 1.5 means “if approx(children_bytes) ≥ 1.5 * parent_bytes,
 * keep parent as a single leaf”.
 */
#ifndef BF_VF_HIER_PARENT_CSR_FACTOR
#  define BF_VF_HIER_PARENT_CSR_FACTOR 1.5
#endif

/* Enable/disable post-build flattening of homogeneous CSR subtrees.
 * 1 = enabled (default), 0 = disabled.
 */
#ifndef BF_VF_HIER_ENABLE_FLATTEN_CSR_SUBTREES
#  define BF_VF_HIER_ENABLE_FLATTEN_CSR_SUBTREES 1
#endif

/* How much cheaper the flat block must be (in estimated bytes)
 * than the current subtree for us to flatten.
 * 1.0 = flatten whenever the flat CSR uses <= current bytes.
 */
#ifndef BF_VF_HIER_FLATTEN_FACTOR
#  define BF_VF_HIER_FLATTEN_FACTOR 1.0
#endif

/* Enable/disable debug logging for CSR subtree flattening */
#ifndef BF_VF_HIER_DEBUG_FLATTEN
#  define BF_VF_HIER_DEBUG_FLATTEN 0
#endif

#if BF_VF_HIER_DEBUG_FLATTEN
static unsigned long long g_vf_flatten_attempts      = 0ull;
static unsigned long long g_vf_flatten_success      = 0ull;
static unsigned long long g_vf_flatten_bytes_before = 0ull;
static unsigned long long g_vf_flatten_bytes_after  = 0ull;
#endif

#ifndef BF_VF_HIER_DEBUG_SVD_FILTER
#  define BF_VF_HIER_DEBUG_SVD_FILTER 0
#endif

#if BF_VF_HIER_DEBUG_SVD_FILTER
static unsigned long long g_svd_near_or_disabled    = 0;
static unsigned long long g_svd_too_small          = 0;
static unsigned long long g_svd_fail               = 0;
static unsigned long long g_svd_rank_reject        = 0;
static unsigned long long g_svd_mem_pre_reject     = 0;  /* NEW: pre-SVD heuristic */
static unsigned long long g_svd_mem_reject         = 0;  /* post-SVD bytesSvd >= bytesCsr */
static unsigned long long g_svd_accept             = 0;
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
  BfQuadtreeNode *row;
  BfQuadtreeNode *col;
} ChildPair;

typedef struct {
  BfSize *globalToRow;  /* size = mapSize, maps global face -> parent row index */
  BfSize *globalToCol;  /* size = mapSize, maps global face -> parent col index */
  BfSize  mapSize;      /* == maxFace + 1 in parent block */
} BfVfFaceMap;

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

static void bfVfFaceMapInitFromParentFaces(
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

static void bfVfFaceMapDeinit(BfVfFaceMap *map)
{
  if (map == NULL) return;
  if (map->globalToRow) bfMemFree(map->globalToRow);
  if (map->globalToCol) bfMemFree(map->globalToCol);
  map->globalToRow = map->globalToCol = NULL;
  map->mapSize = 0;
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

/* NEW: build CSR submatrix from local row/col indices in A_par */
static BfMatCsrReal *
bfMatCsrRealNewSubmatrixFromIndices(BfMatCsrReal const *A_par,
                                    BfSizeArray  const *rowIdx,
                                    BfSizeArray  const *colIdx);


/* NEW: unified CSR leaf policy, using parent face lists.
 *
 * Geometry (leafMin/leafMax) is handled by the caller. This function
 * only decides:
 *   - near (CSR) vs far (eligible for SVD) via eta,
 *   - whether SVD is attempted (minSvdSize),
 *   - whether SVD is accepted (rank fraction, memory vs CSR).
 */
static BfVfHierBlock *makeLeafFromCsrMidlevel(
    BfMatCsrReal const *A_par,
    BfSizeArray  const *rowFaces_par,
    BfSizeArray  const *colFaces_par,
    BfVfFaceMap  const *faceMap,          /* NEW */
    BfQuadtreeNode const *rowNode,
    BfQuadtreeNode const *colNode,
    BfVfBlockMeta const *meta,
    BfReal eta,
    BfReal tol,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac);

/* NEW: mid-level CSR recursive builder (keeps using same A_par + face lists) */
static BfVfHierBlock *buildBlockFromCsrMidlevel(
    BfMatCsrReal const *A_par,
    BfSizeArray  const *rowFaces_par,
    BfSizeArray  const *colFaces_par,
    BfVfFaceMap  const *faceMap,          /* NEW */
    BfQuadtreeNode *rowNode,
    BfQuadtreeNode *colNode,
    BfReal eta,
    BfSize leafMax,
    BfSize leafMin,
    BfSize minArea,
    BfReal tol,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac,
    int depth);

static BfVfHierBlock *buildBlockHybrid(
    BfTrimesh const *tm,
    BfQuadtreeNode  *rowNode,
    BfQuadtreeNode  *colNode,
    BfReal           eta,
    BfSize           leafMax,
    BfSize           leafMin,
    BfReal           minArea,
    BfReal           tol,
    BfSize           minSvdSize,
    BfReal           maxSvdRankFrac);

static BfVfHierBlock *buildSubtreeFromTrimeshUsingCsr(
    BfTrimesh const *tm,
    BfQuadtreeNode  *rowNode,
    BfQuadtreeNode  *colNode,
    BfReal           eta,
    BfSize           leafMax,
    BfSize           leafMin,
    BfReal           minArea,
    BfReal           tol,
    BfSize           minSvdSize,
    BfReal           maxSvdRankFrac);

static void bfVfHierBuildApplyPlan(BfVfHier *vfHier, BfSize tileSize);

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


static double bfVfHierBlockMemBytes(BfVfHierBlock const *block) {
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
static void getNodeInds(BfQuadtreeNode *node,
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
static BfVfBlockMeta getBlockMeta(BfQuadtreeNode *rowNode,
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

  BfBackend backend = BF_BACKEND_ARPACK;
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

  /* Invariant: CSR colind is ALWAYS local (indexes into leaf->colInds) */
  BF_ASSERT(leaf->colsAreLocal);

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
                   BfReal         *y)
{
  bfVfHierApplyMany(vfHier, x, vfHier->n, y, vfHier->n, 1);
}


typedef struct BfVfApplyTask {
  BfVfHierBlock const *leafBlock;
  BfSize iBegin, iEnd;
  BfSize svdIndex;
} BfVfApplyTask;

typedef struct BfVfApplyPlan {
  BfSize tileSize;
  BfSize numTiles;
  BfSize *tilePtr;
  BfVfApplyTask *tasks;
  BfSize numTasks;

  BfVfHierBlock const **svdBlocks;
  BfSize numSvdBlocks;
} BfVfApplyPlan;


void bfVfHierDeinit(BfVfHier *vfHier) {
  if (vfHier == NULL) return;
  if (vfHier->root != NULL)
    bfVfHierBlockDeinitAndDealloc(&vfHier->root);
  vfHier->trimesh = NULL;
  vfHier->n = 0;
  if (vfHier->applyPlan) {
      BfVfApplyPlan *p = vfHier->applyPlan;
      if (p->tilePtr) bfMemFree(p->tilePtr);
      if (p->tasks) bfMemFree(p->tasks);
      if (p->svdBlocks) bfMemFree(p->svdBlocks);
      bfMemFree(p);
      vfHier->applyPlan = NULL;
    }

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
  BfSize *colind = (BfSize *)bfMatCsrRealGetColindConstPtr(Acsr);

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
  BfQuadtree const *qt =
    bfQuadtreeNodeGetQuadtree((BfQuadtreeNode *)rowNode);
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

  /* NEAR or SVD disabled: keep CSR leaf, but ENFORCE local columns invariant */
  if (!far || minSvdSize == 0) {
    BfSize nFaces = bfTrimeshGetNumFaces(tm);
    reindexCsrColsToLocal(Acsr, &colInds, nFaces);

    block->kind                = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat     = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;

    /* IMPORTANT: row_i0/row_i1 are in permutation (tree) index space. */
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
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
    /* MUST be tree index range, not global face ids */
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
    return block;
  }

  /* 1) Memory estimate for CSR */
  BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(Acsr);
  BfSize nnz = rp[mA];

  double bytesCsr = 8.0*nnz + 8.0*nnz + 8.0*(mA + 1);  /* data + colind + rowptr */

  /* NEW: cheap early SVD rejection.
   *
   * If even a rank-1 SVD (U in R^{m×1}, VT in R^{1×n}, S in R^1) would
   * use more bytes than CSR, there is no point calling ARPACK at all.
   *
   * bytesSvd_min = 8 * (mA*1 + nA*1 + 1)  [U + VT + S]
   * If bytesSvd_min >= bytesCsr, skip SVD and keep CSR leaf.
   */
  double bytesSvd_min = 8.0 * ((double)mA + (double)nA + 1.0);
  if (bytesSvd_min >= bytesCsr) {
#if BF_VF_HIER_DEBUG_SVD_FILTER
    g_svd_mem_pre_reject++;
#endif
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat     = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    /* MUST be tree index range, not global face ids */
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
    return block;
  }

  BfTruncSpec truncSpec;
  truncSpec.usingTol = 1;
  truncSpec.tol = tol;

  BfMat *U = NULL, *VT = NULL;
  BfMatDiagReal *S = NULL;

  #if BF_VF_HIER_TIME_SVD
    double t_svd_start = bfVfHierNowSecs();
  #endif

  BfBackend backend = BF_BACKEND_ARPACK;
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

  (void)truncated; /* you can print/log if desired */

  if (U == NULL || S == NULL || VT == NULL) {
    /* Fall back to sparse leaf on failure */
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = rowInds;
    block->data.sparse.colInds = colInds;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
//    bfMatDelete(&A_dense);
    /* MUST be tree index range, not global face ids */
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
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
//    bfMatDelete(&A_dense);
    /* MUST be tree index range, not global face ids */
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
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
//    bfMatDelete(&A_dense);
    /* MUST be tree index range, not global face ids */
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
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
  block->data.svd.U_cache  = NULL;
  block->data.svd.S_cache  = NULL;
  block->data.svd.VT_cache = NULL;

  /* rowInds/colInds are now owned by the SVD leaf; don't deinit them here */

  bfMatCsrRealDeinitAndDealloc(&Acsr);
//  bfMatDelete(&A_dense);
  block->data.svd.row_i0 = meta->i0;
  block->data.svd.row_i1 = meta->i1;
  block->data.svd.col_j0 = meta->j0;
  block->data.svd.col_j1 = meta->j1;
  return block;
}

/* ============================================================
 * CSR SUBMATRIX BUILDER (from an existing parent CSR)
 * ============================================================ */

#if BF_VF_HIER_ENABLE_LEGACY_BUILD
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
#endif

/* ============================================================
 * CSR SUBMATRIX BUILDER FROM LOCAL INDICES (A_par row/col indices)
 * ============================================================ */

/* Build CSR submatrix A_sub = A_par[rowIdx, colIdx], where
 * rowIdx and colIdx contain LOCAL indices into A_par:
 *
 *   0 <= rowIdx[i] < nRows(A_par)
 *   0 <= colIdx[j] < nCols(A_par)
 *
 * Columns of A_sub are LOCAL 0..|colIdx|-1 by construction.
 */
static BfMatCsrReal *
bfMatCsrRealNewSubmatrixFromIndices(BfMatCsrReal const *A_par,
                                    BfSizeArray  const *rowIdx,
                                    BfSizeArray  const *colIdx)
{
  BF_ASSERT(A_par  != NULL);
  BF_ASSERT(rowIdx != NULL);
  BF_ASSERT(colIdx != NULL);

  #if BF_VF_HIER_TIME_SVD
    double t_slice_start = bfVfHierNowSecs();
  #endif

  BfMat *A_base = bfMatCsrRealToMat((BfMatCsrReal *)A_par);
  BfSize nRows  = bfMatGetNumRows(A_base);
  BfSize nCols  = bfMatGetNumCols(A_base);

  BfSize m = bfSizeArrayGetSize((BfSizeArray *)rowIdx);
  BfSize n = bfSizeArrayGetSize((BfSizeArray *)colIdx);

  /* Trivial empty case */
  if (m == 0 || n == 0) {
    return bfMatCsrRealNewFromArrays(
        0, 0,
        bfSizeArrayNewWithDefaultCapacity(),
        bfSizeArrayNewWithDefaultCapacity(),
        bfRealArrayNewWithDefaultCapacity(),
        BF_POLICY_STEAL);
  }

  /* Sanity checks: indices must be in range */
  for (BfSize i = 0; i < m; ++i) {
    BfSize r = bfSizeArrayGet((BfSizeArray *)rowIdx, i);
    if (r >= nRows) {
      fprintf(stderr,
              "[bf] bfMatCsrRealNewSubmatrixFromIndices: BAD row index %lu >= %lu\n",
              (unsigned long)r, (unsigned long)nRows);
      return NULL;
    }
  }

  for (BfSize j = 0; j < n; ++j) {
    BfSize c = bfSizeArrayGet((BfSizeArray *)colIdx, j);
    if (c >= nCols) {
      fprintf(stderr,
              "[bf] bfMatCsrRealNewSubmatrixFromIndices: BAD col index %lu >= %lu\n",
              (unsigned long)c, (unsigned long)nCols);
      return NULL;
    }
  }

  /* Parent CSR data */
  BfSize const *rp_par = bfMatCsrRealGetRowptrConstPtr(A_par);
  BfSize const *ci_par = bfMatCsrRealGetColindConstPtr(A_par);
  BfReal const *da_par = bfMatCsrRealGetDataConstPtr(A_par);

  BF_ASSERT(rp_par != NULL);
  BF_ASSERT(ci_par != NULL);
  BF_ASSERT(da_par != NULL);

  /* Build global->local column map over [0..nCols) */
  BfSize *globalToLocalCol = bfMemAlloc(nCols, sizeof(BfSize));
  if (globalToLocalCol == NULL)
    return NULL;

  for (BfSize i = 0; i < nCols; ++i)
    globalToLocalCol[i] = BF_SIZE_BAD_VALUE;

  for (BfSize j = 0; j < n; ++j) {
    BfSize g = bfSizeArrayGet((BfSizeArray *)colIdx, j);
    BF_ASSERT(g < nCols);
    globalToLocalCol[g] = j;  /* col index in A_par -> local col in sub-block */
  }

  /* Dynamic CSR arrays for the submatrix */
  BfSizeArray *rowptr = bfSizeArrayNewWithDefaultCapacity();
  BfSizeArray *colind = bfSizeArrayNewWithDefaultCapacity();
  BfRealArray *data   = bfRealArrayNewWithDefaultCapacity();

  bfSizeArrayAppend(rowptr, 0); /* rowptr[0] = 0 */

  BfSize nnz_so_far = 0;

  for (BfSize i = 0; i < m; ++i) {
    BfSize gRow = bfSizeArrayGet((BfSizeArray *)rowIdx, i);
    BF_ASSERT(gRow < nRows);

    BfSize r0 = rp_par[gRow];
    BfSize r1 = rp_par[gRow + 1];

    for (BfSize k = r0; k < r1; ++k) {
      BfSize gCol = ci_par[k];
      BF_ASSERT(gCol < nCols);

      BfSize lCol = globalToLocalCol[gCol];
      if (lCol == BF_SIZE_BAD_VALUE)
        continue;  /* col not in selection: skip */

      bfSizeArrayAppend(colind, lCol);
      bfRealArrayAppend(data, da_par[k]);
      ++nnz_so_far;
    }

    bfSizeArrayAppend(rowptr, nnz_so_far);
  }

  bfMemFree(globalToLocalCol);

  /* Convert dynamic arrays into CSR (takes ownership) */
  BfMatCsrReal *A_sub =
    bfMatCsrRealNewFromArrays(m, n, rowptr, colind, data, BF_POLICY_STEAL);

  #if BF_VF_HIER_TIME_SVD
    double t_slice_end = bfVfHierNowSecs();
    #pragma omp atomic
    g_vf_csr_slice_time += (t_slice_end - t_slice_start);
  #endif

  return A_sub;
}

/* ============================================================
 * MID-LEVEL CSR LEAF BUILDER (OLD + NEW hybrid)
 * ============================================================ */

static BfVfHierBlock *makeLeafFromCsrMidlevel(
    BfMatCsrReal const *A_par,
    BfSizeArray  const *rowFaces_par,
    BfSizeArray  const *colFaces_par,
    BfVfFaceMap  const *faceMap,      /* NEW */
    BfQuadtreeNode const *rowNode,
    BfQuadtreeNode const *colNode,
    BfVfBlockMeta const *meta,
    BfReal eta,
    BfReal tol,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac)
{
  if (meta->empty)
    return NULL;

#if BF_VF_HIER_TIME_SVD
  double t_map_start = bfVfHierNowSecs();
#endif

  BfQuadtree const *qt =
    bfQuadtreeNodeGetQuadtree((BfQuadtreeNode *)rowNode);

  BfSizeArray childRowFaces;
  BfSizeArray childColFaces;
  bfSizeArrayInitWithDefaultCapacity(&childRowFaces);
  bfSizeArrayInitWithDefaultCapacity(&childColFaces);

  getNodeInds((BfQuadtreeNode *)rowNode, qt, &childRowFaces);
  getNodeInds((BfQuadtreeNode *)colNode, qt, &childColFaces);

  BfSize mChild = bfSizeArrayGetSize(&childRowFaces);
  BfSize nChild = bfSizeArrayGetSize(&childColFaces);

  if (mChild == 0 || nChild == 0) {
    bfSizeArrayDeinit(&childRowFaces);
    bfSizeArrayDeinit(&childColFaces);
    return NULL;
  }

  /* Use cached global->parent-local maps */
  BF_ASSERT(faceMap != NULL);
  BfSize mapSize           = faceMap->mapSize;
  BfSize const *globalToRow = faceMap->globalToRow;
  BfSize const *globalToCol = faceMap->globalToCol;

  BfSizeArray rowIdxPar;
  BfSizeArray colIdxPar;
  bfSizeArrayInitWithDefaultCapacity(&rowIdxPar);
  bfSizeArrayInitWithDefaultCapacity(&colIdxPar);

  for (BfSize i = 0; i < mChild; ++i) {
    BfSize gRow = bfSizeArrayGet(&childRowFaces, i);
    if (gRow >= mapSize)
      continue;
    BfSize rLoc = globalToRow[gRow];
    if (rLoc == BF_SIZE_BAD_VALUE)
      continue;
    bfSizeArrayAppend(&rowIdxPar, rLoc);
  }

  for (BfSize j = 0; j < nChild; ++j) {
    BfSize gCol = bfSizeArrayGet(&childColFaces, j);
    if (gCol >= mapSize)
      continue;
    BfSize cLoc = globalToCol[gCol];
    if (cLoc == BF_SIZE_BAD_VALUE)
      continue;
    bfSizeArrayAppend(&colIdxPar, cLoc);
  }

#if BF_VF_HIER_TIME_SVD
  double t_map_end = bfVfHierNowSecs();
  #pragma omp atomic
  g_vf_leaf_map_time += (t_map_end - t_map_start);
#endif

  BfSize mSel = bfSizeArrayGetSize(&rowIdxPar);
  BfSize nSel = bfSizeArrayGetSize(&colIdxPar);

  if (mSel == 0 || nSel == 0) {
    bfSizeArrayDeinit(&rowIdxPar);
    bfSizeArrayDeinit(&colIdxPar);
    bfSizeArrayDeinit(&childRowFaces);
    bfSizeArrayDeinit(&childColFaces);
    return NULL;
  }

  /* Build CSR sub-block from local indices in A_par */
  BfMatCsrReal *Acsr =
    bfMatCsrRealNewSubmatrixFromIndices(A_par, &rowIdxPar, &colIdxPar);

  bfSizeArrayDeinit(&rowIdxPar);
  bfSizeArrayDeinit(&colIdxPar);

  if (Acsr == NULL) {
    bfSizeArrayDeinit(&childRowFaces);
    bfSizeArrayDeinit(&childColFaces);
    return NULL;
  }

  BfMat *A = bfMatCsrRealToMat(Acsr);
  BfSize mA = bfMatGetNumRows(A);
  BfSize nA = bfMatGetNumCols(A);

  {
    BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(Acsr);
    BfReal const *da = bfMatCsrRealGetDataConstPtr(Acsr);
    BF_ASSERT(rp != NULL);
    BF_ASSERT(da != NULL);

    BfSize nnz = rp[mA];
    BfReal maxAbs = 0;
    for (BfSize k = 0; k < nnz; ++k) {
      BfReal v = da[k];
      if (v < 0) v = -v;
      if (v > maxAbs) maxAbs = v;
    }

    if (nnz == 0 || maxAbs == 0) {
      /* This (rowNode, colNode) block is identically zero.
       * Represent it by *no block* in the hierarchy.
       */
      bfMatCsrRealDeinitAndDealloc(&Acsr);
      bfSizeArrayDeinit(&childRowFaces);
      bfSizeArrayDeinit(&childColFaces);
      return NULL;
    }
  }

  /* Far vs near test based on geometry, exactly like other paths */
  BfBool far = isFar(rowNode, colNode, eta);

  BfVfHierBlock *block = bfVfHierBlockNew();

  /* NEAR or SVD disabled: keep CSR leaf (columns are LOCAL by construction) */
  if (!far || minSvdSize == 0) {
#if BF_VF_HIER_DEBUG_SVD_FILTER
  g_svd_near_or_disabled++;
#endif
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = childRowFaces;  /* global faces */
    block->data.sparse.colInds = childColFaces;  /* global faces */
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
    return block;
  }

  /* FAR + SVD enabled: small blocks are kept as CSR */
  unsigned long long blockSize =
    (unsigned long long)mA * (unsigned long long)nA;

  if (blockSize < (unsigned long long)minSvdSize) {
#if BF_VF_HIER_DEBUG_SVD_FILTER
  g_svd_too_small++;
#endif
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = childRowFaces;
    block->data.sparse.colInds = childColFaces;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
    return block;
  }

    /* Also treat very small dimensions as “too small” for ARPACK SVD. */
  if (mA < 3 || nA < 3) {
#if BF_VF_HIER_DEBUG_SVD_FILTER
    g_svd_too_small++;
#endif
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = childRowFaces;
    block->data.sparse.colInds = childColFaces;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
    return block;
  }

  /* Memory estimate for CSR */
  BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(Acsr);
  BfSize const *ci = bfMatCsrRealGetColindConstPtr(Acsr);
  BfReal const *da = bfMatCsrRealGetDataConstPtr(Acsr);

  BF_ASSERT(rp && ci && da);

  BfSize nnz = rp[mA];
  double bytesCsr = 8.0*nnz          /* data */
                  + 8.0*nnz          /* colind */
                  + 8.0*(mA + 1);    /* rowptr */

  /* NEW: cheap early SVD rejection (same logic as trimesh path).
   * If even rank-1 SVD storage would exceed CSR storage,
   * skip ARPACK and keep CSR leaf.
   */
  double bytesSvd_min = 8.0 * ((double)mA + (double)nA + 1.0);
  if (bytesSvd_min >= bytesCsr) {
#if BF_VF_HIER_DEBUG_SVD_FILTER
    g_svd_mem_pre_reject++;
#endif
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = childRowFaces;  /* global faces */
    block->data.sparse.colInds = childColFaces;  /* global faces */
    block->data.sparse.mat     = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
    return block;
  }

  /* Truncated SVD on CSR via ARPACK */
  BfTruncSpec truncSpec;
  truncSpec.usingTol = 1;
  truncSpec.tol      = tol;

#if BF_VF_HIER_TIME_SVD
  double t_svd_start = bfVfHierNowSecs();
#endif

  BfMat *U  = NULL;
  BfMat *VT = NULL;
  BfMatDiagReal *S = NULL;


  BfBackend backend = BF_BACKEND_ARPACK;
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

  (void)truncated;

  if (U == NULL || S == NULL || VT == NULL) {
    /* SVD failed: fall back to CSR leaf */
#if BF_VF_HIER_DEBUG_SVD_FILTER
  g_svd_fail++;
#endif
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = childRowFaces;
    block->data.sparse.colInds = childColFaces;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
    return block;
  }

  BfSize k = bfMatGetNumCols(U); /* rank */
  BfSize minDim = mA < nA ? mA : nA;
  double rankFrac = minDim > 0 ? (double)k / (double)minDim : 1.0;

  /* Reject SVD if rank too high */
  if (rankFrac > maxSvdRankFrac) {
#if BF_VF_HIER_DEBUG_SVD_FILTER
  g_svd_rank_reject++;
#endif
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = childRowFaces;
    block->data.sparse.colInds = childColFaces;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;

    bfMatDelete(&U);
    bfMatDiagRealDeinitAndDealloc(&S);
    bfMatDelete(&VT);
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
    return block;
  }

  /* Memory estimate for SVD representation */
  double bytesSvd =
    8.0*((double)mA*k + (double)nA*k + (double)k); /* U + VT + diag(S) */

  if (bytesSvd >= bytesCsr) {
    /* SVD not worth it; keep sparse leaf */
#if BF_VF_HIER_DEBUG_SVD_FILTER
  g_svd_mem_reject++;
#endif
    block->kind = BF_VF_HIER_BLOCK_SPARSE;
    block->data.sparse.rowInds = childRowFaces;
    block->data.sparse.colInds = childColFaces;
    block->data.sparse.mat = Acsr;
    block->data.sparse.colsAreLocal = BF_TRUE;

    bfMatDelete(&U);
    bfMatDiagRealDeinitAndDealloc(&S);
    bfMatDelete(&VT);
    block->data.sparse.row_i0 = meta->i0;
    block->data.sparse.row_i1 = meta->i1;
    block->data.sparse.col_j0 = meta->j0;
    block->data.sparse.col_j1 = meta->j1;
    return block;
  }

  /* Build MatProduct P = U S VT */
  BfMatProduct *Pprod = bfMatProductNew();
  bfMatProductInit(Pprod);
  bfMatProductPostMultiply(Pprod, U);
  bfMatProductPostMultiply(Pprod, bfMatDiagRealToMat(S));
  bfMatProductPostMultiply(Pprod, VT);

  BfMat *P = bfMatProductToMat(Pprod);

  /* SVD leaf: store global row/col faces + MatProduct */
  block->kind = BF_VF_HIER_BLOCK_SVD;
  block->data.svd.rowInds = childRowFaces;  /* global faces */
  block->data.svd.colInds = childColFaces;  /* global faces */
  block->data.svd.mat     = P;
  block->data.svd.rank    = k;
  block->data.svd.work    = NULL;
  block->data.svd.workLen = 0;
  block->data.svd.U_cache  = NULL;
  block->data.svd.S_cache  = NULL;
  block->data.svd.VT_cache = NULL;

#if BF_VF_HIER_DEBUG_SVD_FILTER
  g_svd_accept++;
#endif

  bfMatCsrRealDeinitAndDealloc(&Acsr);
  block->data.svd.row_i0 = meta->i0;
  block->data.svd.row_i1 = meta->i1;
  block->data.svd.col_j0 = meta->j0;
  block->data.svd.col_j1 = meta->j1;
  return block;
}

/* ============================================================
 * MID-LEVEL CSR RECURSIVE BUILDER (OLD + NEW hybrid)
 * ============================================================ */
static BfVfHierBlock *buildBlockFromCsrMidlevel(
    BfMatCsrReal const *A_par,
    BfSizeArray  const *rowFaces_par,
    BfSizeArray  const *colFaces_par,
    BfVfFaceMap  const *faceMap,
    BfQuadtreeNode *rowNode,
    BfQuadtreeNode *colNode,
    BfReal eta,
    BfSize leafMax,
    BfSize leafMin,
    BfSize minArea,
    BfReal tol,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac,
    int depth)
{
  BfVfBlockMeta meta = getBlockMeta(rowNode, colNode, leafMax);
  if (meta.empty)
    return NULL;

  BfSize mi = meta.mi;
  BfSize mj = meta.mj;
  bool leafI = meta.leafI;
  bool leafJ = meta.leafJ;
  bool small = meta.small;

  unsigned long long area =
    (unsigned long long)mi * (unsigned long long)mj;

  /* --- 1) flat candidate for THIS block (CSR or SVD) --- */
  BfVfHierBlock *flatBlock = makeLeafFromCsrMidlevel(
      A_par,
      rowFaces_par,
      colFaces_par,
      faceMap,
      rowNode,
      colNode,
      &meta,
      eta,
      tol,
      minSvdSize,
      maxSvdRankFrac);

  /* If we reached a geometric leaf or very small area, don't bother recursing:
   * just return the best flat representation we have (or NULL if the block is zero).
   */
  bool forceLeaf =
      (leafI && leafJ) ||
      (small && mi >= leafMin && mj >= leafMin) ||
      (area <= (unsigned long long)minArea);

  if (forceLeaf) {
    return flatBlock;  /* may be NULL for exactly-zero block */
  }

  /* --- 2) build hierarchical children as before --- */

  BfVfHierBlock *nodeBlock = bfVfHierBlockNew();
  nodeBlock->kind = BF_VF_HIER_BLOCK_NODE;
  bfInitPtrArray(&nodeBlock->data.node.children, 4);

  BfTreeNode *ni = bfQuadtreeNodeToTreeNode(rowNode);
  BfTreeNode *nj = bfQuadtreeNodeToTreeNode(colNode);

  if (!leafI && !leafJ) {
    /* --- case 1: split both row and col sides --- */

    BfSize maxChildrenI = bfTreeNodeGetMaxNumChildren(ni);
    BfSize maxChildrenJ = bfTreeNodeGetMaxNumChildren(nj);

    ChildPair *pairs = bfMemAlloc(maxChildrenI * maxChildrenJ, sizeof(ChildPair));
    BfSize numPairs = 0;

    /* 1) collect all (rowChild, colChild) pairs serially */
    for (BfSize a = 0; a < maxChildrenI; ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfTreeNode *ci = bfTreeNodeGetChild(ni, a);
      BfQuadtreeNode *nai = bfTreeNodeToQuadtreeNode(ci);

      for (BfSize b = 0; b < maxChildrenJ; ++b) {
        if (!bfTreeNodeHasChild(nj, b)) continue;
        BfTreeNode *cj = bfTreeNodeGetChild(nj, b);
        BfQuadtreeNode *nbj = bfTreeNodeToQuadtreeNode(cj);

        pairs[numPairs].row = nai;
        pairs[numPairs].col = nbj;
        ++numPairs;
      }
    }

    if (numPairs == 0) {
      bfMemFree(pairs);
      bfVfHierBlockDeinitAndDealloc(&nodeBlock);
      return flatBlock;  /* may be NULL */
    }

    /* 2) allocate per-pair result slots */
    BfVfHierBlock **childBlocks =
      bfMemAlloc(numPairs, sizeof(BfVfHierBlock *));

    /* 3) parallel recursion over pairs */
    int childDepth = depth + 1;

    #pragma omp parallel for schedule(dynamic) if (depth < 4)
    for (BfSize idx = 0; idx < numPairs; ++idx) {
      childBlocks[idx] =
        buildBlockFromCsrMidlevel(
            A_par,
            rowFaces_par,
            colFaces_par,
            faceMap,
            pairs[idx].row,
            pairs[idx].col,
            eta,
            leafMax,
            leafMin,
            minArea,
            tol,
            minSvdSize,
            maxSvdRankFrac,
            childDepth);
    }

    /* 4) serial append into nodeBlock children */
    for (BfSize idx = 0; idx < numPairs; ++idx) {
      if (childBlocks[idx] != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, childBlocks[idx]);
    }

    bfMemFree(childBlocks);
    bfMemFree(pairs);

  } else if (!leafI) {
    /* --- case 2: split row side only --- */

    BfSize maxChildrenI = bfTreeNodeGetMaxNumChildren(ni);

    ChildPair *pairs = bfMemAlloc(maxChildrenI, sizeof(ChildPair));
    BfSize numPairs = 0;

    for (BfSize a = 0; a < maxChildrenI; ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfTreeNode *ci = bfTreeNodeGetChild(ni, a);
      BfQuadtreeNode *nai = bfTreeNodeToQuadtreeNode(ci);

      pairs[numPairs].row = nai;
      pairs[numPairs].col = colNode;
      ++numPairs;
    }

    if (numPairs == 0) {
      bfMemFree(pairs);
      bfVfHierBlockDeinitAndDealloc(&nodeBlock);
      return flatBlock;
    }

    BfVfHierBlock **childBlocks =
      bfMemAlloc(numPairs, sizeof(BfVfHierBlock *));

    int childDepth = depth + 1;

    #pragma omp parallel for schedule(dynamic) if (depth < 4)
    for (BfSize idx = 0; idx < numPairs; ++idx) {
      childBlocks[idx] =
        buildBlockFromCsrMidlevel(
            A_par,
            rowFaces_par,
            colFaces_par,
            faceMap,
            pairs[idx].row,
            pairs[idx].col,  /* == colNode */
            eta,
            leafMax,
            leafMin,
            minArea,
            tol,
            minSvdSize,
            maxSvdRankFrac,
            childDepth);
    }

    for (BfSize idx = 0; idx < numPairs; ++idx) {
      if (childBlocks[idx] != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, childBlocks[idx]);
    }

    bfMemFree(childBlocks);
    bfMemFree(pairs);

  } else { /* !leafJ */
    /* --- case 3: split col side only --- */

    BfSize maxChildrenJ = bfTreeNodeGetMaxNumChildren(nj);

    ChildPair *pairs = bfMemAlloc(maxChildrenJ, sizeof(ChildPair));
    BfSize numPairs = 0;

    for (BfSize b = 0; b < maxChildrenJ; ++b) {
      if (!bfTreeNodeHasChild(nj, b)) continue;
      BfTreeNode *cj = bfTreeNodeGetChild(nj, b);
      BfQuadtreeNode *nbj = bfTreeNodeToQuadtreeNode(cj);

      pairs[numPairs].row = rowNode;
      pairs[numPairs].col = nbj;
      ++numPairs;
    }

    if (numPairs == 0) {
      bfMemFree(pairs);
      bfVfHierBlockDeinitAndDealloc(&nodeBlock);
      return flatBlock;
    }

    BfVfHierBlock **childBlocks =
      bfMemAlloc(numPairs, sizeof(BfVfHierBlock *));

    int childDepth = depth + 1;

    #pragma omp parallel for schedule(dynamic) if (depth < 4)
    for (BfSize idx = 0; idx < numPairs; ++idx) {
      childBlocks[idx] =
        buildBlockFromCsrMidlevel(
            A_par,
            rowFaces_par,
            colFaces_par,
            faceMap,
            pairs[idx].row,  /* == rowNode */
            pairs[idx].col,
            eta,
            leafMax,
            leafMin,
            minArea,
            tol,
            minSvdSize,
            maxSvdRankFrac,
            childDepth);
    }

    for (BfSize idx = 0; idx < numPairs; ++idx) {
      if (childBlocks[idx] != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, childBlocks[idx]);
    }

    bfMemFree(childBlocks);
    bfMemFree(pairs);
  }

  /* If recursion produced nothing, fall back to flat candidate (could be NULL) */
  if (bfPtrArraySize(&nodeBlock->data.node.children) == 0) {
    bfVfHierBlockDeinitAndDealloc(&nodeBlock);
    return flatBlock;
  }

  /* --- 3) choose cheaper: flat vs hierarchy --- */

  if (flatBlock == NULL) {
    /* No flat representation (probably identically-zero); keep hierarchy */
    return nodeBlock;
  }

  double bytesFlat = bfVfHierBlockMemBytes(flatBlock);
  double bytesHier = bfVfHierBlockMemBytes(nodeBlock);

  if (bytesFlat <= bytesHier) {
    /* Flat leaf wins: discard hierarchy */
    bfVfHierBlockDeinitAndDealloc(&nodeBlock);
    return flatBlock;
  } else {
    /* Hierarchy wins: discard flat leaf */
    bfVfHierBlockDeinitAndDealloc(&flatBlock);
    return nodeBlock;
  }
}


/* ============================================================
 * HYBRID BUILDER: geometric recursion + mid-level OLD+NEW
 * ============================================================ */

/* For large blocks: only split geometrically (no FF).
 * Once mi*mj is small enough, we raytrace that block ONCE
 * from the trimesh, then build the whole subtree below it
 * using CSR slicing (buildBlockFromCsrMidlevel).
 */
static BfVfHierBlock *buildBlockHybrid(
    BfTrimesh const *tm,
    BfQuadtreeNode  *rowNode,
    BfQuadtreeNode  *colNode,
    BfReal           eta,
    BfSize           leafMax,
    BfSize           leafMin,
    BfReal           minArea,      /* NEW */
    BfReal           tol,
    BfSize           minSvdSize,
    BfReal           maxSvdRankFrac)
{
  BfVfBlockMeta meta = getBlockMeta(rowNode, colNode, leafMax);
  if (meta.empty)
    return NULL;

  BfSize mi = meta.mi;
  BfSize mj = meta.mj;

  unsigned long long area =
    (unsigned long long)mi * (unsigned long long)mj;

  /* NEW: stop recursion early for small blocks (flux-like _min_size).
   * If the block is small enough by area, treat it as a single leaf,
   * letting the leaf policy decide CSR vs SVD.
   */
  if (area <= (unsigned long long)minArea) {
    BfVfHierBlock *leaf =
      makeLeafWithOptionalSvd(tm,
                              rowNode,
                              colNode,
                              &meta,
                              eta,
                              tol,
                              leafMax,
                              leafMin,
                              minSvdSize,
                              maxSvdRankFrac);
    if (leaf != NULL)
      return leaf;
    /* If leaf creation fails (e.g. dimensions < leafMin),
     * fall back to the usual logic below.
     */
  }

  /* Approximate nnz as if the block were dense.
   * This is only a safety cap to avoid raytracing
   * blocks that are too large to hold in memory.
   */
  unsigned long long approxNnz = area;

  if (approxNnz <= BF_VF_HIER_MIDLEVEL_MAX_NNZ) {
    /* Block is small enough: raytrace once from tm (OLD)
     * and then use CSR slicing for all descendants (NEW).
     */
    return buildSubtreeFromTrimeshUsingCsr(
        tm, rowNode, colNode,
        eta, leafMax, leafMin, minArea,
        tol, minSvdSize, maxSvdRankFrac);
  }

  /* Too large to realize as one CSR block: only split geometrically. */
  BfVfHierBlock *nodeBlock = bfVfHierBlockNew();
  nodeBlock->kind = BF_VF_HIER_BLOCK_NODE;
  bfInitPtrArray(&nodeBlock->data.node.children, 4);

  BfTreeNode *ni = bfQuadtreeNodeToTreeNode(rowNode);
  BfTreeNode *nj = bfQuadtreeNodeToTreeNode(colNode);

  bool leafI = meta.leafI;
  bool leafJ = meta.leafJ;

  if (!leafI && !leafJ) {
    BfSize maxChildrenI = bfTreeNodeGetMaxNumChildren(ni);
    BfSize maxChildrenJ = bfTreeNodeGetMaxNumChildren(nj);

    ChildPair *pairs = bfMemAlloc(maxChildrenI * maxChildrenJ, sizeof(ChildPair));
    BfSize numPairs = 0;

    for (BfSize a = 0; a < maxChildrenI; ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfTreeNode *ci = bfTreeNodeGetChild(ni, a);
      BfQuadtreeNode *nai = bfTreeNodeToQuadtreeNode(ci);

      for (BfSize b = 0; b < maxChildrenJ; ++b) {
        if (!bfTreeNodeHasChild(nj, b)) continue;
        BfTreeNode *cj = bfTreeNodeGetChild(nj, b);
        BfQuadtreeNode *nbj = bfTreeNodeToQuadtreeNode(cj);

        pairs[numPairs].row = nai;
        pairs[numPairs].col = nbj;
        ++numPairs;
      }
    }

    if (numPairs == 0) {
      bfMemFree(pairs);
      bfVfHierBlockDeinitAndDealloc(&nodeBlock);
      return NULL;
    }

    BfVfHierBlock **childBlocks =
      bfMemAlloc(numPairs, sizeof(BfVfHierBlock *));

    #pragma omp parallel for schedule(dynamic)
    for (BfSize idx = 0; idx < numPairs; ++idx) {
      childBlocks[idx] =
        buildBlockHybrid(
            tm,
            pairs[idx].row,
            pairs[idx].col,
            eta,
            leafMax,
            leafMin,
            minArea,
            tol,
            minSvdSize,
            maxSvdRankFrac);
    }

    for (BfSize idx = 0; idx < numPairs; ++idx) {
      if (childBlocks[idx] != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, childBlocks[idx]);
    }

    bfMemFree(childBlocks);
    bfMemFree(pairs);

  } else if (!leafI) {
    BfSize maxChildrenI = bfTreeNodeGetMaxNumChildren(ni);

    ChildPair *pairs = bfMemAlloc(maxChildrenI, sizeof(ChildPair));
    BfSize numPairs = 0;

    for (BfSize a = 0; a < maxChildrenI; ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfTreeNode *ci = bfTreeNodeGetChild(ni, a);
      BfQuadtreeNode *nai = bfTreeNodeToQuadtreeNode(ci);

      pairs[numPairs].row = nai;
      pairs[numPairs].col = colNode;
      ++numPairs;
    }

    if (numPairs == 0) {
      bfMemFree(pairs);
      bfVfHierBlockDeinitAndDealloc(&nodeBlock);
      return NULL;
    }

    BfVfHierBlock **childBlocks =
      bfMemAlloc(numPairs, sizeof(BfVfHierBlock *));

    #pragma omp parallel for schedule(dynamic)
    for (BfSize idx = 0; idx < numPairs; ++idx) {
      childBlocks[idx] =
        buildBlockHybrid(
            tm,
            pairs[idx].row,
            pairs[idx].col,  /* colNode */
            eta,
            leafMax,
            leafMin,
            minArea,
            tol,
            minSvdSize,
            maxSvdRankFrac);
    }

    for (BfSize idx = 0; idx < numPairs; ++idx) {
      if (childBlocks[idx] != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, childBlocks[idx]);
    }

    bfMemFree(childBlocks);
    bfMemFree(pairs);

  } else { /* !leafJ */
    BfSize maxChildrenJ = bfTreeNodeGetMaxNumChildren(nj);

    ChildPair *pairs = bfMemAlloc(maxChildrenJ, sizeof(ChildPair));
    BfSize numPairs = 0;

    for (BfSize b = 0; b < maxChildrenJ; ++b) {
      if (!bfTreeNodeHasChild(nj, b)) continue;
      BfTreeNode *cj = bfTreeNodeGetChild(nj, b);
      BfQuadtreeNode *nbj = bfTreeNodeToQuadtreeNode(cj);

      pairs[numPairs].row = rowNode;
      pairs[numPairs].col = nbj;
      ++numPairs;
    }

    if (numPairs == 0) {
      bfMemFree(pairs);
      bfVfHierBlockDeinitAndDealloc(&nodeBlock);
      return NULL;
    }

    BfVfHierBlock **childBlocks =
      bfMemAlloc(numPairs, sizeof(BfVfHierBlock *));

    #pragma omp parallel for schedule(dynamic)
    for (BfSize idx = 0; idx < numPairs; ++idx) {
      childBlocks[idx] =
        buildBlockHybrid(
            tm,
            pairs[idx].row,  /* rowNode */
            pairs[idx].col,
            eta,
            leafMax,
            leafMin,
            minArea,
            tol,
            minSvdSize,
            maxSvdRankFrac);
    }

    for (BfSize idx = 0; idx < numPairs; ++idx) {
      if (childBlocks[idx] != NULL)
        bfPtrArrayAppend(&nodeBlock->data.node.children, childBlocks[idx]);
    }

    bfMemFree(childBlocks);
    bfMemFree(pairs);
  }

  if (bfPtrArraySize(&nodeBlock->data.node.children) == 0) {
    bfVfHierBlockDeinitAndDealloc(&nodeBlock);
    return NULL;
  }

  return nodeBlock;
}

/* Build a whole subtree for (rowNode, colNode) by:
 *
 *  1) Raytracing the FULL block from trimesh into a CSR A_par
 *     using bfMatCsrRealNewViewFactorMatrixFromTrimesh (OLD),
 *  2) Using buildBlockFromCsrMidlevel (NEW) to build all
 *     children/leaf blocks by CSR slicing (no more raytracing),
 *  3) Freeing A_par and the parent face lists afterwards.
 */
static BfVfHierBlock *buildSubtreeFromTrimeshUsingCsr(
    BfTrimesh const *tm,
    BfQuadtreeNode  *rowNode,
    BfQuadtreeNode  *colNode,
    BfReal           eta,
    BfSize           leafMax,
    BfSize           leafMin,
    BfReal           minArea,          /* NEW */
    BfReal           tol,
    BfSize           minSvdSize,
    BfReal           maxSvdRankFrac)
{
  BfQuadtree const *qt =
    bfQuadtreeNodeGetQuadtree(rowNode);

  /* Parent face lists: global faces for all rows/cols
   * of this (rowNode, colNode) block.
   */
  BfSizeArray rowFaces_par;
  BfSizeArray colFaces_par;
  bfSizeArrayInitWithDefaultCapacity(&rowFaces_par);
  bfSizeArrayInitWithDefaultCapacity(&colFaces_par);

  getNodeInds(rowNode, qt, &rowFaces_par);
  getNodeInds(colNode, qt, &colFaces_par);

  BfSize mPar = bfSizeArrayGetSize(&rowFaces_par);
  BfSize nPar = bfSizeArrayGetSize(&colFaces_par);

  if (mPar == 0 || nPar == 0) {
    bfSizeArrayDeinit(&rowFaces_par);
    bfSizeArrayDeinit(&colFaces_par);
    return NULL;
  }

  /* OLD: raytrace this block once from the trimesh. */
  BfMatCsrReal *A_par =
    bfMatCsrRealNewViewFactorMatrixFromTrimesh(
        tm, &rowFaces_par, &colFaces_par);

  if (A_par == NULL) {
    bfSizeArrayDeinit(&rowFaces_par);
    bfSizeArrayDeinit(&colFaces_par);
    return NULL;
  }

  /* Reindex to fit bfMatCsrRealNewSubmatrixFromIndices’s assumption (gCol < nCols) */
  {
    BfSize nFaces = bfTrimeshGetNumFaces(tm);
    reindexCsrColsToLocal(A_par, &colFaces_par, nFaces);
  }

  /* NEW: optional heuristic — keep parent CSR as a single leaf
   * if we expect a hierarchy to be more expensive than the full block.
   */
#if BF_VF_HIER_ENABLE_PARENT_CSR_HEURISTIC
  {
    BfMat *A_base = bfMatCsrRealToMat(A_par);
    (void)A_base; /* not used except for sanity */

    BfSize const *rp_par = bfMatCsrRealGetRowptrConstPtr(A_par);
    BF_ASSERT(rp_par != NULL);
    BfSize nnz = rp_par[mPar];

    /* Parent CSR memory: data + colind + rowptr */
    double parentBytes =
      8.0 * (double)nnz +        /* data */
      8.0 * (double)nnz +        /* colind */
      8.0 * (double)(mPar + 1);  /* rowptr */

    /* Very crude upper bound on hierarchy cost: tile by leafMax */
    BfSize numBlocksRows = (mPar + leafMax - 1) / leafMax;
    BfSize numBlocksCols = (nPar + leafMax - 1) / leafMax;
    unsigned long long approxNumBlocks =
      (unsigned long long)numBlocksRows *
      (unsigned long long)numBlocksCols;

    /* Approximate child rowptr overhead: each block has ~leafMax rows */
    double approxChildBytes =
      8.0 * (double)nnz +                 /* data */
      8.0 * (double)nnz +                 /* colind */
      8.0 * (double)(approxNumBlocks * (unsigned long long)leafMax);

    if (approxChildBytes >= BF_VF_HIER_PARENT_CSR_FACTOR * parentBytes) {
      /* Keep parent as a single sparse leaf: move ownership of
       * rowFaces_par, colFaces_par, and A_par into the leaf.
       */
      BfVfHierBlock *block = bfVfHierBlockNew();
      block->kind = BF_VF_HIER_BLOCK_SPARSE;
      block->data.sparse.rowInds = rowFaces_par;
      block->data.sparse.colInds = colFaces_par;
      block->data.sparse.mat     = A_par;
      block->data.sparse.colsAreLocal = BF_TRUE;

      BfTreeNode *ni = bfQuadtreeNodeToTreeNode(rowNode);
      BfSize i0 = bfTreeNodeGetFirstIndex(ni);
      BfSize i1 = bfTreeNodeGetLastIndex(ni);

      block->data.sparse.row_i0 = i0;
      block->data.sparse.row_i1 = i1;

      return block;  /* NOTE: no deinit of rowFaces_par/colFaces_par/A_par here */
    }
  }
#endif /* BF_VF_HIER_ENABLE_PARENT_CSR_HEURISTIC */

  BfVfFaceMap faceMap;
  bfVfFaceMapInitFromParentFaces(&faceMap,
                                 (BfSizeArray const *)&rowFaces_par,
                                 (BfSizeArray const *)&colFaces_par);

  /* NEW: build the subtree purely from CSR slicing + SVD decisions. */
  BfVfHierBlock *subtree =
    buildBlockFromCsrMidlevel(
        (BfMatCsrReal const *)A_par,
        (BfSizeArray  const *)&rowFaces_par,
        (BfSizeArray  const *)&colFaces_par,
        (BfVfFaceMap  const *)&faceMap,
        rowNode,
        colNode,
        eta,
        leafMax,
        leafMin,
        minArea,              /* NEW */
        tol,
        minSvdSize,
        maxSvdRankFrac,
        0);  /* depth = 0 */

  /* A_par and parent face arrays are no longer needed:
   * all leaves now own their own CSR or SVD MatProduct blocks.
   */
  bfVfFaceMapDeinit(&faceMap);
  bfMatCsrRealDeinitAndDealloc(&A_par);
  bfSizeArrayDeinit(&rowFaces_par);
  bfSizeArrayDeinit(&colFaces_par);

  return subtree;
}


/* ============================================================
 * VfHier binary serialization
 * ============================================================ */

#define BF_VFHIER_MAGIC "BFVFHIER"
#define BF_VFHIER_MAGIC_LEN 7
#define BF_VFHIER_VERSION 2u

static BfBool write_bytes(FILE *fp, void const *buf, size_t n) {
  return fwrite(buf, 1, n, fp) == n;
}
static BfBool read_bytes(FILE *fp, void *buf, size_t n) {
  return fread(buf, 1, n, fp) == n;
}

static BfBool write_u8(FILE *fp, uint8_t v)   { return write_bytes(fp, &v, sizeof(v)); }
static BfBool write_u32(FILE *fp, uint32_t v) { return write_bytes(fp, &v, sizeof(v)); }
static BfBool write_u64(FILE *fp, uint64_t v) { return write_bytes(fp, &v, sizeof(v)); }

static BfBool read_u8(FILE *fp, uint8_t *v)   { return read_bytes(fp, v, sizeof(*v)); }
static BfBool read_u32(FILE *fp, uint32_t *v) { return read_bytes(fp, v, sizeof(*v)); }
static BfBool read_u64(FILE *fp, uint64_t *v) { return read_bytes(fp, v, sizeof(*v)); }

static BfBool write_size_array(FILE *fp, BfSizeArray const *a) {
  BfSize n = bfSizeArrayGetSize((BfSizeArray *)a);
  if (!write_u64(fp, (uint64_t)n)) return BF_FALSE;
  for (BfSize i = 0; i < n; ++i) {
    uint64_t v = (uint64_t)bfSizeArrayGet((BfSizeArray *)a, i);
    if (!write_u64(fp, v)) return BF_FALSE;
  }
  return BF_TRUE;
}

static BfBool read_size_array(FILE *fp, BfSizeArray *a) {
  uint64_t n64;
  if (!read_u64(fp, &n64)) return BF_FALSE;

  bfSizeArrayInitWithDefaultCapacity(a);
  for (uint64_t i = 0; i < n64; ++i) {
    uint64_t v64;
    if (!read_u64(fp, &v64)) return BF_FALSE;
    bfSizeArrayAppend(a, (BfSize)v64);
  }
  return BF_TRUE;
}

/* ---- CSR leaf I/O ---- */

static BfBool write_csr(FILE *fp, BfMatCsrReal const *Acsr) {
  BfMat const *A = bfMatCsrRealToMat((BfMatCsrReal *)Acsr);
  BfSize m = bfMatGetNumRows(A);
  BfSize n = bfMatGetNumCols(A);

  BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(Acsr);
  BfSize const *ci = bfMatCsrRealGetColindConstPtr(Acsr);
  BfReal const *da = bfMatCsrRealGetDataConstPtr(Acsr);
  if (rp == NULL || ci == NULL || da == NULL) return BF_FALSE;

  BfSize nnz = rp[m];

  if (!write_u64(fp, (uint64_t)m)) return BF_FALSE;
  if (!write_u64(fp, (uint64_t)n)) return BF_FALSE;
  if (!write_u64(fp, (uint64_t)nnz)) return BF_FALSE;

  /* rowptr: m+1 */
  for (BfSize i = 0; i < m + 1; ++i) {
    if (!write_u64(fp, (uint64_t)rp[i])) return BF_FALSE;
  }
  /* colind: nnz */
  for (BfSize k = 0; k < nnz; ++k) {
    if (!write_u64(fp, (uint64_t)ci[k])) return BF_FALSE;
  }
  /* data: nnz */
  if (!write_bytes(fp, da, (size_t)nnz * sizeof(BfReal))) return BF_FALSE;

  return BF_TRUE;
}

static BfMatCsrReal *read_csr(FILE *fp) {
  uint64_t m64, n64, nnz64;
  if (!read_u64(fp, &m64)) return NULL;
  if (!read_u64(fp, &n64)) return NULL;
  if (!read_u64(fp, &nnz64)) return NULL;

  BfSize m = (BfSize)m64;
  BfSize n = (BfSize)n64;
  BfSize nnz = (BfSize)nnz64;

  /* Build arrays (STEAL) */
  BfSizeArray *rowptr = bfSizeArrayNewWithDefaultCapacity();
  BfSizeArray *colind = bfSizeArrayNewWithDefaultCapacity();
  BfRealArray *data   = bfRealArrayNewWithDefaultCapacity();

  /* rowptr */
  for (BfSize i = 0; i < m + 1; ++i) {
    uint64_t v64;
    if (!read_u64(fp, &v64)) goto fail;
    bfSizeArrayAppend(rowptr, (BfSize)v64);
  }

  /* colind */
  for (BfSize k = 0; k < nnz; ++k) {
    uint64_t v64;
    if (!read_u64(fp, &v64)) goto fail;
    bfSizeArrayAppend(colind, (BfSize)v64);
  }

  /* data */
  for (BfSize k = 0; k < nnz; ++k) {
    BfReal v;
    if (!read_bytes(fp, &v, sizeof(BfReal))) goto fail;
    bfRealArrayAppend(data, v);
  }

  return bfMatCsrRealNewFromArrays(m, n, rowptr, colind, data, BF_POLICY_STEAL);

fail:
  bfSizeArrayDeinitAndDealloc(&rowptr);
  bfSizeArrayDeinitAndDealloc(&colind);
  bfRealArrayDeinitAndDealloc(&data);
  return NULL;
}

/* ---- SVD leaf I/O (assumes leaf->mat is MatProduct with 3 factors: U, S(diag), VT) ---- */

static BfBool unpack_matproduct_usvt(BfMat *P,
                                    BfMatDenseReal **U,
                                    BfMatDiagReal **S,
                                    BfMatDenseReal **VT)
{
  *U = NULL; *S = NULL; *VT = NULL;

  BfMatProduct *prod = bfMatToMatProduct(P);
  if (prod == NULL) return BF_FALSE;

  if (bfMatProductNumFactors(prod) != 3) return BF_FALSE;

  BfMat *f0 = bfMatProductGetFactor(prod, 0);
  BfMat *f1 = bfMatProductGetFactor(prod, 1);
  BfMat *f2 = bfMatProductGetFactor(prod, 2);

  BfMatDenseReal *Udr  = bfMatToMatDenseReal(f0);
  BfMatDiagReal  *Sdg  = bfMatToMatDiagReal(f1);
  BfMatDenseReal *VTdr = bfMatToMatDenseReal(f2);

  if (Udr == NULL || Sdg == NULL || VTdr == NULL) return BF_FALSE;

  *U = Udr; *S = Sdg; *VT = VTdr;
  return BF_TRUE;
}

static BfBool write_dense(FILE *fp, BfMatDenseReal const *M) {
  BfMat const *A = bfMatDenseRealToMat((BfMatDenseReal *)M);
  BfSize m = bfMatGetNumRows(A);
  BfSize n = bfMatGetNumCols(A);

  if (!write_u64(fp, (uint64_t)m)) return BF_FALSE;
  if (!write_u64(fp, (uint64_t)n)) return BF_FALSE;

  /* Assume contiguous storage in M->data as used elsewhere in your codebase */
  if (!write_bytes(fp, M->data, (size_t)m * (size_t)n * sizeof(BfReal))) return BF_FALSE;

  return BF_TRUE;
}

static BfMatDenseReal *read_dense(FILE *fp) {
  uint64_t m64, n64;
  if (!read_u64(fp, &m64)) return NULL;
  if (!read_u64(fp, &n64)) return NULL;

  BfSize m = (BfSize)m64;
  BfSize n = (BfSize)n64;

  BfMatDenseReal *M = bfMatDenseRealNew();
  if (M == NULL) return NULL;
  bfMatDenseRealInitZeros(M, m, n);

  if (!read_bytes(fp, M->data, (size_t)m * (size_t)n * sizeof(BfReal))) {
    bfMatDenseRealDeinitAndDealloc(&M);
    return NULL;
  }

  return M;
}

static BfBool write_diag(FILE *fp, BfMatDiagReal const *D) {
  BfMat const *A = bfMatDiagRealToMat((BfMatDiagReal *)D);
  BfSize m = bfMatGetNumRows(A);
  BfSize n = bfMatGetNumCols(A);
  if (m != n) return BF_FALSE;

  if (!write_u64(fp, (uint64_t)m)) return BF_FALSE;
  if (!write_bytes(fp, D->data, (size_t)m * sizeof(BfReal))) return BF_FALSE;

  return BF_TRUE;
}

static BfMatDiagReal *read_diag(FILE *fp) {
  uint64_t n64;
  if (!read_u64(fp, &n64)) return NULL;
  BfSize n = (BfSize)n64;

  BfMatDiagReal *D = bfMatDiagRealNew();
  if (D == NULL) return NULL;

  bfMatDiagRealInit(D, n, n);

  if (!read_bytes(fp, D->data, (size_t)n * sizeof(BfReal))) {
    bfMatDiagRealDeinitAndDealloc(&D);
    return NULL;
  }

  return D;
}

/* ---- Block I/O (recursive) ---- */

static BfBool write_block(FILE *fp, BfVfHierBlock const *block) {
  if (block == NULL) return BF_FALSE;

  if (!write_u8(fp, (uint8_t)block->kind)) return BF_FALSE;

  switch (block->kind) {
  case BF_VF_HIER_BLOCK_SPARSE: {
    BfVfSparseLeaf const *leaf = &block->data.sparse;

    /* Invariant: always local columns */
    if (!write_u8(fp, (uint8_t)1)) return BF_FALSE;
    BF_ASSERT(leaf->colsAreLocal);
    if (!write_size_array(fp, &leaf->rowInds)) return BF_FALSE;
    if (!write_size_array(fp, &leaf->colInds)) return BF_FALSE;
    if (!write_csr(fp, leaf->mat)) return BF_FALSE;

    if (!write_u64(fp, (uint64_t)leaf->row_i0)) return BF_FALSE;
    if (!write_u64(fp, (uint64_t)leaf->row_i1)) return BF_FALSE;
    if (!write_u64(fp, (uint64_t)leaf->col_j0)) return BF_FALSE;
    if (!write_u64(fp, (uint64_t)leaf->col_j1)) return BF_FALSE;

    return BF_TRUE;
  }

  case BF_VF_HIER_BLOCK_SVD: {
    BfVfSvdLeaf const *leaf = &block->data.svd;

    if (!write_size_array(fp, &leaf->rowInds)) return BF_FALSE;
    if (!write_size_array(fp, &leaf->colInds)) return BF_FALSE;
    if (!write_u64(fp, (uint64_t)leaf->rank)) return BF_FALSE;

    BfMatDenseReal *U = NULL, *VT = NULL;
    BfMatDiagReal *S = NULL;
    if (!unpack_matproduct_usvt(leaf->mat, &U, &S, &VT)) return BF_FALSE;

    if (!write_dense(fp, U)) return BF_FALSE;
    if (!write_diag(fp, S)) return BF_FALSE;
    if (!write_dense(fp, VT)) return BF_FALSE;

    if (!write_u64(fp, (uint64_t)leaf->row_i0)) return BF_FALSE;
    if (!write_u64(fp, (uint64_t)leaf->row_i1)) return BF_FALSE;
    if (!write_u64(fp, (uint64_t)leaf->col_j0)) return BF_FALSE;
    if (!write_u64(fp, (uint64_t)leaf->col_j1)) return BF_FALSE;

    return BF_TRUE;
  }

  case BF_VF_HIER_BLOCK_NODE: {
    BfPtrArray const *children = &block->data.node.children;
    BfSize n = bfPtrArraySize(children);

    if (!write_u64(fp, (uint64_t)n)) return BF_FALSE;
    for (BfSize i = 0; i < n; ++i)
      if (!write_block(fp, bfPtrArrayGet(children, i))) return BF_FALSE;

    return BF_TRUE;
  }

  default:
    return BF_FALSE;
  }
}

static BfVfHierBlock *read_block(FILE *fp) {
  uint8_t kind8;
  if (!read_u8(fp, &kind8)) return NULL;

  BfVfHierBlock *block = bfVfHierBlockNew();
  block->kind = (BfVfHierBlockKind)kind8;

  switch (block->kind) {
  case BF_VF_HIER_BLOCK_SPARSE: {
    uint8_t colsAreLocal8;
    if (!read_u8(fp, &colsAreLocal8)) goto fail;
    if (!read_size_array(fp, &block->data.sparse.rowInds)) goto fail;
    if (!read_size_array(fp, &block->data.sparse.colInds)) goto fail;

    block->data.sparse.colsAreLocal = (colsAreLocal8 != 0);

    block->data.sparse.mat = read_csr(fp);
    if (block->data.sparse.mat == NULL) goto fail;

    uint64_t i0_64, i1_64, j0_64, j1_64;
    if (!read_u64(fp, &i0_64)) goto fail;
    if (!read_u64(fp, &i1_64)) goto fail;
    block->data.sparse.row_i0 = (BfSize)i0_64;
    block->data.sparse.row_i1 = (BfSize)i1_64;
    if (!read_u64(fp, &j0_64)) goto fail;
    if (!read_u64(fp, &j1_64)) goto fail;
    block->data.sparse.col_j0 = (BfSize)j0_64;
    block->data.sparse.col_j1 = (BfSize)j1_64;

    return block;
  }

  case BF_VF_HIER_BLOCK_SVD: {
    if (!read_size_array(fp, &block->data.svd.rowInds)) goto fail;
    if (!read_size_array(fp, &block->data.svd.colInds)) goto fail;

    uint64_t rank64;
    if (!read_u64(fp, &rank64)) goto fail;
    block->data.svd.rank = (BfSize)rank64;

    BfMatDenseReal *U = read_dense(fp);
    BfMatDiagReal  *S = read_diag(fp);
    BfMatDenseReal *VT = read_dense(fp);
    if (U == NULL || S == NULL || VT == NULL) {
      if (U) bfMatDenseRealDeinitAndDealloc(&U);
      if (S) bfMatDiagRealDeinitAndDealloc(&S);
      if (VT) bfMatDenseRealDeinitAndDealloc(&VT);
      goto fail;
    }

    /* Rebuild MatProduct P = U * S * VT */
    BfMatProduct *Pprod = bfMatProductNew();
    bfMatProductInit(Pprod);
    bfMatProductPostMultiply(Pprod, bfMatDenseRealToMat(U));
    bfMatProductPostMultiply(Pprod, bfMatDiagRealToMat(S));
    bfMatProductPostMultiply(Pprod, bfMatDenseRealToMat(VT));

    block->data.svd.mat = bfMatProductToMat(Pprod);
    block->data.svd.work = NULL;
    block->data.svd.workLen = 0;

    uint64_t i0_64, i1_64, j0_64, j1_64;
    if (!read_u64(fp, &i0_64)) goto fail;
    if (!read_u64(fp, &i1_64)) goto fail;
    block->data.svd.row_i0 = (BfSize)i0_64;
    block->data.svd.row_i1 = (BfSize)i1_64;
    if (!read_u64(fp, &j0_64)) goto fail;
    if (!read_u64(fp, &j1_64)) goto fail;
    block->data.svd.col_j0 = (BfSize)j0_64;
    block->data.svd.col_j1 = (BfSize)j1_64;

    return block;
  }

  case BF_VF_HIER_BLOCK_NODE: {
    uint64_t nChildren64;
    if (!read_u64(fp, &nChildren64)) goto fail;

    bfInitPtrArray(&block->data.node.children, (BfSize)nChildren64);

    for (uint64_t i = 0; i < nChildren64; ++i) {
      BfVfHierBlock *child = read_block(fp);
      if (child != NULL)
        bfPtrArrayAppend(&block->data.node.children, child);
      else
        goto fail;
    }
    return block;
  }

  case BF_VF_HIER_BLOCK_NONE:
  default:
    goto fail;
  }

fail:
  bfVfHierBlockDeinitAndDealloc(&block);
  return NULL;
}

/* ---- Public API ---- */

BfBool bfVfHierSave(BfVfHier const *vfHier, char const *path) {
  if (vfHier == NULL || vfHier->root == NULL || path == NULL) return BF_FALSE;

  FILE *fp = fopen(path, "wb");
  if (fp == NULL) return BF_FALSE;

  BfBool ok = BF_TRUE;

  /* header */
  ok = ok && write_bytes(fp, BF_VFHIER_MAGIC, BF_VFHIER_MAGIC_LEN);
  ok = ok && write_u32(fp, (uint32_t)BF_VFHIER_VERSION);
  ok = ok && write_u64(fp, (uint64_t)vfHier->n);

  /* body */
  ok = ok && write_block(fp, vfHier->root);

  fclose(fp);
  return ok;
}

static void bfVfHierNormalizeCsrLeavesToLocal_(BfVfHierBlock *block, BfSize nFaces) {
  if (block == NULL) return;

  switch (block->kind) {
  case BF_VF_HIER_BLOCK_SPARSE: {
    BfVfSparseLeaf *leaf = &block->data.sparse;
    if (!leaf->colsAreLocal) {
      /* leaf->colInds is the global-face list defining the column ordering */
      reindexCsrColsToLocal(leaf->mat, &leaf->colInds, nFaces);
      leaf->colsAreLocal = BF_TRUE;
    }
  } break;

  case BF_VF_HIER_BLOCK_NODE: {
    BfPtrArray *children = &block->data.node.children;
    for (BfSize i = 0; i < bfPtrArraySize(children); ++i)
      bfVfHierNormalizeCsrLeavesToLocal_((BfVfHierBlock *)bfPtrArrayGet(children, i), nFaces);
  } break;

  case BF_VF_HIER_BLOCK_SVD:
  case BF_VF_HIER_BLOCK_NONE:
  default:
    break;
  }
}


BfVfHier *bfVfHierLoad(char const *path) {
  if (path == NULL) return NULL;

  FILE *fp = fopen(path, "rb");
  if (fp == NULL) return NULL;

  char magic[BF_VFHIER_MAGIC_LEN];
  uint32_t version;
  uint64_t n64;

  if (!read_bytes(fp, magic, BF_VFHIER_MAGIC_LEN)) { fclose(fp); return NULL; }
  if (memcmp(magic, BF_VFHIER_MAGIC, BF_VFHIER_MAGIC_LEN) != 0) { fclose(fp); return NULL; }

  if (!read_u32(fp, &version)) { fclose(fp); return NULL; }
  if (version != BF_VFHIER_VERSION) { fclose(fp); return NULL; }

  if (!read_u64(fp, &n64)) { fclose(fp); return NULL; }

  BfVfHier *vf = bfVfHierNew();
  vf->trimesh = NULL;
  vf->n = (BfSize)n64;

  vf->root = read_block(fp);
  fclose(fp);

  if (vf->root == NULL) {
    bfVfHierDeinitAndDealloc(&vf);
    return NULL;
  }

  /* Enforce invariant for any legacy files: CSR colind is local */
  bfVfHierNormalizeCsrLeavesToLocal_(vf->root, vf->n);

  return vf;
}

BfSize bfVfHierGetNumFaces(const BfVfHier *vfHier) {
  return vfHier->n;  // or whatever field stores the global size
}

static void bfVfSparseLeafApplyMany(BfVfSparseLeaf const *leaf,
                                    BfReal const         *X,
                                    BfSize                ldX,
                                    BfReal               *Y,
                                    BfSize                ldY,
                                    BfSize                n,
                                    BfSize                nrhs)
{
  BfMat const *A_mat = bfMatCsrRealToMat(leaf->mat);
  BfSize mA = bfMatGetNumRows(A_mat);
  BfSize nA = bfMatGetNumCols(A_mat);

  BF_ASSERT(mA == bfSizeArrayGetSize((BfSizeArray *)&leaf->rowInds));
  BF_ASSERT(nA == bfSizeArrayGetSize((BfSizeArray *)&leaf->colInds));

  BfSize const *rowptr = bfMatCsrRealGetRowptrConstPtr(leaf->mat);
  BfSize const *colind = bfMatCsrRealGetColindConstPtr(leaf->mat);
  BfReal const *data   = bfMatCsrRealGetDataConstPtr(leaf->mat);

  BF_ASSERT(rowptr && colind && data);

  BfSizeArray const *rowInds = &leaf->rowInds;
  BfSizeArray const *colInds = &leaf->colInds;

  BF_ASSERT(leaf->colsAreLocal);

  for (BfSize i = 0; i < mA; ++i) {
    BfSize row_start = rowptr[i];
    BfSize row_end   = rowptr[i + 1];

    BfSize globalRow = bfSizeArrayGet((BfSizeArray *)rowInds, i);
    BF_ASSERT(globalRow < n);

    BfReal *yRow = Y + globalRow;

    for (BfSize k = row_start; k < row_end; ++k) {
      BfSize cLocal = colind[k];
      BF_ASSERT(cLocal < bfSizeArrayGetSize((BfSizeArray *)colInds));

      BfSize globalCol = bfSizeArrayGet((BfSizeArray *)colInds, cLocal);
      BF_ASSERT(globalCol < n);

      BfReal a = data[k];
      BfReal const *xCol = X + globalCol;

      for (BfSize r = 0; r < nrhs; ++r)
        yRow[r*ldY] += a * xCol[r*ldX];
    }
  }
}

static void bfVfSvdLeafApplyMany(BfVfSvdLeaf const *leaf,
                                BfReal const      *X,
                                BfSize             ldX,
                                BfReal            *Y,
                                BfSize             ldY,
                                BfSize             n,
                                BfSize             nrhs)
{
  for (BfSize r = 0; r < nrhs; ++r) {
    /* Each RHS is contiguous if X/Y are column-major with ldX/ldY */
    bfVfSvdLeafApply(leaf, X + r*ldX, Y + r*ldY, n);
  }
}

static void bfVfHierBlockApplyMany(BfVfHierBlock const *block,
                                   BfReal const        *X,
                                   BfSize               ldX,
                                   BfReal              *Y,
                                   BfSize               ldY,
                                   BfSize               n,
                                   BfSize               nrhs)
{
  if (block == NULL) return;

  switch (block->kind) {
  case BF_VF_HIER_BLOCK_SPARSE:
    bfVfSparseLeafApplyMany(&block->data.sparse, X, ldX, Y, ldY, n, nrhs);
    break;

  case BF_VF_HIER_BLOCK_SVD:
    bfVfSvdLeafApplyMany(&block->data.svd, X, ldX, Y, ldY, n, nrhs);
    break;

  case BF_VF_HIER_BLOCK_NODE:
    for (BfSize i = 0; i < bfPtrArraySize(&block->data.node.children); ++i) {
      BfVfHierBlock *child = bfPtrArrayGet(&block->data.node.children, i);
      bfVfHierBlockApplyMany(child, X, ldX, Y, ldY, n, nrhs);
    }
    break;

  case BF_VF_HIER_BLOCK_NONE:
  default:
    break;
  }
}

static void collect_leaf_blocks(BfVfHierBlock const *block, BfPtrArray *out) {
  if (!block) return;
  switch (block->kind) {
  case BF_VF_HIER_BLOCK_SPARSE:
  case BF_VF_HIER_BLOCK_SVD:
    bfPtrArrayAppend(out, (void *)block);
    return;
  case BF_VF_HIER_BLOCK_NODE:
    for (BfSize i = 0; i < bfPtrArraySize(&block->data.node.children); ++i) {
      BfVfHierBlock *child = bfPtrArrayGet(&block->data.node.children, i);
      collect_leaf_blocks(child, out);
    }
    return;
  default:
    return;
  }
}

/* Count leaf blocks (sparse+svd) without allocating a BfPtrArray.
 *
 * NOTE: bfInitPtrArray(&arr, 0) is *not* valid in this codebase because
 * extendPtrArray() asserts new_capacity > arr->capacity and doubling 0 stays 0.
 */
static BfSize count_leaf_blocks(BfVfHierBlock const *block) {
  if (!block) return 0;

  switch (block->kind) {
  case BF_VF_HIER_BLOCK_SPARSE:
  case BF_VF_HIER_BLOCK_SVD:
    return 1;

  case BF_VF_HIER_BLOCK_NODE: {
    BfSize cnt = 0;
    for (BfSize i = 0; i < bfPtrArraySize(&block->data.node.children); ++i) {
      BfVfHierBlock *child = bfPtrArrayGet(&block->data.node.children, i);
      cnt += count_leaf_blocks(child);
    }
    return cnt;
  }

  default:
    return 0;
  }
}

/* -------- leaf block dump (for plotting/debug) -------- */
BfSize bfVfHierGetNumLeafBlocks(BfVfHier const *vfHier) {
  BF_ASSERT(vfHier != NULL);
  BF_ASSERT(vfHier->root != NULL);

  /* Count directly (no temporary ptr-array). */
  return count_leaf_blocks(vfHier->root);
}

void bfVfHierDumpLeafBlocks(
    BfVfHier const *vfHier,
    BfSize *row_i0, BfSize *row_i1,
    BfSize *col_j0, BfSize *col_j1,
    uint8_t *kind,
    BfSize *rank,
    unsigned long long *nnz
) {
  BF_ASSERT(vfHier != NULL);
  BF_ASSERT(vfHier->root != NULL);
  BF_ASSERT(row_i0 != NULL && row_i1 != NULL);
  BF_ASSERT(col_j0 != NULL && col_j1 != NULL);
  BF_ASSERT(kind != NULL);
  /* rank and nnz may be NULL */

  BfPtrArray leaves;
  /* Pre-size to avoid repeated reallocations and avoid invalid cap=0. */
  BfSize n_est = count_leaf_blocks(vfHier->root);
  bfInitPtrArray(&leaves, n_est > 0 ? n_est : 16);

  collect_leaf_blocks(vfHier->root, &leaves);

  BfSize n = (BfSize)bfPtrArraySize(&leaves);

  for (BfSize k = 0; k < n; ++k) {
    BfVfHierBlock const *leaf = (BfVfHierBlock const *)bfPtrArrayGet(&leaves, k);

    /* leaf must be SPARSE or SVD */
    if (leaf->kind == BF_VF_HIER_BLOCK_SPARSE) {
      row_i0[k] = leaf->data.sparse.row_i0;
      row_i1[k] = leaf->data.sparse.row_i1;
      col_j0[k] = leaf->data.sparse.col_j0;
      col_j1[k] = leaf->data.sparse.col_j1;
      kind[k]   = (uint8_t)0;
      if (rank) rank[k] = (BfSize)0;

      if (nnz) {
        /* CSR nnz = rowptr[m] where m = num rows */
        BfMatCsrReal const *Acsr = leaf->data.sparse.mat;
        if (Acsr) {
            BfSize m = row_i1[k] - row_i0[k];
            BfSize const *rp = bfMatCsrRealGetRowptrConstPtr((BfMatCsrReal *)Acsr);
            nnz[k] = (unsigned long long)rp[m];
        } else {
          nnz[k] = 0ull;
        }
      }

    } else if (leaf->kind == BF_VF_HIER_BLOCK_SVD) {
      row_i0[k] = leaf->data.svd.row_i0;
      row_i1[k] = leaf->data.svd.row_i1;
      col_j0[k] = leaf->data.svd.col_j0;
      col_j1[k] = leaf->data.svd.col_j1;
      kind[k]   = (uint8_t)1;
      if (rank) rank[k] = leaf->data.svd.rank;

      /* No CSR here; nnz is undefined for SVD leaves. */
      if (nnz) nnz[k] = 0ull;

    } else {
      /* should never happen: collect_leaf_blocks only returns leaves */
      BF_ASSERT(false);
      row_i0[k] = row_i1[k] = col_j0[k] = col_j1[k] = 0;
      kind[k] = (uint8_t)255;
      if (rank) rank[k] = 0;
      if (nnz) nnz[k] = 0ull;
    }
  } /* end for (k) */
  bfPtrArrayDeinit(&leaves);
}


static BfSize get_leaf_row_i0(BfVfHierBlock const *b) {
  if (b->kind == BF_VF_HIER_BLOCK_SPARSE) return b->data.sparse.row_i0;
  if (b->kind == BF_VF_HIER_BLOCK_SVD)    return b->data.svd.row_i0;
  return 0;
}

static BfSize get_leaf_row_i1(BfVfHierBlock const *b) {
  if (b->kind == BF_VF_HIER_BLOCK_SPARSE) return b->data.sparse.row_i1;
  if (b->kind == BF_VF_HIER_BLOCK_SVD)    return b->data.svd.row_i1;
  return 0;
}

static BfSize get_leaf_col_j0(BfVfHierBlock const *leaf) {
  if (leaf->kind == BF_VF_HIER_BLOCK_SPARSE) return leaf->data.sparse.col_j0;
  if (leaf->kind == BF_VF_HIER_BLOCK_SVD)    return leaf->data.svd.col_j0;
  if (leaf->kind == BF_VF_HIER_BLOCK_NONE)   return BF_SIZE_BAD_VALUE;
  BF_DIE();
}

static BfSize get_leaf_col_j1(BfVfHierBlock const *leaf) {
  if (leaf->kind == BF_VF_HIER_BLOCK_SPARSE) return leaf->data.sparse.col_j1;
  if (leaf->kind == BF_VF_HIER_BLOCK_SVD)    return leaf->data.svd.col_j1;
  if (leaf->kind == BF_VF_HIER_BLOCK_NONE)   return BF_SIZE_BAD_VALUE;
  BF_DIE();
}

static uint8_t classify_leaf_kind(BfVfHierBlock const *leaf) {
  switch (leaf->kind) {
  case BF_VF_HIER_BLOCK_SPARSE: return 0;
  case BF_VF_HIER_BLOCK_SVD:    return 1;
  case BF_VF_HIER_BLOCK_NONE:   return 2;
  default:                      return 3;
  }
}

static BfSize get_leaf_rank(BfVfHierBlock const *leaf) {
  if (leaf->kind == BF_VF_HIER_BLOCK_SVD) return leaf->data.svd.rank;
  return 0;
}

static BfSize get_leaf_num_rows(BfVfHierBlock const *b) {
  if (b->kind == BF_VF_HIER_BLOCK_SPARSE)
    return bfSizeArrayGetSize(&b->data.sparse.rowInds);
  if (b->kind == BF_VF_HIER_BLOCK_SVD)
    return bfSizeArrayGetSize(&b->data.svd.rowInds);
  return 0;
}

static BfSize find_svd_index(BfVfHierBlock const *b,
                             BfVfHierBlock const *const *svdBlocks,
                             BfSize numSvdBlocks)
{
  for (BfSize i = 0; i < numSvdBlocks; ++i)
    if (svdBlocks[i] == b) return i;
  return BF_SIZE_BAD_VALUE;
}

/* build (or rebuild) vfHier->applyPlan */
static void bfVfHierBuildApplyPlan(BfVfHier *vfHier, BfSize tileSize) {
  BF_ASSERT(vfHier && vfHier->root);
  if (tileSize == 0) tileSize = 1024; /* safe default */

  /* free old plan if present */
  if (vfHier->applyPlan) {
    BfVfApplyPlan *pOld = vfHier->applyPlan;
    if (pOld->tilePtr)   bfMemFree(pOld->tilePtr);
    if (pOld->tasks)     bfMemFree(pOld->tasks);
    if (pOld->svdBlocks) bfMemFree(pOld->svdBlocks);
    bfMemFree(pOld);
    vfHier->applyPlan = NULL;
  }

  /* collect leaf blocks */
  BfPtrArray leaves;
  bfInitPtrArray(&leaves, 1024);
  collect_leaf_blocks(vfHier->root, &leaves);

  BfSize n = vfHier->n;
  BfSize numTiles = (n + tileSize - 1)/tileSize;

  /* count tasks per tile */
  BfSize *counts = bfMemAlloc(numTiles, sizeof(BfSize));
  for (BfSize t = 0; t < numTiles; ++t) counts[t] = 0;

  /* collect SVD blocks */
  BfPtrArray svdBlocksTmp;
  bfInitPtrArray(&svdBlocksTmp, 256);

  for (BfSize li = 0; li < bfPtrArraySize(&leaves); ++li) {
    BfVfHierBlock const *b = bfPtrArrayGet(&leaves, li);

    if (b->kind == BF_VF_HIER_BLOCK_SVD)
      bfPtrArrayAppend(&svdBlocksTmp, (void *)b);

    BfSize i0 = get_leaf_row_i0(b);
    BfSize i1 = get_leaf_row_i1(b);
    BF_ASSERT(i1 >= i0);

    BfSize mLeaf = get_leaf_num_rows(b);
    BF_ASSERT(i1 - i0 == mLeaf);

    if (mLeaf == 0) continue;

    BfSize t0 = i0 / tileSize;
    BfSize t1 = (i1 - 1) / tileSize;

    for (BfSize t = t0; t <= t1; ++t)
      counts[t] += 1;
  }

  /* prefix sum -> tilePtr */
  BfSize *tilePtr = bfMemAlloc(numTiles + 1, sizeof(BfSize));
  tilePtr[0] = 0;
  for (BfSize t = 0; t < numTiles; ++t)
    tilePtr[t + 1] = tilePtr[t] + counts[t];

  BfSize numTasks = tilePtr[numTiles];
  BfVfApplyTask *tasks = bfMemAlloc(numTasks, sizeof(BfVfApplyTask));

  /* cursors per tile */
  BfSize *cursor = bfMemAlloc(numTiles, sizeof(BfSize));
  for (BfSize t = 0; t < numTiles; ++t)
    cursor[t] = tilePtr[t];

  /* allocate/fill plan NOW so we can refer to p->svdBlocks while filling tasks */
  BfVfApplyPlan *p = bfMemAlloc(1, sizeof(BfVfApplyPlan));
  memset(p, 0, sizeof(*p));
  p->tileSize = tileSize;
  p->numTiles = numTiles;
  p->tilePtr  = tilePtr;
  p->tasks    = tasks;
  p->numTasks = numTasks;

  p->numSvdBlocks = bfPtrArraySize(&svdBlocksTmp);
  if (p->numSvdBlocks > 0) {
    p->svdBlocks = bfMemAlloc(p->numSvdBlocks, sizeof(BfVfHierBlock const *));
    for (BfSize i = 0; i < p->numSvdBlocks; ++i)
      p->svdBlocks[i] = bfPtrArrayGet(&svdBlocksTmp, i);
  } else {
    p->svdBlocks = NULL;
  }

  /* fill tasks */
  for (BfSize li = 0; li < bfPtrArraySize(&leaves); ++li) {
    BfVfHierBlock const *b = bfPtrArrayGet(&leaves, li);

    BfSize i0 = get_leaf_row_i0(b);
    BfSize i1 = get_leaf_row_i1(b);

    BfSize mLeaf = get_leaf_num_rows(b);
    BF_ASSERT(i1 - i0 == mLeaf);

    if (mLeaf == 0) continue;

    /* If this is an SVD leaf, find its svdIndex ONCE (not per-tile) */
    BfSize svdIndex = BF_SIZE_BAD_VALUE;
    if (b->kind == BF_VF_HIER_BLOCK_SVD) {
      for (BfSize si = 0; si < p->numSvdBlocks; ++si) {
        if (p->svdBlocks[si] == b) { svdIndex = si; break; }
      }
      BF_ASSERT(svdIndex != BF_SIZE_BAD_VALUE);
    }

    BfSize t0 = i0 / tileSize;
    BfSize t1 = (i1 - 1) / tileSize;

    for (BfSize t = t0; t <= t1; ++t) {
      BfSize tile_i0 = t * tileSize;
      BfSize tile_i1 = (t + 1) * tileSize;
      if (tile_i1 > n) tile_i1 = n;

      BfSize inter0 = i0 > tile_i0 ? i0 : tile_i0;
      BfSize inter1 = i1 < tile_i1 ? i1 : tile_i1;
      BF_ASSERT(inter1 >= inter0);

      /* IMPORTANT: iBegin/iEnd are local row indices inside this leaf:
         local = permIndex - leaf->row_i0 */
      BfSize local0 = inter0 - i0;
      BfSize local1 = inter1 - i0;
      BF_ASSERT(local1 <= mLeaf);

      BfSize pos = cursor[t]++;
      tasks[pos].leafBlock = b;
      tasks[pos].iBegin = local0;
      tasks[pos].iEnd   = local1;
      tasks[pos].svdIndex = (b->kind == BF_VF_HIER_BLOCK_SVD) ? svdIndex : BF_SIZE_BAD_VALUE;
    }
  }

  /* sanity: did we fill exactly the computed ranges? */
  for (BfSize t = 0; t < numTiles; ++t)
    BF_ASSERT(cursor[t] == tilePtr[t + 1]);

  bfMemFree(counts);
  bfMemFree(cursor);
  bfPtrArrayDeinit(&leaves);
  bfPtrArrayDeinit(&svdBlocksTmp);

  vfHier->applyPlan = p;
  vfHier->applyTileSize = tileSize;
}

static void bfVfSparseLeafApplyRange(BfVfSparseLeaf const *leaf,
                                     BfReal const         *x,
                                     BfReal               *y,
                                     BfSize                n,
                                     BfSize                iBegin,
                                     BfSize                iEnd)
{
  BfMat const *A_mat = bfMatCsrRealToMat(leaf->mat);
  BfSize mA = bfMatGetNumRows(A_mat);
  BfSize nA = bfMatGetNumCols(A_mat);

  BF_ASSERT(iBegin <= iEnd);
  BF_ASSERT(iEnd <= mA);

  BfSize const *rowptr = bfMatCsrRealGetRowptrConstPtr(leaf->mat);
  BfSize const *colind = bfMatCsrRealGetColindConstPtr(leaf->mat);
  BfReal const *data   = bfMatCsrRealGetDataConstPtr(leaf->mat);
  BF_ASSERT(rowptr && colind && data);

  BfSizeArray const *rowInds = &leaf->rowInds;
  BfSizeArray const *colInds = &leaf->colInds;

  if (leaf->colsAreLocal) {
    for (BfSize i = iBegin; i < iEnd; ++i) {
      BfSize r0 = rowptr[i];
      BfSize r1 = rowptr[i + 1];

      BfReal acc = 0;
      for (BfSize k = r0; k < r1; ++k) {
        BfSize cLocal = colind[k];
        BF_ASSERT(cLocal < nA);

        BfSize gCol = bfSizeArrayGet((BfSizeArray *)colInds, cLocal);
        BF_ASSERT(gCol < n);

        acc += data[k] * x[gCol];
      }

      BfSize gRow = bfSizeArrayGet((BfSizeArray *)rowInds, i);
      BF_ASSERT(gRow < n);
      y[gRow] += acc;
    }
  } else {
    for (BfSize i = iBegin; i < iEnd; ++i) {
      BfSize r0 = rowptr[i];
      BfSize r1 = rowptr[i + 1];

      BfReal acc = 0;
      for (BfSize k = r0; k < r1; ++k) {
        BfSize gCol = colind[k];
        BF_ASSERT(gCol < n);
        acc += data[k] * x[gCol];
      }

      BfSize gRow = bfSizeArrayGet((BfSizeArray *)rowInds, i);
      BF_ASSERT(gRow < n);
      y[gRow] += acc;
    }
  }
}


static void bfVfSparseLeafApplyManyRange(BfVfSparseLeaf const *leaf,
                                        BfReal const         *X, BfSize ldX,
                                        BfReal               *Y, BfSize ldY,
                                        BfSize                n,
                                        BfSize                nrhs,
                                        BfSize                iBegin,
                                        BfSize                iEnd)
{
  BfMat const *A_mat = bfMatCsrRealToMat(leaf->mat);
  BfSize mA = bfMatGetNumRows(A_mat);
  BF_ASSERT(iEnd <= mA);

  BfSize const *rowptr = bfMatCsrRealGetRowptrConstPtr(leaf->mat);
  BfSize const *colind = bfMatCsrRealGetColindConstPtr(leaf->mat);
  BfReal const *data   = bfMatCsrRealGetDataConstPtr(leaf->mat);
  BF_ASSERT(rowptr && colind && data);

  BfSizeArray const *rowInds = &leaf->rowInds;
  BfSizeArray const *colInds = &leaf->colInds;

  if (leaf->colsAreLocal) {
    for (BfSize i = iBegin; i < iEnd; ++i) {
      BfSize globalRow = bfSizeArrayGet((BfSizeArray *)rowInds, i);
      BfReal *yRow = Y + globalRow;

      BfSize rs = rowptr[i], re = rowptr[i + 1];
      for (BfSize k = rs; k < re; ++k) {
        BfSize cLocal = colind[k];
        BfSize globalCol = bfSizeArrayGet((BfSizeArray *)colInds, cLocal);
        BfReal a = data[k];

        BfReal const *xCol = X + globalCol;
        for (BfSize r = 0; r < nrhs; ++r)
          yRow[r*ldY] += a * xCol[r*ldX];
      }
    }
  } else {
    for (BfSize i = iBegin; i < iEnd; ++i) {
      BfSize globalRow = bfSizeArrayGet((BfSizeArray *)rowInds, i);
      BfReal *yRow = Y + globalRow;

      BfSize rs = rowptr[i], re = rowptr[i + 1];
      for (BfSize k = rs; k < re; ++k) {
        BfSize globalCol = colind[k];
        BfReal a = data[k];

        BfReal const *xCol = X + globalCol;
        for (BfSize r = 0; r < nrhs; ++r)
          yRow[r*ldY] += a * xCol[r*ldX];
      }
    }
  }
}

static void ensure_svd_cache(BfVfSvdLeaf const *leaf) {
  if (leaf->U_cache && leaf->S_cache && leaf->VT_cache) return;

  #pragma omp critical(bf_vf_svd_cache_init)
  {
    if (!(leaf->U_cache && leaf->S_cache && leaf->VT_cache)) {
      /* cast away const: cache pointers only */
      BfVfSvdLeaf *m = (BfVfSvdLeaf *)leaf;
      BfMatDenseReal *U = NULL, *VT = NULL;
      BfMatDiagReal *S = NULL;
      BF_ASSERT(unpack_matproduct_usvt(leaf->mat, &U, &S, &VT));

      m->U_cache  = U;
      m->S_cache  = S;
      m->VT_cache = VT;
    }
  }
}

/* Z layout: contiguous [nrhs][rank] for this leaf: Z[rhs*rank + k] */
static void svd_precompute_Z(BfVfSvdLeaf const *leaf,
                            BfReal const      *X, BfSize ldX,
                            BfSize             n,
                            BfSize             nrhs,
                            BfReal            *Z /* len = nrhs*rank */)
{
  ensure_svd_cache(leaf);

  BfSize mj   = bfSizeArrayGetSize((BfSizeArray *)&leaf->colInds);
  BfSize rank = leaf->rank;

  BfMatDenseReal const *VTm = leaf->VT_cache; /* rank x mj */
  BfMatDiagReal  const *Sm  = leaf->S_cache;

  BfReal const *VT = VTm->data;
  BfSize rsVT = VTm->super.rowStride;
  BfSize csVT = VTm->super.colStride;

  BfReal const *S = Sm->data; /* length rank */

  /* zero */
  for (BfSize r = 0; r < nrhs*rank; ++r) Z[r] = 0;

  for (BfSize j = 0; j < mj; ++j) {
    BfSize globalCol = bfSizeArrayGet((BfSizeArray *)&leaf->colInds, j);
    BF_ASSERT(globalCol < n);

    BfReal const *xCol = X + globalCol; /* xCol[rhs*ldX] */

    for (BfSize k = 0; k < rank; ++k) {
      BfReal vt = VT[k*rsVT + j*csVT];
      for (BfSize rhs = 0; rhs < nrhs; ++rhs)
        Z[rhs*rank + k] += vt * xCol[rhs*ldX];
    }
  }

  /* scale by S */
  for (BfSize rhs = 0; rhs < nrhs; ++rhs)
    for (BfSize k = 0; k < rank; ++k)
      Z[rhs*rank + k] *= S[k];
}

/* accumulate only rows [iBegin,iEnd) */
static void svd_accumulate_rows_from_Z(BfVfSvdLeaf const *leaf,
                                      BfReal const      *Z, /* [nrhs][rank] */
                                      BfReal            *Y, BfSize ldY,
                                      BfSize             n,
                                      BfSize             nrhs,
                                      BfSize             iBegin,
                                      BfSize             iEnd)
{
  ensure_svd_cache(leaf);

  BfSize m    = bfSizeArrayGetSize((BfSizeArray *)&leaf->rowInds);
  BfSize rank = leaf->rank;
  BF_ASSERT(iEnd <= m);

  BfMatDenseReal const *Um = leaf->U_cache; /* m x rank */
  BfReal const *U = Um->data;
  BfSize rsU = Um->super.rowStride;
  BfSize csU = Um->super.colStride;

  for (BfSize i = iBegin; i < iEnd; ++i) {
    BfSize globalRow = bfSizeArrayGet((BfSizeArray *)&leaf->rowInds, i);
    BF_ASSERT(globalRow < n);

    BfReal *yRow = Y + globalRow;

    for (BfSize rhs = 0; rhs < nrhs; ++rhs) {
      BfReal const *z = Z + rhs*rank;
      BfReal acc = 0;
      for (BfSize k = 0; k < rank; ++k)
        acc += U[i*rsU + k*csU] * z[k];

      yRow[rhs*ldY] += acc;
    }
  }
}

void bfVfHierApplyMany(BfVfHier const *vfHier,
                       BfReal const   *X,   BfSize ldX,
                       BfReal         *Y,   BfSize ldY,
                       BfSize          nrhs)
{
  BF_ASSERT(vfHier && vfHier->root);
  BF_ASSERT(X && Y);
  BF_ASSERT(ldX >= vfHier->n);
  BF_ASSERT(ldY >= vfHier->n);

  /* build plan lazily if needed */
  if (vfHier->applyPlan == NULL) {
    /* cast away const: cached plan is immutable afterwards */
    bfVfHierBuildApplyPlan((BfVfHier *)vfHier, 1024);
  }

  BfVfApplyPlan const *p = vfHier->applyPlan;
  BfSize n = vfHier->n;

  /* -------- phase 1: precompute Z for every SVD leaf (thread-safe) -------- */
  BfReal **Zptr = NULL;
  BfReal  *Zbuf = NULL;

  if (p->numSvdBlocks > 0) {
    Zptr = bfMemAlloc(p->numSvdBlocks, sizeof(BfReal *));
    /* compute total storage = sum(rank_i * nrhs) */
    BfSize total = 0;
    for (BfSize i = 0; i < p->numSvdBlocks; ++i) {
      BfVfSvdLeaf const *leaf = &p->svdBlocks[i]->data.svd;
      total += leaf->rank * nrhs;
    }
    Zbuf = bfMemAlloc(total, sizeof(BfReal));

    /* assign pointers */
    BfSize off = 0;
    for (BfSize i = 0; i < p->numSvdBlocks; ++i) {
      BfVfSvdLeaf const *leaf = &p->svdBlocks[i]->data.svd;
      Zptr[i] = Zbuf + off;
      off += leaf->rank * nrhs;
    }

    #pragma omp parallel for schedule(dynamic) if(!omp_in_parallel())
    for (BfSize i = 0; i < p->numSvdBlocks; ++i) {
      BfVfSvdLeaf const *leaf = &p->svdBlocks[i]->data.svd;
      svd_precompute_Z(leaf, X, ldX, n, nrhs, Zptr[i]);
    }
  }

  /* -------- phase 2: parallel over row tiles (only tile thread writes Y rows) -------- */
  #pragma omp parallel for schedule(dynamic) if(!omp_in_parallel())
  for (BfSize t = 0; t < p->numTiles; ++t) {
    BfSize start = p->tilePtr[t];
    BfSize end   = p->tilePtr[t + 1];

    for (BfSize q = start; q < end; ++q) {
      BfVfApplyTask const *task = &p->tasks[q];
      BfVfHierBlock const *b = task->leafBlock;

      if (b->kind == BF_VF_HIER_BLOCK_SPARSE) {
        bfVfSparseLeafApplyManyRange(&b->data.sparse, X, ldX, Y, ldY,
                                     n, nrhs, task->iBegin, task->iEnd);
      } else if (b->kind == BF_VF_HIER_BLOCK_SVD) {
        BfSize sid = task->svdIndex;
        BF_ASSERT(sid != BF_SIZE_BAD_VALUE);
        svd_accumulate_rows_from_Z(&b->data.svd, Zptr[sid], Y, ldY,
                                   n, nrhs, task->iBegin, task->iEnd);
      }
    }
  }

  if (Zptr) bfMemFree(Zptr);
  if (Zbuf) bfMemFree(Zbuf);
}

