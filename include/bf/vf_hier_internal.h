#pragma once

#include <bf/def.h>
#include <bf/vf_hier.h>

#include <bf/size_array.h>
#include <stdbool.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <bf/ptr_array.h>

/* Forward declarations */
typedef struct BfQuadtreeNode BfQuadtreeNode;
typedef struct BfQuadtree     BfQuadtree;
typedef struct BfTrimesh      BfTrimesh;
typedef struct BfMatCsrReal   BfMatCsrReal;

typedef struct {
  BfSize *globalToRow;  /* size = mapSize, maps global face -> parent row index */
  BfSize *globalToCol;  /* size = mapSize, maps global face -> parent col index */
  BfSize  mapSize;      /* == maxFace + 1 in parent block */
} BfVfFaceMap;

typedef struct {
  BfSize i0, i1;
  BfSize j0, j1;
  BfSize mi, mj;
  bool leafI, leafJ;
  bool small;
  bool empty;
} BfVfBlockMeta;

//=====

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
extern double g_vf_csr_to_dense_time;
extern double g_vf_svd_time          ;
extern double g_vf_csr_slice_time    ;  /* bfMatCsrRealNewSubmatrixFromIndices */
extern double g_vf_leaf_map_time     ;  /* build childRowFaces + global→parent maps */
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

/* =========================
 * Internal shared helpers
 * ========================= */

void bfVfFaceMapInitFromParentFaces(
    BfVfFaceMap      *map,
    BfSizeArray const *rowFaces_par,
    BfSizeArray const *colFaces_par);

void bfVfFaceMapDeinit(BfVfFaceMap *map);

BfVfBlockMeta getBlockMeta(BfQuadtreeNode *rowNode,
                           BfQuadtreeNode *colNode,
                           BfSize leafMax);

typedef struct BfVfApplyTask {
  BfVfHierBlock const *leafBlock;
  BfSize iBegin, iEnd;
  BfSize svdIndex;
} BfVfApplyTask;

struct BfVfApplyPlan {
  BfSize tileSize;
  BfSize numTiles;
  BfSize *tilePtr;
  BfVfApplyTask *tasks;
  BfSize numTasks;

  BfVfHierBlock const **svdBlocks;
  BfSize numSvdBlocks;
};
typedef struct BfVfApplyPlan BfVfApplyPlan;


void bfVfHierBuildApplyPlan(BfVfHier *vfHier, BfSize tileSize);

static inline BfReal bf_local_max(BfReal a, BfReal b) { return a > b ? a : b; }
static inline BfReal bf_local_sqrt(BfReal x) { return x <= 0 ? 0 : (BfReal)sqrt((double)x); }

//BfVfBlockMeta getBlockMeta(BfQuadtreeNode *rowNode,
//                                  BfQuadtreeNode *colNode,
//                                  BfSize          leafMax);

BfVfHierBlock *bfVfHierBlockNew(void);
void           bfVfHierBlockDeinit(BfVfHierBlock *block);
void           bfVfHierBlockDealloc(BfVfHierBlock **blockPtr);
void           bfVfHierBlockDeinitAndDealloc(BfVfHierBlock **blockPtr); /* NEW */

typedef struct { BfSize row; BfSize col; } ChildPair;

extern unsigned long long g_svd_tries_total;
extern unsigned long long g_svd_tries_done;

void vfHierInitProgressFromEnv_(void);
void vfHierMaybePrintProgress_(unsigned long long done, unsigned long long total);

BfBool isFar(BfQuadtreeNode const *rowNode, BfQuadtreeNode const *colNode, BfReal eta);
void getNodeInds(BfQuadtreeNode *node, BfQuadtree const *qt, BfSizeArray *inds);

BfVfHierBlock *buildBlockHybrid(/* full args */);
BfVfHierBlock *buildBlockFromCsrMidlevel(/* full args */);
void countSvdTriesFromCsrMidlevel(/* full args */);

void bfVfSvdLeafApply(BfVfSvdLeaf const *leaf, BfReal const *x, BfReal *y, BfSize n);
void bfVfSvdLeafApplyMany(BfVfSvdLeaf const *leaf, BfReal const *X, BfSize ldX,
                          BfReal *Y, BfSize ldY, BfSize n, BfSize r);

void reindexCsrColsToLocal(BfMatCsrReal *A, BfSizeArray const *colFaces, BfSize nFaces);
void collect_leaf_blocks(BfVfHierBlock const *root, BfPtrArray *leaves);

double bfVfHierBlockMemBytes(BfVfHierBlock const *block);

BfSize get_leaf_num_rows(BfVfHierBlock const *b);
BfSize get_leaf_row_i0(BfVfHierBlock const *b);
BfSize get_leaf_row_i1(BfVfHierBlock const *b);