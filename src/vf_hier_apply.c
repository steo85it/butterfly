/*
 * vf_hier_apply.c
 * --------------
 * Application (matvec) code for BfVfHier.
 *
 * Responsibilities:
 *   - bfVfHierApply / bfVfHierApplyMany
 *   - apply-plan construction (tiling + per-leaf tasks)
 *   - optimized kernels for CSR leaves and SVD leaves
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
#include <stdbool.h>
#include <stdlib.h>  /* getenv, atoi */
#include <stdio.h>
#include <math.h>
#include <time.h>  /* for timing instrumentation */
#include <string.h>         /* for memset */
#include <bf/real_array.h>  /* for BfRealArray, bfRealArrayNewWithDefaultCapacity, etc */
#include <bf/ptr_array.h>
#include <bf/vf_hier_internal.h>

#ifdef BF_OPENMP
#include <omp.h>
#endif

// Prototypes


typedef struct BfVfApplyTask BfVfApplyTask;
typedef struct BfVfApplyPlan BfVfApplyPlan;

void bfVfHierBuildApplyPlan(BfVfHier *vfHier, BfSize tileSize);

/* NEW: build CSR submatrix from local row/col indices in A_par */
static BfMatCsrReal *
bfMatCsrRealNewSubmatrixFromIndices(BfMatCsrReal const *A_par,
                                    BfSizeArray  const *rowIdx,
                                    BfSizeArray  const *colIdx);

/* topology-agnostic far test: uses your generic helper in vf_hier_build.c */
extern BfBool bfVfIsFarTreeNodes_(BfTreeNode const *rowNode,
                                  BfTreeNode const *colNode,
                                  BfReal eta);

/* topology-agnostic index extraction: already defined in vf_hier_build.c */
extern void bfVfGetNodeIndsFromTreeNode_(BfTreeNode *node,
                                        BfTree const *tree,
                                        BfSizeArray *inds);

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
    BfVfFaceMap  const *faceMap,
    BfTree const *tree,
    BfTreeNode const *rowNode,
    BfTreeNode const *colNode,
    BfVfBlockMeta const *meta,
    BfReal eta,
    BfReal tol,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac);

/* Quadtree-only pairing used by buildBlockHybrid (raytrace/topology path) */
typedef struct {
  BfQuadtreeNode *row;
  BfQuadtreeNode *col;
} QuadChildPair;


BfVfHierBlock *buildBlockHybrid(
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

BfVfHierBlock *buildSubtreeFromTrimeshUsingCsr(
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

void bfVfSvdLeafApply(BfVfSvdLeaf const *leaf,
                             BfReal const      *x,
                             BfReal            *y,
                             BfSize             n);

static void           bfVfHierBlockApply(BfVfHierBlock const *block,
                                         BfReal const        *x,
                                         BfReal              *y,
                                         BfSize               n);

// END Prototypes

/* build (or rebuild) vfHier->applyPlan */
void bfVfHierBuildApplyPlan(BfVfHier *vfHier, BfSize tileSize) {
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

void bfVfSvdLeafApply(BfVfSvdLeaf const *leaf,
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
  BfVecReal const *vy = bfVecConstToVecRealConst(vyBase);
  BfReal const *yData = bfVecRealGetDataConstPtr(vy);

  /* Scatter-add into global y */
  for (BfSize i = 0; i < m; ++i) {
    BfSize row = bfSizeArrayGet((BfSizeArray *)&leaf->rowInds, i);
    BF_ASSERT(row < n);
    y[row] += yData[i];
  }

  bfVecRealDeinitAndDealloc(&vx);
  bfVecDelete(&vyBase);
}

void bfVfHierApply(BfVfHier const *vfHier,
                   BfReal const   *x,
                   BfReal         *y)
{
  bfVfHierApplyMany(vfHier, x, vfHier->n, y, vfHier->n, 1);
}

/* ============================================================
 * ApplyMany implementation (moved from vf_hier_io.c)
 * ============================================================ */

/* Local helper: unpack MatProduct factors (U, S, VT) */
static BfBool unpack_matproduct_usvt_(BfMat *P,
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

static void ensure_svd_cache_(BfVfSvdLeaf const *leaf) {
  if (leaf->U_cache && leaf->S_cache && leaf->VT_cache) return;

  #pragma omp critical(bf_vf_svd_cache_init)
  {
    if (!(leaf->U_cache && leaf->S_cache && leaf->VT_cache)) {
      BfVfSvdLeaf *m = (BfVfSvdLeaf *)leaf;
      BfMatDenseReal *U = NULL, *VT = NULL;
      BfMatDiagReal *S = NULL;
      BF_ASSERT(unpack_matproduct_usvt_(leaf->mat, &U, &S, &VT));
      m->U_cache  = U;
      m->S_cache  = S;
      m->VT_cache = VT;
    }
  }
}

static void svd_precompute_Z_(BfVfSvdLeaf const *leaf,
                            BfReal const      *X, BfSize ldX,
                            BfSize             n,
                            BfSize             nrhs,
                            BfReal            *Z /* len = nrhs*rank */)
{
  ensure_svd_cache_(leaf);

  BfSize mj   = bfSizeArrayGetSize((BfSizeArray *)&leaf->colInds);
  BfSize rank = leaf->rank;

  BfMatDenseReal const *VTm = leaf->VT_cache; /* rank x mj */
  BfMatDiagReal  const *Sm  = leaf->S_cache;

  BfReal const *VT = VTm->data;
  BfSize rsVT = VTm->super.rowStride;
  BfSize csVT = VTm->super.colStride;
  BfReal const *S = Sm->data; /* length rank */

  for (BfSize r = 0; r < nrhs*rank; ++r) Z[r] = 0;

  for (BfSize j = 0; j < mj; ++j) {
    BfSize globalCol = bfSizeArrayGet((BfSizeArray *)&leaf->colInds, j);
    BF_ASSERT(globalCol < n);

    BfReal const *xCol = X + globalCol;
    for (BfSize k = 0; k < rank; ++k) {
      BfReal vt = VT[k*rsVT + j*csVT];
      for (BfSize rhs = 0; rhs < nrhs; ++rhs)
        Z[rhs*rank + k] += vt * xCol[rhs*ldX];
    }
  }

  for (BfSize rhs = 0; rhs < nrhs; ++rhs)
    for (BfSize k = 0; k < rank; ++k)
      Z[rhs*rank + k] *= S[k];
}

static void svd_accumulate_rows_from_Z_(BfVfSvdLeaf const *leaf,
                                      BfReal const      *Z,
                                      BfReal            *Y, BfSize ldY,
                                      BfSize             n,
                                      BfSize             nrhs,
                                      BfSize             iBegin,
                                      BfSize             iEnd)
{
  ensure_svd_cache_(leaf);

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

  if (vfHier->applyPlan == NULL) {
    bfVfHierBuildApplyPlan((BfVfHier *)vfHier, 1024);
  }

  BfVfApplyPlan const *p = vfHier->applyPlan;
  BfSize n = vfHier->n;

  BfReal **Zptr = NULL;
  BfReal  *Zbuf = NULL;

  if (p->numSvdBlocks > 0) {
    Zptr = bfMemAlloc(p->numSvdBlocks, sizeof(BfReal *));

    BfSize total = 0;
    for (BfSize i = 0; i < p->numSvdBlocks; ++i) {
      BfVfSvdLeaf const *leaf = &p->svdBlocks[i]->data.svd;
      total += leaf->rank * nrhs;
    }
    Zbuf = bfMemAlloc(total, sizeof(BfReal));

    BfSize off = 0;
    for (BfSize i = 0; i < p->numSvdBlocks; ++i) {
      BfVfSvdLeaf const *leaf = &p->svdBlocks[i]->data.svd;
      Zptr[i] = Zbuf + off;
      off += leaf->rank * nrhs;
    }

    #pragma omp parallel for schedule(dynamic) if(!omp_in_parallel())
    for (BfSize i = 0; i < p->numSvdBlocks; ++i) {
      BfVfSvdLeaf const *leaf = &p->svdBlocks[i]->data.svd;
      svd_precompute_Z_(leaf, X, ldX, n, nrhs, Zptr[i]);
    }
  }

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
        svd_accumulate_rows_from_Z_(&b->data.svd, Zptr[sid], Y, ldY,
                                   n, nrhs, task->iBegin, task->iEnd);
      }
    }
  }

  if (Zptr) bfMemFree(Zptr);
  if (Zbuf) bfMemFree(Zbuf);
}



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

void reindexCsrColsToLocal(BfMatCsrReal *A, BfSizeArray const *colFaces, BfSize nFaces) {
  (void)colFaces;
  (void)nFaces;

  BfSize mA = bfMatGetNumRows(bfMatCsrRealToMat(A));
  BfSize nA = bfMatGetNumCols(bfMatCsrRealToMat(A));
  BfSize nnz = A->rowptr[mA];

  /* Hard invariant (no heuristics): CSR column indices MUST already be local. */
  for (BfSize k = 0; k < nnz; ++k) {
    BF_ASSERT(A->colind[k] < nA);
  }
}

static void assertCsrColsLocal(BfMatCsrReal const *A) {
  BfMat const *A0 = bfMatCsrRealConstToMatConst(A);
  BfSize m = bfMatGetNumRows(A0);
  BfSize n = bfMatGetNumCols(A0);

  BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(A);
  BfSize const *ci = bfMatCsrRealGetColindConstPtr(A);
  BF_ASSERT(rp && ci);

  BfSize nnz = rp[m];
  for (BfSize k = 0; k < nnz; ++k)
    BF_ASSERT(ci[k] < n);
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
  BfQuadtree const *qt = bfQuadtreeNodeGetQuadtree((BfQuadtreeNode *)rowNode);
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
    /* CSR colind MUST already be local indices into colInds[] */
    assertCsrColsLocal(Acsr);

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

  /* FAR + SVD enabled: CSR colind MUST already be local indices into colInds[] */
  assertCsrColsLocal(Acsr);

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
    BfVfFaceMap  const *faceMap,
    BfTree const *tree,
    BfTreeNode const *rowNode,
    BfTreeNode const *colNode,
    BfVfBlockMeta const *meta,
    BfReal eta,
    BfReal tol,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac)
{
  (void)rowFaces_par;
  (void)colFaces_par;

  if (meta->empty)
    return NULL;

#if BF_VF_HIER_TIME_SVD
  double t_map_start = bfVfHierNowSecs();
#endif

  /* Topology-agnostic node index extraction */
  BfSizeArray childRowFaces;
  BfSizeArray childColFaces;
  bfSizeArrayInitWithDefaultCapacity(&childRowFaces);
  bfSizeArrayInitWithDefaultCapacity(&childColFaces);

  BF_ASSERT(tree != NULL);
  bfVfGetNodeIndsFromTreeNode_((BfTreeNode *)rowNode, tree, &childRowFaces);
  bfVfGetNodeIndsFromTreeNode_((BfTreeNode *)colNode, tree, &childColFaces);

  BfSize mChild = bfSizeArrayGetSize(&childRowFaces);
  BfSize nChild = bfSizeArrayGetSize(&childColFaces);

  if (mChild == 0 || nChild == 0) {
    bfSizeArrayDeinit(&childRowFaces);
    bfSizeArrayDeinit(&childColFaces);
    return NULL;
  }

  /* Use cached global->parent-local maps */
  BF_ASSERT(faceMap != NULL);
  BfSize mapSize             = faceMap->mapSize;
  BfSize const *globalToRow  = faceMap->globalToRow;
  BfSize const *globalToCol  = faceMap->globalToCol;

  BfSizeArray rowIdxPar;
  BfSizeArray colIdxPar;
  bfSizeArrayInitWithDefaultCapacity(&rowIdxPar);
  bfSizeArrayInitWithDefaultCapacity(&colIdxPar);

  for (BfSize i = 0; i < mChild; ++i) {
    BfSize gRow = bfSizeArrayGet(&childRowFaces, i);
    if (gRow >= mapSize) continue;
    BfSize rLoc = globalToRow[gRow];
    if (rLoc == BF_SIZE_BAD_VALUE) continue;
    bfSizeArrayAppend(&rowIdxPar, rLoc);
  }

  for (BfSize j = 0; j < nChild; ++j) {
    BfSize gCol = bfSizeArrayGet(&childColFaces, j);
    if (gCol >= mapSize) continue;
    BfSize cLoc = globalToCol[gCol];
    if (cLoc == BF_SIZE_BAD_VALUE) continue;
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

  assertCsrColsLocal(Acsr);
  BfMat *A = bfMatCsrRealToMat(Acsr);
  BfSize mA = bfMatGetNumRows(A);
  BfSize nA = bfMatGetNumCols(A);

  /* Zero-block elision (as you had) */
  {
    BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(Acsr);
    BfReal const *da = bfMatCsrRealGetDataConstPtr(Acsr);
    BF_ASSERT(rp && da);

    BfSize nnz = rp[mA];
    BfReal maxAbs = 0;
    for (BfSize k = 0; k < nnz; ++k) {
      BfReal v = da[k];
      if (v < 0) v = -v;
      if (v > maxAbs) maxAbs = v;
    }
    if (nnz == 0 || maxAbs == 0) {
      bfMatCsrRealDeinitAndDealloc(&Acsr);
      bfSizeArrayDeinit(&childRowFaces);
      bfSizeArrayDeinit(&childColFaces);
      return NULL;
    }
  }

  /* Far vs near test based on geometry, exactly like other paths */
  BfBool far = bfVfIsFarTreeNodes_(rowNode, colNode, eta);

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


  BfBackend backend = BF_BACKEND_SVDS;
  BfBool truncated;

  /* progress: count an attempted SVD (passes all pre-SVD gates) */
  if (g_svd_tries_total > 0ull) {
    unsigned long long done;
    #pragma omp atomic capture
    done = ++g_svd_tries_done;
    vfHierMaybePrintProgress_(done, g_svd_tries_total);
  }

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
BfVfHierBlock *buildBlockFromCsrMidlevel(
    BfMatCsrReal const *A_par,
    BfSizeArray  const *rowFaces_par,
    BfSizeArray  const *colFaces_par,
    BfVfFaceMap  const *faceMap,
    BfTree const *tree,
    BfTreeNode *rowNode,
    BfTreeNode *colNode,
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
      tree,
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
    /* Final leaf: this is where recursion stops and a block is materialized
     * (sparse OR SVD). Use this as a proxy for overall build completion.
     *
     * NOTE: flatBlock may be NULL for exactly-zero blocks; we still count it
     * so “done” reaches “total” in the dry-run model. */

    /* NEW: record/cache anything needed once the final leaf is chosen */
    vfHierOnLeafMaterialized_();  /* OK if flatBlock == NULL */

    return flatBlock;  /* may be NULL for exactly-zero block */
  }

  /* --- 2) build hierarchical children as before --- */

  BfVfHierBlock *nodeBlock = bfVfHierBlockNew();
  nodeBlock->kind = BF_VF_HIER_BLOCK_NODE;
  bfInitPtrArray(&nodeBlock->data.node.children, 4);

  BfTreeNode *ni = rowNode;
  BfTreeNode *nj = colNode;

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

      for (BfSize b = 0; b < maxChildrenJ; ++b) {
        if (!bfTreeNodeHasChild(nj, b)) continue;
        BfTreeNode *cj = bfTreeNodeGetChild(nj, b);

        pairs[numPairs].row = ci;
        pairs[numPairs].col = cj;

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
            tree,
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
      pairs[numPairs].row = bfTreeNodeToQuadtreeNode(ci);
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
            tree,
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

      pairs[numPairs].row = rowNode;
      pairs[numPairs].col = cj;

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
              tree,
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
BfVfHierBlock *buildBlockHybrid(
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
    if (leaf != NULL) {
      /* NEW: leaf materialized via early-stop path */
      vfHierOnLeafMaterialized_();
      return leaf;
    }
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

  BfTreeNode *ni = rowNode;
  BfTreeNode *nj = colNode;

  bool leafI = meta.leafI;
  bool leafJ = meta.leafJ;

  if (!leafI && !leafJ) {
    BfSize maxChildrenI = bfTreeNodeGetMaxNumChildren(ni);
    BfSize maxChildrenJ = bfTreeNodeGetMaxNumChildren(nj);

    QuadChildPair *pairs = bfMemAlloc(maxChildrenI * maxChildrenJ, sizeof(QuadChildPair));
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

    QuadChildPair *pairs = bfMemAlloc(maxChildrenI, sizeof(QuadChildPair));
    BfSize numPairs = 0;

    for (BfSize a = 0; a < maxChildrenI; ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfTreeNode *ci = bfTreeNodeGetChild(ni, a);

      pairs[numPairs].row = ci;
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

    QuadChildPair *pairs = bfMemAlloc(maxChildrenJ, sizeof(QuadChildPair));
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
BfVfHierBlock *buildSubtreeFromTrimeshUsingCsr(
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

  /* Parent CSR colind MUST already be local indices into colFaces_par[] */
  assertCsrColsLocal(A_par);

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

      /* NEW: parent CSR heuristic returns a final leaf */
      vfHierOnLeafMaterialized_();

      return block;  /* NOTE: no deinit of rowFaces_par/colFaces_par/A_par here */
    }
  }
#endif /* BF_VF_HIER_ENABLE_PARENT_CSR_HEURISTIC */

  BfVfFaceMap faceMap;
  bfVfFaceMapInitFromParentFaces(&faceMap,
                                 (BfSizeArray const *)&rowFaces_par,
                                 (BfSizeArray const *)&colFaces_par);

  vfHierInitProgressFromEnv_();
  const char *doCount = getenv("BF_VF_HIER_COUNT_SVD_TRIES");
  if (doCount && atoi(doCount) != 0) {
    unsigned long long total = 0ull;
    countSvdTriesFromCsrMidlevel(
        (BfMatCsrReal const *)A_par,
        (BfVfFaceMap  const *)&faceMap,
        (BfTreeNode *)rowNode, (BfTreeNode *)colNode,
        eta, leafMax, leafMin, (BfSize)minArea,
        tol, minSvdSize, maxSvdRankFrac,
        0, &total);

    g_svd_tries_total = total;
    g_svd_tries_done  = 0ull;
    fprintf(stderr, "[vf_hier] dry-run: total SVD tries (by gates) = %llu\n",
            (unsigned long long)g_svd_tries_total);
  }


  /* NEW: build the subtree purely from CSR slicing + SVD decisions. */
  BfVfHierBlock *subtree =
    buildBlockFromCsrMidlevel(
        (BfMatCsrReal const *)A_par,
        (BfSizeArray  const *)&rowFaces_par,
        (BfSizeArray  const *)&colFaces_par,
        (BfVfFaceMap  const *)&faceMap,
        (BfTree const *)qt,
        (BfTreeNode *)rowNode,
        (BfTreeNode *)colNode,
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

