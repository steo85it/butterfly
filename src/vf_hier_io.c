#include <bf/def.h>
#include <bf/vf_hier.h>
#include <stdbool.h>  /* for bool in this TU, even if internal.h is included */

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

// Prototypes
void countSvdTriesFromCsrMidlevel(
    BfMatCsrReal const *A_par,
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
    int depth,
    unsigned long long *counter);

static void countSvdTryInLeafFromCsrMidlevel_(
    BfMatCsrReal const *A_par,
    BfVfFaceMap  const *faceMap,
    BfQuadtreeNode const *rowNode,
    BfQuadtreeNode const *colNode,
    BfVfBlockMeta const *meta,
    BfReal eta,
    BfReal tol,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac,
    unsigned long long *counter);



// End Prototypes

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

/* ============================================================
 * Helper: count nnz and maxAbs in a submatrix of a parent CSR,
 * restricted to parent-local row indices (rowIdxPar) and parent-local
 * column indices (colIdxPar), without slicing/materializing.
 * ============================================================ */
static BfSize countSubmatrixNnzAndMaxAbs_(
    BfMatCsrReal const *A_par,
    BfSizeArray const  *rowIdxPar,
    BfSizeArray const  *colIdxPar,
    BfReal             *maxAbsOut)
{
  BfSize mSub = bfSizeArrayGetSize((BfSizeArray *)rowIdxPar);
  BfSize nSub = bfSizeArrayGetSize((BfSizeArray *)colIdxPar);
  if (mSub == 0 || nSub == 0) { if (maxAbsOut) *maxAbsOut = 0; return 0; }

  /* Build a membership bitmap for columns (parent-local col indices). */
  BfSize maxCol = 0;
  for (BfSize j = 0; j < nSub; ++j) {
    BfSize c = bfSizeArrayGet((BfSizeArray *)colIdxPar, j);
    if (c > maxCol) maxCol = c;
  }

  unsigned char *isCol = bfMemAlloc(maxCol + 1, sizeof(unsigned char));
  if (!isCol) { if (maxAbsOut) *maxAbsOut = 0; return 0; }
  memset(isCol, 0, (maxCol + 1)*sizeof(unsigned char));

  for (BfSize j = 0; j < nSub; ++j) {
    BfSize c = bfSizeArrayGet((BfSizeArray *)colIdxPar, j);
    if (c <= maxCol) isCol[c] = 1;
  }

  BfSize const *rp = bfMatCsrRealGetRowptrConstPtr((BfMatCsrReal *)A_par);
  BfSize const *ci = bfMatCsrRealGetColindConstPtr((BfMatCsrReal *)A_par);
  BfReal const *da = bfMatCsrRealGetDataConstPtr((BfMatCsrReal *)A_par);
  BF_ASSERT(rp && ci && da);

  BfSize nnz = 0;
  BfReal maxAbs = 0;
  for (BfSize i = 0; i < mSub; ++i) {
    BfSize r = bfSizeArrayGet((BfSizeArray *)rowIdxPar, i);
    for (BfSize k = rp[r]; k < rp[r + 1]; ++k) {
      BfSize c = ci[k];
      if (c <= maxCol && isCol[c]) {
        ++nnz;
        BfReal v = da[k]; if (v < 0) v = -v;
        if (v > maxAbs) maxAbs = v;
      }
    }
  }

  bfMemFree(isCol);
  if (maxAbsOut) *maxAbsOut = maxAbs;
  return nnz;
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

void bfVfSvdLeafApplyMany(BfVfSvdLeaf const *leaf,
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

void collect_leaf_blocks(BfVfHierBlock const *block, BfPtrArray *out) {
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


BfSize get_leaf_row_i0(BfVfHierBlock const *b) {
  if (b->kind == BF_VF_HIER_BLOCK_SPARSE) return b->data.sparse.row_i0;
  if (b->kind == BF_VF_HIER_BLOCK_SVD)    return b->data.svd.row_i0;
  return 0;
}

BfSize get_leaf_row_i1(BfVfHierBlock const *b) {
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

BfSize get_leaf_num_rows(BfVfHierBlock const *b) {
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

static void countSvdTryInLeafFromCsrMidlevel_(
    BfMatCsrReal const *A_par,
    BfVfFaceMap  const *faceMap,
    BfQuadtreeNode const *rowNode,
    BfQuadtreeNode const *colNode,
    BfVfBlockMeta const *meta,
    BfReal eta,
    BfReal tol,
    BfSize minSvdSize,
    BfReal maxSvdRankFrac,
    unsigned long long *counter)
{
  (void)tol; (void)maxSvdRankFrac; /* not used in gates-before-SVD except tol */

  if (!counter) return;
  if (meta->empty) return;

  BfQuadtree const *qt =
    bfQuadtreeNodeGetQuadtree((BfQuadtreeNode *)rowNode);

  /* global faces covered by these nodes */
  BfSizeArray childRowFaces, childColFaces;
  bfSizeArrayInitWithDefaultCapacity(&childRowFaces);
  bfSizeArrayInitWithDefaultCapacity(&childColFaces);
  getNodeInds((BfQuadtreeNode *)rowNode, qt, &childRowFaces);
  getNodeInds((BfQuadtreeNode *)colNode, qt, &childColFaces);

  BfSize mChild = bfSizeArrayGetSize(&childRowFaces);
  BfSize nChild = bfSizeArrayGetSize(&childColFaces);
  if (mChild == 0 || nChild == 0) goto cleanup_faces;

  /* map global face -> parent-local row/col index */
  BF_ASSERT(faceMap && faceMap->globalToRow && faceMap->globalToCol);

  BfSizeArray rowIdxPar, colIdxPar;
  bfSizeArrayInitWithDefaultCapacity(&rowIdxPar);
  bfSizeArrayInitWithDefaultCapacity(&colIdxPar);

  for (BfSize i = 0; i < mChild; ++i) {
    BfSize g = bfSizeArrayGet(&childRowFaces, i);
    if (g >= faceMap->mapSize) continue;
    BfSize rLoc = faceMap->globalToRow[g];
    if (rLoc == BF_SIZE_BAD_VALUE) continue;
    bfSizeArrayAppend(&rowIdxPar, rLoc);
  }
  for (BfSize j = 0; j < nChild; ++j) {
    BfSize g = bfSizeArrayGet(&childColFaces, j);
    if (g >= faceMap->mapSize) continue;
    BfSize cLoc = faceMap->globalToCol[g];
    if (cLoc == BF_SIZE_BAD_VALUE) continue;
    bfSizeArrayAppend(&colIdxPar, cLoc);
  }

  if (bfSizeArrayGetSize(&rowIdxPar) == 0 || bfSizeArrayGetSize(&colIdxPar) == 0) {
    bfSizeArrayDeinit(&rowIdxPar);
    bfSizeArrayDeinit(&colIdxPar);
    goto cleanup_faces;
  }

  /* No slicing here: just scan parent CSR restricted to (rowIdxPar, colIdxPar). */
  BfSize mA = bfSizeArrayGetSize(&rowIdxPar);
  BfSize nA = bfSizeArrayGetSize(&colIdxPar);
  BfReal maxAbs = 0;
  BfSize nnz = countSubmatrixNnzAndMaxAbs_(A_par, &rowIdxPar, &colIdxPar, &maxAbs);
  bfSizeArrayDeinit(&rowIdxPar);
  bfSizeArrayDeinit(&colIdxPar);
  if (nnz == 0) goto cleanup_faces;
  if (maxAbs == 0) goto cleanup_faces;

  /* gates that determine “would call SVD” */
  BfBool far = isFar(rowNode, colNode, eta);
  if (!far || minSvdSize == 0) goto cleanup_faces;

  unsigned long long blockSize = (unsigned long long)mA * (unsigned long long)nA;
  if (blockSize < (unsigned long long)minSvdSize) goto cleanup_faces;

//  if (mA < 3 || nA < 3) { bfMatCsrRealDeinitAndDealloc(&Acsr); goto cleanup_faces; }
//
//  /* memory pre-reject gate (rank-1 vs CSR) */
//  {
//    BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(Acsr);
//    BF_ASSERT(rp);
//    BfSize nnz = rp[mA];
//    double bytesCsr = 8.0*(double)nnz + 8.0*(double)nnz + 8.0*(double)(mA + 1);
//    double bytesSvd_min = 8.0 * ((double)mA + (double)nA + 1.0);
//    if (bytesSvd_min >= bytesCsr) { bfMatCsrRealDeinitAndDealloc(&Acsr); goto cleanup_faces; }
//  }

  /* If we reach here, the real code would call bfGetTruncatedSvd */
  (*counter)++;

//  bfMatCsrRealDeinitAndDealloc(&Acsr);

cleanup_faces:
  bfSizeArrayDeinit(&childRowFaces);
  bfSizeArrayDeinit(&childColFaces);
}

void countSvdTriesFromCsrMidlevel(
    BfMatCsrReal const *A_par,
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
    int depth,
    unsigned long long *counter)
{
  (void)depth; (void)tol; (void)maxSvdRankFrac;

  BfVfBlockMeta meta = getBlockMeta(rowNode, colNode, leafMax);
  if (meta.empty) return;

  /* count leaf candidate at this (rowNode,colNode) */
  countSvdTryInLeafFromCsrMidlevel_(
      A_par, faceMap, rowNode, colNode, &meta,
      eta, tol, minSvdSize, maxSvdRankFrac, counter);

  BfSize mi = meta.mi, mj = meta.mj;
  bool leafI = meta.leafI, leafJ = meta.leafJ, small = meta.small;

  unsigned long long area = (unsigned long long)mi * (unsigned long long)mj;

  bool forceLeaf =
      (leafI && leafJ) ||
      (small && mi >= leafMin && mj >= leafMin) ||
      (area <= (unsigned long long)minArea);

  if (forceLeaf) return;

  /* recurse exactly like buildBlockFromCsrMidlevel’s splitting cases */
  BfTreeNode *ni = bfQuadtreeNodeToTreeNode(rowNode);
  BfTreeNode *nj = bfQuadtreeNodeToTreeNode(colNode);

  if (!leafI && !leafJ) {
    BfSize maxChildrenI = bfTreeNodeGetMaxNumChildren(ni);
    BfSize maxChildrenJ = bfTreeNodeGetMaxNumChildren(nj);

    for (BfSize a = 0; a < maxChildrenI; ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfQuadtreeNode *nai = bfTreeNodeToQuadtreeNode(bfTreeNodeGetChild(ni, a));

      for (BfSize b = 0; b < maxChildrenJ; ++b) {
        if (!bfTreeNodeHasChild(nj, b)) continue;
        BfQuadtreeNode *nbj = bfTreeNodeToQuadtreeNode(bfTreeNodeGetChild(nj, b));

        countSvdTriesFromCsrMidlevel(
            A_par, faceMap, nai, nbj,
            eta, leafMax, leafMin, minArea,
            tol, minSvdSize, maxSvdRankFrac,
            depth + 1, counter);
      }
    }

  } else if (!leafI) {
    BfSize maxChildrenI = bfTreeNodeGetMaxNumChildren(ni);
    for (BfSize a = 0; a < maxChildrenI; ++a) {
      if (!bfTreeNodeHasChild(ni, a)) continue;
      BfQuadtreeNode *nai = bfTreeNodeToQuadtreeNode(bfTreeNodeGetChild(ni, a));
      countSvdTriesFromCsrMidlevel(
          A_par, faceMap, nai, colNode,
          eta, leafMax, leafMin, minArea,
          tol, minSvdSize, maxSvdRankFrac,
          depth + 1, counter);
    }

  } else { /* !leafJ */
    BfSize maxChildrenJ = bfTreeNodeGetMaxNumChildren(nj);
    for (BfSize b = 0; b < maxChildrenJ; ++b) {
      if (!bfTreeNodeHasChild(nj, b)) continue;
      BfQuadtreeNode *nbj = bfTreeNodeToQuadtreeNode(bfTreeNodeGetChild(nj, b));
      countSvdTriesFromCsrMidlevel(
          A_par, faceMap, rowNode, nbj,
          eta, leafMax, leafMin, minArea,
          tol, minSvdSize, maxSvdRankFrac,
          depth + 1, counter);
    }
  }
}