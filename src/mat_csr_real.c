#include <bf/mat_csr_real.h>

#include <bf/assert.h>
#include <bf/const.h>
#include <bf/error.h>
#include <bf/error_macros.h>
#include <bf/mem.h>
#include <bf/points.h>
#include <bf/real_array.h>
#include <bf/size_array.h>
#include <bf/trimesh.h>
#include <bf/util.h>
#include <bf/vec_real.h>
#include <bf/vectors.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>   /* strcmp/strcasecmp */
#include <stdint.h>   /* uint64_t */

#ifdef BF_OPENMP
#include <omp.h>
#endif

/** Interface: Mat */

static BfMatVtable MAT_VTABLE = {
  .GetView = (__typeof__(&bfMatGetView))bfMatCsrRealGetView,
  .Copy = (__typeof__(&bfMatCsrRealCopy))bfMatCsrRealCopy,
  .Delete = (__typeof__(&bfMatCsrRealDelete))bfMatCsrRealDelete,
  .GetType = (__typeof__(&bfMatCsrRealGetType))bfMatCsrRealGetType,
  .GetNumRows = (__typeof__(&bfMatCsrRealGetNumRows))bfMatCsrRealGetNumRows,
  .GetNumCols = (__typeof__(&bfMatCsrRealGetNumCols))bfMatCsrRealGetNumCols,
  .AddInplace = (__typeof__(&bfMatCsrRealAddInplace))bfMatCsrRealAddInplace,
  .Scale = (__typeof__(&bfMatCsrRealScale))bfMatCsrRealScale,
  .MulVec = (__typeof__(&bfMatCsrRealMulVec))bfMatCsrRealMulVec,
  .IsZero = (__typeof__(&bfMatCsrRealIsZero))bfMatCsrRealIsZero,
  .GetSubmatByMask = (__typeof__(&bfMatGetSubmatByMask))bfMatCsrRealGetSubmatByMask,
};

BfMat *bfMatCsrRealGetView(BfMatCsrReal *matCsrReal) {
  BF_ERROR_BEGIN();

  BfMatCsrReal *matCsrRealView = bfMatCsrRealNew();
  HANDLE_ERROR();

  *matCsrRealView = *matCsrReal;

  BfMat *matView = bfMatCsrRealToMat(matCsrRealView);

  matView->props |= BF_MAT_PROPS_VIEW;

  BF_ERROR_END()
    matView = NULL;

  return matView;
}

BfMat *bfMatCsrRealCopy(BfMat const *mat) {
  BF_ERROR_BEGIN();

  BfMatCsrReal const *matCsrReal = bfMatConstToMatCsrRealConst(mat);
  HANDLE_ERROR();

  BfSize m = bfMatGetNumRows(mat);
  BfSize n = bfMatGetNumCols(mat);

  BfMatCsrReal *copy = bfMatCsrRealNewFromPtrs(m, n, matCsrReal->rowptr, matCsrReal->colind, matCsrReal->data);
  HANDLE_ERROR();

  BF_ERROR_END() {
    BF_DIE();
  }

  return bfMatCsrRealToMat(copy);
}

void bfMatCsrRealDelete(BfMat **mat) {
  bfMatCsrRealDeinitAndDealloc((BfMatCsrReal **)mat);
}

BfType bfMatCsrRealGetType(BfMat const *mat) {
  (void)mat;
  return BF_TYPE_MAT_CSR_REAL;
}

BfSize bfMatCsrRealGetNumRows(BfMat const *mat) {
  if (bfMatGetType(mat) != BF_TYPE_MAT_CSR_REAL) {
    bfSetError(BF_ERROR_TYPE_ERROR);
    return BF_SIZE_BAD_VALUE;
  } else {
    return mat->numRows;
  }
}

BfSize bfMatCsrRealGetNumCols(BfMat const *mat) {
  if (bfMatGetType(mat) != BF_TYPE_MAT_CSR_REAL) {
    bfSetError(BF_ERROR_TYPE_ERROR);
    return BF_SIZE_BAD_VALUE;
  } else {
    return mat->numCols;
  }
}

void bfMatCsrRealScale(BfMat *mat, BfComplex scalar) {
  BF_ERROR_BEGIN();

  if (cimag(scalar) != 0)
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  BfReal scalar_ = creal(scalar);

  BfMatCsrReal *matCsrReal = bfMatToMatCsrReal(mat);
  HANDLE_ERROR();

  BfSize m = bfMatGetNumRows(mat);
  BfSize nnz = matCsrReal->rowptr[m];
  for (BfSize i = 0; i < nnz; ++i)
    matCsrReal->data[i] *= scalar_;

  BF_ERROR_END() {}
}

void bfMatCsrRealAddInplace(BfMat *mat, BfMat const *otherMat) {
  BF_ERROR_BEGIN();

  BfMatCsrReal *matCsrReal = bfMatToMatCsrReal(mat);
  HANDLE_ERROR();

  BfMatCsrReal const *otherMatCsrReal = bfMatConstToMatCsrRealConst(otherMat);
  HANDLE_ERROR();

  if (!bfMatCsrRealHasSameSparsityPattern(matCsrReal, otherMatCsrReal))
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  BfSize m = bfMatGetNumRows(mat);
  BfSize nnz = matCsrReal->rowptr[m];
  for (BfSize i = 0; i < nnz; ++i)
    matCsrReal->data[i] += otherMatCsrReal->data[i];

  BF_ERROR_END() {}
}

static BfVec *mulVec_vecReal(BfMat const *mat, BfVecReal const *vecReal) {
  BF_ERROR_BEGIN();

  BfMatCsrReal const *matCsrReal = bfMatConstToMatCsrRealConst(mat);
  HANDLE_ERROR();

  BfSize m = bfMatGetNumRows(mat);

  BfSize n = bfMatGetNumCols(mat);
  if (vecReal->super.size != n)
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);

  BfVecReal *result = bfVecRealNew();
  HANDLE_ERROR();

  bfVecRealInit(result, m);
  HANDLE_ERROR();

  BfReal const *src = vecReal->data;
  BfReal *dst = result->data;
  for (BfSize i = 0; i < m; ++i) {
    *dst = 0;
    for (BfSize j = matCsrReal->rowptr[i]; j < matCsrReal->rowptr[i + 1]; ++j) {
      BfReal elt = *(src + vecReal->stride*matCsrReal->colind[j]);
      *dst += elt*matCsrReal->data[j];
    }
    dst += result->stride;
  }

  BF_ERROR_END() {
    bfVecRealDeinitAndDealloc(&result);
  }

  return bfVecRealToVec(result);
}

BfVec *bfMatCsrRealMulVec(BfMat const *mat, BfVec const *vec) {
  BF_ERROR_BEGIN();

  BfVec *result = NULL;

  switch (bfVecGetType(vec)) {
  case BF_TYPE_VEC_REAL:
    result = mulVec_vecReal(mat, bfVecConstToVecRealConst(vec));
    HANDLE_ERROR();
    break;
  default:
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);
  }

  BF_ERROR_END() {
    bfVecDelete(&result);
  }

  return result;
}

bool bfMatCsrRealIsZero(BfMat const *mat) {
  return bfMatConstToMatCsrRealConst(mat)->rowptr[mat->numRows] == 0;
}

BfMat *bfMatCsrRealGetSubmatByMask(BfMatCsrReal const *matCsrReal, bool const *rowMask, bool const *colMask) {
  BF_ERROR_BEGIN();

  BfMat const *mat = bfMatCsrRealConstToMatConst(matCsrReal);

  BfSize numRows = 0;
  for (BfSize i = 0; i < bfMatGetNumRows(mat); ++i)
    if (rowMask[i])
      ++numRows;

  BfSize numCols = 0;
  for (BfSize j = 0; j < bfMatGetNumCols(mat); ++j)
    if (colMask[j])
      ++numCols;

  /* Count the number of nonzero entries in the submatrix */
  BfSize nnz = 0;
  for (BfSize i = 0; i < bfMatGetNumRows(mat); ++i) {
    if (!rowMask[i]) continue;
    for (BfSize k = matCsrReal->rowptr[i]; k < matCsrReal->rowptr[i + 1]; ++k) {
      BfSize j = matCsrReal->colind[k];
      if (colMask[j])
        ++nnz;
    }
  }

  BfSize *rowptr = bfMemAllocAndZero(numRows + 1, sizeof(BfSize));
  HANDLE_ERROR();

  BfSize *colind = bfMemAlloc(nnz, sizeof(BfSize));
  HANDLE_ERROR();

  BfReal *data = bfMemAlloc(nnz, sizeof(BfReal));
  HANDLE_ERROR();

  /* Set up arrays used to map rows and columns in the parent matrix
   * to the indexed rows and columns in the submatrix */

  BfSize *rowRemap = bfMemAlloc(bfMatGetNumRows(mat), sizeof(BfSize));
  HANDLE_ERROR();

  BfSize i_ = 0;
  for (BfSize i = 0; i < bfMatGetNumRows(mat); ++i)
    rowRemap[i] = rowMask[i] ? i_++ : BF_SIZE_BAD_VALUE;

  BfSize *colRemap = bfMemAlloc(bfMatGetNumCols(mat), sizeof(BfSize));
  HANDLE_ERROR();

  BfSize j_ = 0;
  for (BfSize j = 0; j < bfMatGetNumCols(mat); ++j)
    colRemap[j] = colMask[j] ? j_++ : BF_SIZE_BAD_VALUE;

  BfSize l = 0;
  for (BfSize i = 0; i < bfMatGetNumRows(mat); ++i) {
    if (!rowMask[i]) continue;
    for (BfSize k = matCsrReal->rowptr[i]; k < matCsrReal->rowptr[i + 1]; ++k) {
      BfSize j = matCsrReal->colind[k];
      if (colMask[j]) {
        ++rowptr[rowRemap[i] + 1];
        colind[l] = colRemap[matCsrReal->colind[k]];
        data[l++] = matCsrReal->data[k];
      }
    }
  }
  BF_ASSERT(l == nnz);

  bfSizeRunningSum(numRows + 1, rowptr);

  BfMatCsrReal *submat = bfMatCsrRealNewFromPtrs(numRows, numCols, rowptr, colind, data);
  HANDLE_ERROR();

  BF_ERROR_END() {
    BF_DIE();
  }

  bfMemFree(rowptr);
  bfMemFree(colind);
  bfMemFree(data);
  bfMemFree(rowRemap);
  bfMemFree(colRemap);

  return bfMatCsrRealToMat(submat);
}

/** Indices: */
BfSize const *bfMatCsrRealGetRowptrConstPtr(BfMatCsrReal const *A) { return A->rowptr; }
BfSize const *bfMatCsrRealGetColindConstPtr(BfMatCsrReal const *A) { return A->colind; }
BfReal const *bfMatCsrRealGetDataConstPtr  (BfMatCsrReal const *A) { return A->data;   }

/** Upcasting: */

BfMat *bfMatCsrRealToMat(BfMatCsrReal *matCsrReal) {
  return &matCsrReal->super;
}

BfMat const *bfMatCsrRealConstToMatConst(BfMatCsrReal const *matCsrReal) {
  return &matCsrReal->super;
}

/** Downcasting: */

BfMatCsrReal *bfMatToMatCsrReal(BfMat *mat) {
  if (!bfMatInstanceOf(mat, BF_TYPE_MAT_CSR_REAL)) {
    bfSetError(BF_ERROR_TYPE_ERROR);
    return NULL;
  } else {
    return (BfMatCsrReal *)mat;
  }
}

BfMatCsrReal const *bfMatConstToMatCsrRealConst(BfMat const *mat) {
  if (!bfMatInstanceOf(mat, BF_TYPE_MAT_CSR_REAL)) {
    bfSetError(BF_ERROR_TYPE_ERROR);
    return NULL;
  } else {
    return (BfMatCsrReal const *)mat;
  }
}

/** Implementation: MatCsrReal */

BfMatCsrReal *bfMatCsrRealNew() {
  BF_ERROR_BEGIN();

  BfMatCsrReal *mat = bfMemAlloc(1, sizeof(BfMatCsrReal));
  HANDLE_ERROR();

  BF_ERROR_END() {
    BF_DIE();
  }

  return mat;
}

BfMatCsrReal *bfMatCsrRealNewFromPtrs(BfSize numRows, BfSize numCols,
                                      BfSize const *rowptr, BfSize const *colind,
                                      BfReal const *data) {
  BF_ERROR_BEGIN();

  BfMatCsrReal *matCsrReal = bfMatCsrRealNew();
  HANDLE_ERROR();

  bfMatCsrRealInitFromPtrs(matCsrReal, numRows, numCols, rowptr, colind, data);
  HANDLE_ERROR();

  BF_ERROR_END() {
    BF_DIE();
  }

  return matCsrReal;
}

BfMatCsrReal *bfMatCsrRealNewFromArrays(BfSize numRows, BfSize numCols, BfSizeArray *rowptrArray, BfSizeArray *colindArray, BfRealArray *dataArray, BfPolicy policy) {
  BF_ERROR_BEGIN();

  BfMatCsrReal *matCsrReal = bfMatCsrRealNew();
  HANDLE_ERROR();

  bfMatCsrRealInitFromArrays(matCsrReal, numRows, numCols, rowptrArray, colindArray, dataArray, policy);
  HANDLE_ERROR();

  BF_ERROR_END() {
    BF_DIE();
  }

  return matCsrReal;
}

BfMatCsrReal *bfMatCsrRealNewFromBinaryFiles(char const *rowptrPath, char const *colindPath, char const *dataPath) {
  BF_ERROR_BEGIN();

  BfSizeArray *rowptrArray = bfSizeArrayNewFromFile(rowptrPath);
  HANDLE_ERROR();

  BfSizeArray *colindArray = bfSizeArrayNewFromFile(colindPath);
  HANDLE_ERROR();

  BfRealArray *dataArray = bfRealArrayNewFromFile(dataPath);
  HANDLE_ERROR();

  BfSize numCols = bfSizeArrayGetMaximum(colindArray) + 1;
  BfSize numRows = bfSizeArrayGetSize(rowptrArray) - 1;

  BfMatCsrReal *matCsrReal = bfMatCsrRealNewFromArrays(
    numRows, numCols, rowptrArray, colindArray, dataArray, BF_POLICY_STEAL);
  HANDLE_ERROR();

  BF_ERROR_END() {
    BF_DIE();
  }

  return matCsrReal;
}

static BfReal integrateViewFactorMidpointRule(BfTrimesh const *trimesh, BfSize srcInd, BfSize tgtInd) {
  BfReal const *pSrc = bfTrimeshGetFaceCentroidConstPtr(trimesh, srcInd);
  BfReal const *pTgt = bfTrimeshGetFaceCentroidConstPtr(trimesh, tgtInd);

  BfReal const *nSrc = bfTrimeshGetFaceUnitNormalConstPtr(trimesh, srcInd);
  BfReal const *nTgt = bfTrimeshGetFaceUnitNormalConstPtr(trimesh, tgtInd);

  BfReal areaTgt = bfTrimeshGetFaceArea(trimesh, tgtInd);

  BfVector3 dp;
  bfPoint3Sub(pSrc, pTgt, dp);

  BfReal rSquared = bfVector3Dot(dp, dp);

  BfReal dotSrc = -bfVector3Dot(nSrc, dp);
  BfReal dotTgt = bfVector3Dot(nTgt, dp);

  const char *dbg = getenv("BF_VF_DOT_DIAG");
  if (dbg && dbg[0] != '\0') {
    if (!(dotSrc > 0) || !(dotTgt > 0)) {
      const char *si = getenv("BF_VF_DOT_I");
      if (si) {
        BfSize srcFilter = (BfSize)strtoull(si, NULL, 10);
        if (srcInd == srcFilter) {
          fprintf(stderr,
            "[bf] dotDiag src=%zu tgt=%zu dotSrc=%g dotTgt=%g r2=%g\n",
            (size_t)srcInd, (size_t)tgtInd,
            (double)dotSrc, (double)dotTgt, (double)rSquared);
        }
      }
    }
  }

  return areaTgt*fmax(0, dotSrc)*fmax(0, dotTgt)/(BF_PI*rSquared*rSquared);
}

/* ---------------------------
 * View-factor upgrade (midpoint -> gauss) for near-field pairs
 * Controlled by env vars:
 *   BF_VF_METHOD=midpoint|gauss|adaptive
 *   BF_VF_GAUSS_ORDER=1|3|7     (1->1pt, 3->7pt, 7->13pt)
 *   BF_VF_NEAR_RATIO=2.0        (refine if d/h < nearRatio)
 *   BF_VF_MIN_MIDPOINT_TO_REFINE=1e-6
 *   BF_VF_STATS=1              (prints refinedPairs)
 * --------------------------- */

typedef enum {
  BF_VF_METHOD_MIDPOINT = 0,
  BF_VF_METHOD_GAUSS    = 1,
  BF_VF_METHOD_ADAPTIVE = 2,
} BfVfMethod;

typedef struct {
  BfVfMethod method;
  int gaussOrder;             /* 1, 3, 7 -> {1pt, 7pt, 13pt} */
  BfReal nearRatio;           /* refine if d/h < nearRatio */
  BfReal minMidpointToRefine; /* refine if F_mid > this */
  int adaptiveMaxLevels;      /* reserved/debug */
  bool stats;                 /* BF_VF_STATS=1 */
} BfVfParams;

static BfVfParams g_vfParams;
static bool g_vfParamsInitialized = false;

static BfVfMethod parseMethod(char const *s) {
  if (s == NULL || s[0] == '\0') return BF_VF_METHOD_MIDPOINT;
  if (!strcmp(s, "midpoint")) return BF_VF_METHOD_MIDPOINT;
  if (!strcmp(s, "gauss"))    return BF_VF_METHOD_GAUSS;
  if (!strcmp(s, "adaptive")) return BF_VF_METHOD_ADAPTIVE;
  return BF_VF_METHOD_MIDPOINT;
}

static int parseIntEnv(char const *name, int def) {
  char const *s = getenv(name);
  if (s == NULL || s[0] == '\0') return def;
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s) return def;
  return (int)v;
}

static BfReal parseRealEnv(char const *name, BfReal def) {
  char const *s = getenv(name);
  if (s == NULL || s[0] == '\0') return def;
  char *end = NULL;
  double v = strtod(s, &end);
  if (end == s) return def;
  return (BfReal)v;
}

static bool parseBoolEnv(char const *name, bool def) {
  char const *s = getenv(name);
  if (s == NULL || s[0] == '\0') return def;
  if (!strcmp(s, "0")) return false;
  if (!strcasecmp(s, "false") || !strcasecmp(s, "no")) return false;
  return true;
}

static void initVfParamsOnce(void) {
  if (g_vfParamsInitialized) return;

  g_vfParams.method = parseMethod(getenv("BF_VF_METHOD"));

  g_vfParams.gaussOrder = parseIntEnv("BF_VF_GAUSS_ORDER", 3);
  if (!(g_vfParams.gaussOrder == 1 || g_vfParams.gaussOrder == 3 || g_vfParams.gaussOrder == 7))
    g_vfParams.gaussOrder = 3;

  g_vfParams.nearRatio = parseRealEnv("BF_VF_NEAR_RATIO", (BfReal)2.0);
  g_vfParams.minMidpointToRefine = parseRealEnv("BF_VF_MIN_MIDPOINT_TO_REFINE", (BfReal)1e-6);

  g_vfParams.adaptiveMaxLevels = parseIntEnv("BF_VF_ADAPTIVE_MAX_LEVELS", 2);
  g_vfParams.stats = parseBoolEnv("BF_VF_STATS", false);

  g_vfParamsInitialized = true;
}

/* --- triangle quadrature rules on reference triangle (u,v), u>=0, v>=0, u+v<=1 --- */
typedef struct {
  int n;
  BfReal const *w;   /* length n */
  BfReal const *uv;  /* length 2n: (u0,v0,u1,v1,...) */
} BfTriQuadRule;

static BfReal const W1[]  = { (BfReal)1.0 };
static BfReal const UV1[] = { (BfReal)(1.0/3.0), (BfReal)(1.0/3.0) };

/* Dunavant 7-point (degree 5) */
static BfReal const W7[] = {
  (BfReal)0.225,
  (BfReal)0.132394152788506, (BfReal)0.132394152788506, (BfReal)0.132394152788506,
  (BfReal)0.125939180544827, (BfReal)0.125939180544827, (BfReal)0.125939180544827
};
static BfReal const UV7[] = {
  (BfReal)(1.0/3.0), (BfReal)(1.0/3.0),
  (BfReal)0.059715871789770, (BfReal)0.059715871789770,
  (BfReal)(1.0 - 2.0*0.059715871789770), (BfReal)0.059715871789770,
  (BfReal)0.059715871789770, (BfReal)(1.0 - 2.0*0.059715871789770),
  (BfReal)0.470142064105115, (BfReal)0.470142064105115,
  (BfReal)(1.0 - 2.0*0.470142064105115), (BfReal)0.470142064105115,
  (BfReal)0.470142064105115, (BfReal)(1.0 - 2.0*0.470142064105115),
};

/* Dunavant 13-point (degree 7) */
static BfReal const W13[] = {
  (BfReal)-0.149570044467670,
  (BfReal)0.175615257433204, (BfReal)0.175615257433204, (BfReal)0.175615257433204,
  (BfReal)0.053347235608839, (BfReal)0.053347235608839, (BfReal)0.053347235608839,
  (BfReal)0.077113760890257, (BfReal)0.077113760890257, (BfReal)0.077113760890257,
  (BfReal)0.077113760890257, (BfReal)0.077113760890257, (BfReal)0.077113760890257
};
static BfReal const UV13[] = {
  (BfReal)(1.0/3.0), (BfReal)(1.0/3.0),
  (BfReal)0.260345966079038, (BfReal)0.260345966079038,
  (BfReal)0.479308067841923, (BfReal)0.260345966079038,
  (BfReal)0.260345966079038, (BfReal)0.479308067841923,
  (BfReal)0.065130102902216, (BfReal)0.065130102902216,
  (BfReal)0.869739794195568, (BfReal)0.065130102902216,
  (BfReal)0.065130102902216, (BfReal)0.869739794195568,
  (BfReal)0.312865496004875, (BfReal)0.048690315425316,
  (BfReal)0.638444188569809, (BfReal)0.312865496004875,
  (BfReal)0.048690315425316, (BfReal)0.638444188569809,
  (BfReal)0.048690315425316, (BfReal)0.312865496004875,
  (BfReal)0.312865496004875, (BfReal)0.638444188569809,
  (BfReal)0.638444188569809, (BfReal)0.048690315425316,
};

static BfTriQuadRule getRule(int order) {
  if (order == 1) return (BfTriQuadRule){ .n = 1,  .w = W1,  .uv = UV1 };
  if (order == 3) return (BfTriQuadRule){ .n = 7,  .w = W7,  .uv = UV7 };
  return (BfTriQuadRule){ .n = 13, .w = W13, .uv = UV13 };
}

static inline void getFaceVerts(BfTrimesh const *trimesh, BfSize faceInd,
                                BfReal const **x0, BfReal const **x1, BfReal const **x2) {
  BfSize const *F = bfTrimeshGetFaceConstPtr(trimesh, faceInd);
  *x0 = bfTrimeshGetVertPtrConst(trimesh, F[0]);
  *x1 = bfTrimeshGetVertPtrConst(trimesh, F[1]);
  *x2 = bfTrimeshGetVertPtrConst(trimesh, F[2]);
}

/* p(u,v) = x0*(1-u-v) + x1*u + x2*v */
static inline void sampleFacePoint(BfTrimesh const *trimesh, BfSize faceInd,
                                   BfReal u, BfReal v, BfReal p[3]) {
  BfReal const *x0, *x1, *x2;
  getFaceVerts(trimesh, faceInd, &x0, &x1, &x2);

  BfReal w = (BfReal)1.0 - u - v;

  p[0] = w*x0[0] + u*x1[0] + v*x2[0];
  p[1] = w*x0[1] + u*x1[1] + v*x2[1];
  p[2] = w*x0[2] + u*x1[2] + v*x2[2];
}

/* robust characteristic length: max edge length */
static BfReal computeFaceCharLen(BfTrimesh const *trimesh, BfSize faceInd) {
  BfReal const *x0, *x1, *x2;
  getFaceVerts(trimesh, faceInd, &x0, &x1, &x2);

  BfVector3 e01, e12, e20;
  bfPoint3Sub(x1, x0, e01);
  bfPoint3Sub(x2, x1, e12);
  bfPoint3Sub(x0, x2, e20);

  BfReal l01 = bfVector3Norm(e01);
  BfReal l12 = bfVector3Norm(e12);
  BfReal l20 = bfVector3Norm(e20);

  BfReal h = l01;
  if (l12 > h) h = l12;
  if (l20 > h) h = l20;
  return h;
}

static BfReal integrateViewFactorGauss(BfTrimesh const *trimesh,
                                       BfSize srcInd, BfSize tgtInd,
                                       int gaussOrder) {
  BfTriQuadRule rule = getRule(gaussOrder);

  BfReal const *nSrc = bfTrimeshGetFaceUnitNormalConstPtr(trimesh, srcInd);
  BfReal const *nTgt = bfTrimeshGetFaceUnitNormalConstPtr(trimesh, tgtInd);

  BfReal areaTgt = bfTrimeshGetFaceArea(trimesh, tgtInd);

  BfReal sum = 0;

  for (int si = 0; si < rule.n; ++si) {
    BfReal uS = rule.uv[2*si + 0];
    BfReal vS = rule.uv[2*si + 1];
    BfReal wS = rule.w[si];

    BfReal pSrc[3];
    sampleFacePoint(trimesh, srcInd, uS, vS, pSrc);

    for (int ti = 0; ti < rule.n; ++ti) {
      BfReal uT = rule.uv[2*ti + 0];
      BfReal vT = rule.uv[2*ti + 1];
      BfReal wT = rule.w[ti];

      BfReal pTgt[3];
      sampleFacePoint(trimesh, tgtInd, uT, vT, pTgt);

      BfVector3 dp;
      bfPoint3Sub(pSrc, pTgt, dp);

      BfReal r2 = bfVector3Dot(dp, dp);
      if (r2 <= 0) continue;

      /* Match midpoint convention:
         dotSrc = -nSrc·(pSrc-pTgt), dotTgt = +nTgt·(pSrc-pTgt) */
      BfReal dotSrc = -bfVector3Dot(nSrc, dp);
      BfReal dotTgt =  bfVector3Dot(nTgt, dp);

      if (dotSrc <= 0 || dotTgt <= 0) continue;

      BfReal r4 = r2*r2;
      BfReal kernel = (dotSrc*dotTgt)/(BF_PI*r4);

      sum += (wS*wT)*kernel;
    }
  }

  return areaTgt*sum;
}

static BfReal integrateViewFactorHybrid(BfTrimesh const *trimesh,
                                        BfReal const *faceCharLen,
                                        BfSize srcInd, BfSize tgtInd,
                                        BfVfParams const *p,
                                        uint64_t *numRefined) {
  /* Midpoint intermediates */
  BfReal const *pSrcC = bfTrimeshGetFaceCentroidConstPtr(trimesh, srcInd);
  BfReal const *pTgtC = bfTrimeshGetFaceCentroidConstPtr(trimesh, tgtInd);

  BfReal const *nSrc = bfTrimeshGetFaceUnitNormalConstPtr(trimesh, srcInd);
  BfReal const *nTgt = bfTrimeshGetFaceUnitNormalConstPtr(trimesh, tgtInd);

  BfReal areaTgt = bfTrimeshGetFaceArea(trimesh, tgtInd);

  BfVector3 dp;
  bfPoint3Sub(pSrcC, pTgtC, dp);

  BfReal r2 = bfVector3Dot(dp, dp);
  if (r2 <= 0) return 0;

  BfReal dotSrc = -bfVector3Dot(nSrc, dp);
  BfReal dotTgt =  bfVector3Dot(nTgt, dp);

  BfReal posDotSrc = fmax((BfReal)0, dotSrc);
  BfReal posDotTgt = fmax((BfReal)0, dotTgt);

  BfReal F_mid = areaTgt*posDotSrc*posDotTgt/(BF_PI*r2*r2);

  if (p->method == BF_VF_METHOD_MIDPOINT)
    return F_mid;

  if (F_mid <= p->minMidpointToRefine)
    return F_mid;

  BfReal hSrc = faceCharLen[srcInd];
  BfReal hTgt = faceCharLen[tgtInd];
  BfReal h = (hSrc > hTgt ? hSrc : hTgt);
  if (h <= 0) return F_mid;

  BfReal d = sqrt(r2);
  if (d/h >= p->nearRatio)
    return F_mid;

  if (p->method == BF_VF_METHOD_GAUSS) {
    if (numRefined) ++(*numRefined);
    return integrateViewFactorGauss(trimesh, srcInd, tgtInd, p->gaussOrder);
  }

  /* ADAPTIVE reserved/debug: fall back for now */
  return F_mid;
}

#ifdef BF_EMBREE
BfMatCsrReal *bfMatCsrRealNewViewFactorMatrixFromTrimesh(
    BfTrimesh const *trimesh,
    BfSizeArray const *rowInds,
    BfSizeArray const *colInds)
{
  BF_ERROR_BEGIN();

  BfSize numRows = bfSizeArrayGetSize(rowInds);
  BfSize numCols = bfSizeArrayGetSize(colInds);

  /* Build GLOBAL->LOCAL col lookup so CSR stores LOCAL col indices [0,numCols) */
  BfSize nFaces = bfTrimeshGetNumFaces(trimesh);
  BfSize *colGlobalToLocal = bfMemAlloc(nFaces, sizeof(BfSize));
  HANDLE_ERROR();
  for (BfSize f = 0; f < nFaces; ++f) colGlobalToLocal[f] = BF_SIZE_BAD_VALUE;

  for (BfSize j = 0; j < numCols; ++j) {
    BfSize g = bfSizeArrayGet(colInds, j);
    BF_ASSERT(g < nFaces);
    colGlobalToLocal[g] = j;
  }

  BfSizeArray *rowptr = bfSizeArrayNewWithDefaultCapacity();
  HANDLE_ERROR();
  bfSizeArrayAppend(rowptr, 0);

  BfSizeArray *colind = bfSizeArrayNewWithDefaultCapacity();
  HANDLE_ERROR();
  BfRealArray *data = bfRealArrayNewWithDefaultCapacity();
  HANDLE_ERROR();

  /* Per-row scratch (built in parallel, stitched serially) */
  BfSizeArray **row_colind = bfMemAlloc(numRows, sizeof(*row_colind));
  BfRealArray **row_data   = bfMemAlloc(numRows, sizeof(*row_data));
  HANDLE_ERROR();
  for (BfSize i = 0; i < numRows; ++i) { row_colind[i] = NULL; row_data[i] = NULL; }

  /* Optional progress */
  const char *env = getenv("BF_PROGRESS");
  size_t progress_step = env ? strtoul(env, NULL, 10) : 0;

  /* Optional numeric cutoff (default 0 to avoid “surprise” sparsification) */
  BfReal eps = 0;
  const char *eps_env = getenv("BF_VIEW_FACTOR_EPS");
  if (eps_env != NULL) {
    char *endptr = NULL;
    double tmp = strtod(eps_env, &endptr);
    if (endptr != eps_env) eps = (BfReal)tmp;
  }

  /* Optional near-field refinement of the view-factor kernel */
  initVfParamsOnce();

  /* Precompute per-face characteristic length for d/h gate */
  BfReal *faceCharLen = bfMemAlloc(nFaces, sizeof(BfReal));
  HANDLE_ERROR();

#ifdef BF_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (BfSize f = 0; f < nFaces; ++f)
    faceCharLen[f] = computeFaceCharLen(trimesh, f);

  uint64_t refinedTotal = 0;


#ifdef BF_OPENMP
#pragma omp parallel for schedule(dynamic,8)
#endif
  for (BfSize i = 0; i < numRows; ++i) {
    if (progress_step && (i % progress_step == 0)) {
#ifdef BF_OPENMP
#pragma omp critical
#endif
      fprintf(stderr, "[bf] view-factor rows: %zu/%zu\n", (size_t)i, (size_t)numRows);
    }

    BfSize rowGlobal = bfSizeArrayGet(rowInds, i);

    /* Visibility returns GLOBAL face ids in [0, nFaces). We always map GLOBAL->LOCAL
       using colGlobalToLocal so the stored CSR uses LOCAL column indices [0, numCols). */
    BfSizeArray *vis =
       bfTrimeshGetVisibilityOpenSegment(trimesh, rowGlobal, colInds);
//      bfTrimeshGetVisibility(trimesh, rowGlobal, colInds);

    if (vis == NULL)
      vis = bfSizeArrayNewWithCapacity(0);

    BfSizeArray *ci = bfSizeArrayNewWithCapacity(bfSizeArrayGetSize(vis));
    BfRealArray *ri = bfRealArrayNewWithCapacity(bfSizeArrayGetSize(vis));

    for (BfSize t = 0; t < bfSizeArrayGetSize(vis); ++t) {
      BfSize x = bfSizeArrayGet(vis, t);

      BF_ASSERT(x < nFaces); /* visibility must return GLOBAL face ids */

      /* vis returns GLOBAL face ids */
      BfSize colGlobal = x;
      if (colGlobal >= nFaces) continue;

      /* Map GLOBAL col -> LOCAL col index in [0, numCols) */
      BfSize colLocal = colGlobalToLocal[colGlobal];
      if (colLocal == BF_SIZE_BAD_VALUE) continue;

      /* Flux-like: force diagonal to 0 */
      if (colGlobal == rowGlobal) continue;

#ifdef BF_OPENMP
      /* count refinements per-thread then atomically accumulate */
      uint64_t refinedLocal = 0;
      BfReal value = integrateViewFactorHybrid(trimesh, faceCharLen,
                                               rowGlobal, colGlobal,
                                               &g_vfParams,
                                               g_vfParams.stats ? &refinedLocal : NULL);
      if (g_vfParams.stats && refinedLocal) {
#pragma omp atomic
        refinedTotal += refinedLocal;
      }
#else
      BfReal value = integrateViewFactorHybrid(trimesh, faceCharLen,
                                               rowGlobal, colGlobal,
                                               &g_vfParams,
                                               g_vfParams.stats ? &refinedTotal : NULL);
#endif

      /* Defensive + optional cutoff */
      if (!isfinite(value)) continue;
      if (value < 0) value = 0;          /* view factors should be non-negative */
      if (value <= eps) continue;

      bfSizeArrayAppend(ci, colLocal);
      bfRealArrayAppend(ri, value);
    }

    bfSizeArrayDeinitAndDealloc(&vis);

    row_colind[i] = ci;
    row_data[i]   = ri;
  }

  /* Stitch CSR */
  for (BfSize i = 0; i < numRows; ++i) {
    BfSizeArray *ci = row_colind[i];
    BfRealArray *ri = row_data[i];

    bfSizeArrayExtend(colind, ci);
    bfRealArrayExtend(data, ri);

    BfSize nnz_i = bfSizeArrayGetSize(ci);
    BfSize prev = bfSizeArrayGetLast(rowptr);
    bfSizeArrayAppend(rowptr, prev + nnz_i);

    bfSizeArrayDeinitAndDealloc(&ci);
    bfRealArrayDeinitAndDealloc(&ri);
  }

  bfMemFree(row_colind);
  bfMemFree(row_data);
  bfMemFree(colGlobalToLocal);

  if (g_vfParams.stats) {
    fprintf(stderr,
            "[bf] vf refine: method=%d gaussOrder=%d nearRatio=%g minMid=%g refinedPairs=%llu\n",
            (int)g_vfParams.method, g_vfParams.gaussOrder,
            (double)g_vfParams.nearRatio, (double)g_vfParams.minMidpointToRefine,
            (unsigned long long)refinedTotal);
  }

  bfMemFree(faceCharLen);


  BfMatCsrReal *matCsrReal =
    bfMatCsrRealNewFromArrays(numRows, numCols, rowptr, colind, data, BF_POLICY_STEAL);
  HANDLE_ERROR();

  BF_ERROR_END() {
    BF_DIE();
  }

  return matCsrReal;
}
#endif

void bfMatCsrRealInitFromPtrs(BfMatCsrReal *mat, BfSize numRows, BfSize numCols,
                              BfSize const *rowptr, BfSize const *colind,
                              BfReal const *data) {
  BF_ERROR_BEGIN();

  BfSize nnz = rowptr[numRows];

  bfMatInit(&mat->super, &MAT_VTABLE, numRows, numCols);
  HANDLE_ERROR();

  mat->rowptr = bfMemAlloc(numRows + 1, sizeof(BfSize));
  HANDLE_ERROR();

  mat->colind = bfMemAlloc(nnz, sizeof(BfSize));
  HANDLE_ERROR();

  mat->data = bfMemAlloc(nnz, sizeof(BfReal));
  HANDLE_ERROR();

  bfMemCopy(rowptr, numRows + 1, sizeof(BfSize), mat->rowptr);
  bfMemCopy(colind, nnz, sizeof(BfSize), mat->colind);
  bfMemCopy(data, nnz, sizeof(BfReal), mat->data);

  BF_ERROR_END() {
    BF_DIE();
  }
}

void bfMatCsrRealInitFromArrays(BfMatCsrReal *matCsrReal, BfSize numRows, BfSize numCols, BfSizeArray *rowptrArray, BfSizeArray *colindArray, BfRealArray *dataArray, BfPolicy policy) {
  BF_ERROR_BEGIN();

  bfMatInit(&matCsrReal->super, &MAT_VTABLE, numRows, numCols);
  HANDLE_ERROR();

  if (policy == BF_POLICY_STEAL) {
    bfSizeArrayShrinkCapacityToSize(rowptrArray);
    bfSizeArrayShrinkCapacityToSize(colindArray);
    bfRealArrayShrinkCapacityToSize(dataArray);

    matCsrReal->rowptr = bfSizeArrayStealPtr(rowptrArray);
    HANDLE_ERROR();

    matCsrReal->colind = bfSizeArrayStealPtr(colindArray);
    HANDLE_ERROR();

    matCsrReal->data = bfRealArrayStealPtr(dataArray);
    HANDLE_ERROR();
  }

  else RAISE_ERROR(BF_ERROR_NOT_IMPLEMENTED);

  BF_ERROR_END() {
    BF_DIE();
  }
}

void bfMatCsrRealDeinit(BfMatCsrReal *mat) {
  if (!(mat->super.props & BF_MAT_PROPS_VIEW)) {
    bfMemFree(mat->rowptr);
    bfMemFree(mat->colind);
    bfMemFree(mat->data);
  }

  mat->rowptr = NULL;
  mat->colind = NULL;
  mat->data = NULL;

  bfMatDeinit(&mat->super);
}

void bfMatCsrRealDealloc(BfMatCsrReal **mat) {
  bfMemFree(*mat);
  *mat = NULL;
}

void bfMatCsrRealDeinitAndDealloc(BfMatCsrReal **mat) {
  bfMatCsrRealDeinit(*mat);
  bfMatCsrRealDealloc(mat);
}

void bfMatCsrRealDump(BfMatCsrReal const *matCsrReal, char const *rowptrPath,
                      char const *colindPath, char const *dataPath) {
  BF_ERROR_BEGIN();

  BfSize numRows = matCsrReal->super.numRows;
  BfSize nnz = matCsrReal->rowptr[numRows];

  FILE *fp;

  /* Save `rowptr` to disk: */

  fp = fopen(rowptrPath, "w");
  if (fp == NULL)
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);
  fwrite(matCsrReal->rowptr, sizeof(BfSize), numRows + 1, fp);
  fclose(fp);

  /* Save `colind` to disk: */

  fp = fopen(colindPath, "w");
  if (fp == NULL)
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);
  fwrite(matCsrReal->colind, sizeof(BfSize), nnz, fp);
  fclose(fp);

  /* Save `data` to disk: */

  fp = fopen(dataPath, "w");
  if (fp == NULL)
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);
  fwrite(matCsrReal->data, sizeof(BfReal), nnz, fp);
  fclose(fp);

  BF_ERROR_END() {
    // TODO: delete all files
    fclose(fp);
  }
}

bool bfMatCsrRealHasSameSparsityPattern(BfMatCsrReal const *matCsrReal, BfMatCsrReal const *otherMatCsrReal) {
  BfMat const *mat = &matCsrReal->super;
  BfMat const *otherMat = &otherMatCsrReal->super;

  BfSize m = bfMatGetNumRows(mat);
  if (m != bfMatGetNumRows(otherMat))
    return false;

  if (bfMatGetNumCols(mat) != bfMatGetNumCols(otherMat))
    return false;

  for (BfSize i = 0; i < m; ++i)
    if (matCsrReal->rowptr[i] != otherMatCsrReal->rowptr[i])
      return false;

  BfSize nnz = matCsrReal->rowptr[m];
  if (nnz != otherMatCsrReal->rowptr[m])
    return false;

  for (BfSize i = 0; i < nnz; ++i)
    if (matCsrReal->colind[i] != otherMatCsrReal->colind[i])
      return false;

  return true;
}
