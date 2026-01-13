#include <bf/linalg.h>

#include <math.h>
#include <float.h>
#include <limits.h>

#include <bf/assert.h>
#include <bf/const.h>
#include <bf/disjoint_interval_list.h>
#include <bf/error.h>
#include <bf/error_macros.h>
#include <bf/logging.h>
#include <bf/lu_csr_real.h>
#include <bf/mat_dense_real.h>
#include <bf/mem.h>
#include <bf/ptr_array.h>
#include <bf/real_array.h>
#include <bf/timer.h>
#include <bf/util.h>
#include <bf/vec_real.h>
#include <bf/mat_csr_real.h>

#if BF_DEBUG
#include <bf/vec_complex.h>
#endif

#include <arpack.h>

#include <stdio.h>  /* make sure this is present near the top of the file */

#ifndef BF_HAVE_PRIMME_SVDS
#define BF_HAVE_PRIMME_SVDS 1
#endif

#if BF_HAVE_PRIMME_SVDS
#  include <primme.h>
#  include <primme_svds.h>
#endif

/* ARPACK backend assumes BfReal is double precision */
_Static_assert(sizeof(BfReal) == sizeof(double),
               "ARPACK and PRIMME backends require BfReal == double");

/* ---- CSR SVD debug logging ----------------------------------------- */

#ifndef BF_SPARSE_SVD_DEBUG
#define BF_SPARSE_SVD_DEBUG 0
#endif

#if BF_SPARSE_SVD_DEBUG
  #define SPARSE_SVD_LOG(fmt, ...)                                           \
    do {                                                                     \
      fprintf(stderr, "[bf][sparse_svd] " fmt, ##__VA_ARGS__);               \
      fflush(stderr);                                                        \
    } while (0)
#else
  #define SPARSE_SVD_LOG(...) do {} while (0)
#endif

#if BF_HAVE_PRIMME_SVDS
static int bfSparseSvdsMonitorWarned = 0;
static long long gPrimmeCsrMatvecCalls_A  = 0;
static long long gPrimmeCsrMatvecCalls_AT = 0;
/* Max wall-clock time per PRIMME_SVDS call, in seconds.
 * If <= 0, no time-based stopping is applied.
 */
static const double BF_PRIMME_SVDS_MAX_WALLTIME = 1.0;  /* e.g. 1 second per block */

static void bfPrimmeSvdsCappedMonitor(
    void *basisSvals, int *basisSize, int *basisFlags,
    int *iblock, int *blockSize, void *basisNorms, int *numConverged,
    void *lockedSvals, int *numLocked, int *lockedFlags, void *lockedNorms,
    int *inner_its, void *LSRes, const char *msg, double *time,
    primme_event *event, int *stage,
    primme_svds_params *primme_svds, int *err)
{
  (void)basisSvals;
  (void)basisSize;
  (void)basisFlags;
  (void)iblock;
  (void)blockSize;
  (void)basisNorms;
  (void)numConverged;
  (void)lockedSvals;
  (void)numLocked;
  (void)lockedFlags;
  (void)lockedNorms;
  (void)inner_its;
  (void)LSRes;
  (void)msg;
  (void)time;
  (void)event;
  (void)stage;

  /* IMPORTANT: never signal error via *err; that triggers an assert in PRIMME.
   * We only steer via maxMatvecs, as suggested in primme#51.
   */
  if (err) *err = 0;

  if (primme_svds == NULL)
    return;

  /* 1) Respect the normal maxMatvecs cap (optional log) */
  if (primme_svds->maxMatvecs > 0 &&
      primme_svds->stats.numMatvecs >= primme_svds->maxMatvecs &&
      !bfSparseSvdsMonitorWarned) {

    SPARSE_SVD_LOG(
      "[bf] sparse SVD: PRIMME monitor: hit maxMatvecs=%lld (numMatvecs=%lld)\n",
      (long long)primme_svds->maxMatvecs,
      (long long)primme_svds->stats.numMatvecs);

    bfSparseSvdsMonitorWarned = 1;
  }

  /* 2) Time-based abort: when elapsedTime > BF_PRIMME_SVDS_MAX_WALLTIME,
   *    force maxMatvecs=0 so PRIMME exits with PRIMME_MAIN_ITER_FAILURE (-3).
   *
   *    This is exactly the pattern suggested in:
   *      https://github.com/primme/primme/issues/51
   */
  if (BF_PRIMME_SVDS_MAX_WALLTIME > 0.0 &&
      primme_svds->stats.elapsedTime > BF_PRIMME_SVDS_MAX_WALLTIME &&
      primme_svds->maxMatvecs != 0) {

    SPARSE_SVD_LOG(
      "[bf] sparse SVD: PRIMME monitor: elapsedTime=%.3e > %.3e s, "
      "forcing maxMatvecs=0 to abort\n",
      primme_svds->stats.elapsedTime,
      BF_PRIMME_SVDS_MAX_WALLTIME);

    /* Outer SVDS matvec budget */
    primme_svds->maxMatvecs = 0;

    /* Inner eigensolvers as well, for safety */
    primme_svds->primme.maxMatvecs       = 0;
    primme_svds->primmeStage2.maxMatvecs = 0;
  }
}
#endif

static bool bfIsFiniteReal(BfReal x) {
  return isfinite(x);
}

/* ---- Sparse SVD accept/reject heuristics ---------------------------- */

/* Keep CSR if SVD isn't at least this fraction smaller (0.9 = save >=10%) */
#ifndef BF_SPARSE_SVD_MIN_SAVINGS_FRAC
#define BF_SPARSE_SVD_MIN_SAVINGS_FRAC 0.90
#endif

/* Enable a simple nonnegativity/physics probe for nonnegative operators */
#ifndef BF_SPARSE_SVD_PHYSICS_PROBE
#define BF_SPARSE_SVD_PHYSICS_PROBE 1
#endif

/* Allowed negativity magnitude relative to p99 scale, as a function of tol */
static double bfSparseSvdNegEtaFromTol(BfTruncSpec const *truncSpec) {
  if (truncSpec && truncSpec->usingTol) {
    double t = (double)truncSpec->tol;
    /* You can tune these. Conservative defaults: */
    if (t >= 1.0)  return 5e-2;
    if (t >= 0.1)  return 1e-2;
    if (t >= 0.01) return 5e-3;
    return 1e-3;
  }
  return 1e-2;
}

static double bfSparseSvdEstimateBytesCsr(BfSize m, BfSize nnz) {
  /* CSR: data[nnz] + colind[nnz] + rowptr[m+1] */
  return (double)nnz * ((double)sizeof(BfReal) + (double)sizeof(BfSize)) +
         (double)(m + 1) * (double)sizeof(BfSize);
}

static double bfSparseSvdEstimateBytesSvd(BfSize m, BfSize n, BfSize k) {
  /* Dense U(m×k) + dense VT(k×n) + diag S(k) */
  return ((double)m * (double)k + (double)n * (double)k + (double)k) *
         (double)sizeof(BfReal);
}

/* Compute p-quantile of y (0<=p<=1). Uses full sort via bfRealArgsort. */
static BfReal bfSparseSvdQuantile(BfReal const *y, BfSize n, double p) {
  if (n == 0) return 0;
  if (p <= 0) return y[0];
  if (p >= 1) return y[n - 1];

  BfSize *perm = bfMemAlloc(n, sizeof(BfSize));
  if (perm == NULL) return y[n - 1]; /* fallback */
  bfRealArgsort(y, n, perm);

  /* index in [0, n-1] */
  double idx_f = p * (double)(n - 1);
  BfSize idx = (BfSize)floor(idx_f + 0.5);
  if (idx >= n) idx = n - 1;

  BfReal q = y[perm[idx]];
  bfMemFree(perm);
  return q;
}

/* Forward declaration for CSR+ARPACK SVD helper */
static bool bfMatCsrRealSparseTruncatedSvd(BfMatCsrReal const *Acsr,
                                           BfSize              maxRank,
                                           BfTruncSpec const  *truncSpec,
                                           BfMat             **UPtr,
                                           BfMatDiagReal     **SPtr,
                                           BfMat             **VTPtr);

BfSize bfTruncSpecGetNumTerms(BfTruncSpec const *truncSpec, BfMatDiagReal const *S) {
  BfSize k = 0;

  if (S == NULL || S->numElts == 0)
    return 0;

  if (!bfIsFiniteReal(S->data[0]) || S->data[0] <= 0)
    return 0;

  if (truncSpec->usingTol) {
    while (k < S->numElts &&
           bfIsFiniteReal(S->data[k]) &&
           S->data[k] >= truncSpec->tol*S->data[0])
      ++k;
  } else {
    bfSetError(BF_ERROR_NOT_IMPLEMENTED);
  }
  return k;
}

/* Solve the linear system with matrix `A`, RHS `B`, and initial guess
 * `X0`. Tolerance and maximum number of iterations specified by `tol`
 * and `maxNumIter`, respectively. The final iteration count will be
 * passed back in `numIter` if `numIter != NULL`.
 *
 * For left preconditioned GMRES, set the preconditioner in `M`.  Note
 * that the residual will be determined from the *preconditioned*
 * residual vectors. See Saad for more details.
 *
 * TODO: right and split preconditioned GMRES. */
BfMat *bfSolveGMRES(BfMat const *A, BfMat const *B, BfMat *X0,
                    BfReal tol, BfSize maxNumIter, BfSize *numIter,
                    BfMat const *M) {
  BF_ERROR_BEGIN();

  /* Solution of the system */
  BfMat *X = NULL;

  /* Intermediate matrices */
  BfMat *R = NULL;
  BfMat **V = NULL;
  BfMat **H = NULL;
  BfMat *S = NULL;

  /* Givens rotations for solving the least squares problem */
  BfMat **J = NULL;

  bool converged = false;

  /* Make sure B is a dense matrix */
  if (!bfMatInstanceOf(B, BF_TYPE_MAT_DENSE_REAL) &&
      !bfMatInstanceOf(B, BF_TYPE_MAT_DENSE_COMPLEX))
    RAISE_ERROR(BF_ERROR_TYPE_ERROR);

  /* Make sure that either X0 wasn't passed or is a dense matrix */
  if (X0 != NULL)
    if (!bfMatInstanceOf(X0, BF_TYPE_MAT_DENSE_REAL) &&
        !bfMatInstanceOf(X0, BF_TYPE_MAT_DENSE_COMPLEX))
      RAISE_ERROR(BF_ERROR_TYPE_ERROR);

  /* Make sure m is positive */
  if (maxNumIter == 0)
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  /* Get order of system (n) and check compatibility */
  BfSize n = bfMatGetNumRows(A);
  if (n != bfMatGetNumCols(A))
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);
  if (n != bfMatGetNumRows(B))
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);
  if (X0 != NULL && n != bfMatGetNumRows(X0))
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  bool leftPrecond = M != NULL;

  /* If we're using a left preconditioner, make sure it has a
   * compatible shape. */
  if (leftPrecond) {
    if (n != bfMatGetNumRows(M))
      RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);
    if (n != bfMatGetNumCols(M))
      RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);
  }

  /* Get number of RHSs (P) and check compatibility */
  BfSize numRhs = bfMatGetNumCols(B);
  if (X0 != NULL && numRhs != bfMatGetNumCols(X0))
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  V = bfMemAllocAndZero(maxNumIter + 1, sizeof(BfMatDenseComplex *));
  if (V == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  H = bfMemAllocAndZero(maxNumIter, sizeof(BfMatDenseComplex *));
  if (H == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  J = bfMemAllocAndZero(numRhs*maxNumIter, sizeof(BfMat *));
  if (J == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  /* If an initial iterate wasn't passed, set it to zero */
  if (X0 == NULL) {
    X0 = bfMatZerosLike(B, n, numRhs);
    HANDLE_ERROR();
  }

  /* Compute residual for first iteration */
  BfMat *Y = bfMatMul(A, X0);
  HANDLE_ERROR();
  R = bfMatSub(B, Y);
  HANDLE_ERROR();
  bfMatDelete(&Y);
  if (leftPrecond) {
    BfMat *_ = bfMatSolve(M, R);
    bfMatDelete(&R);
    R = _;
  }

  /* Compute the norm of the residual for each righthand side */
  BfVec *RNorm = bfMatColNorms(R);

  // TODO: beta is different for each RHS
  BfReal beta = bfVecNormMax(RNorm);

  /* Set the first basis vector to the normalized residual */
  V[0] = bfMatCopy(R);
  bfMatDivideCols(V[0], RNorm);

  /* Initialize S to ||r||_2*e1---need to take the reciprocal of
   * `R_col_norms` again first */
  S = bfMatZerosLike(B, maxNumIter + 1, numRhs);
  bfMatSetRow(S, 0, RNorm);

  /* The iteration count/the current column: */
  BfSize j = 0;

  for (j = 0; j < maxNumIter; ++j) {
    BfMat *W = bfMatMul(A, V[j]);

    /* Apply left preconditioner if we're using one: */
    if (leftPrecond) {
      BfMat *_ = bfMatSolve(M, W);
      bfMatDelete(&W);
      W = _;
    }

    /** Run modified Gram-Schmidt: */

    /* Compute W column norms before applying Gram-Schmidt: */
    BfVec *WNormBefore = bfMatColNorms(W); // 433

    /* Allocate space for next column of H: */
    H[j] = bfMatEmptyLike(W, j + 2, numRhs);

    for (BfSize i = 0; i <= j; ++i) {
      BfVec *Hij = bfMatColDots(V[i], W); // 435
      bfMatSetRow(H[j], i, Hij);          // "

      BfMat *Hij_Vi = bfMatCopy(V[i]);
      bfMatScaleCols(Hij_Vi, Hij); // 436
      bfMatSubInplace(W, Hij_Vi);  // "

      bfMatDelete(&Hij_Vi);
      bfVecDelete(&Hij);
    }

    BfVec *WNorm = bfMatColNorms(W); // 438
    bfMatSetRow(H[j], j + 1, WNorm); // 439

#if BF_DEBUG // TODO: deal with breakdown
    {
      BF_ASSERT(numRhs == 1); // TODO: handle numRhs > 1
      BfReal _ = bfVecNormMax(WNorm)/bfVecNormMax(WNormBefore);
      BF_ASSERT(_ >= tol);
    }
#endif

    V[j + 1] = bfMatCopy(W); // 440
    bfMatDivideCols(V[j + 1], WNorm); // 447 & 448

    bfMatDelete(&W);
    bfVecDelete(&WNorm);
    bfVecDelete(&WNormBefore);

    /** Use Givens rotations to reduce H to upper triangular form: */

    for (BfSize i = 0; i < j; ++i) {
      for (BfSize p = 0; p < numRhs; ++p) {
        BfVec *h = bfMatGetColRangeView(H[j], 0, i + 2, p);
        bfVecMulInplace(h, J[numRhs*i + p]);
        bfVecDelete(&h);
      }
    }

    for (BfSize p = 0; p < numRhs; ++p) {
      BfVec *h = bfMatGetColView(H[j], p);
      J[numRhs*j + p] = bfVecGetGivensRotation(h, j, j + 1);
      bfVecMulInplace(h, J[numRhs*j + p]);
      bfVecDelete(&h);
    }

    /* Apply most recent set of Givens rotations to S */
    for (BfSize p = 0; p < numRhs; ++p) {
      BfVec *s = bfMatGetColView(S, p);
      BfVec *sSub = bfVecGetSubvecView(s, 0, j + 2);
      bfVecMulInplace(sSub, J[numRhs*j + p]);
      bfVecDelete(&sSub);
      bfVecDelete(&s);
    }

    BfVec *sLastRow = bfMatGetRowView(S, j + 1);
    BfReal residual = bfVecNormMax(sLastRow)/beta;

    /** Deal with convergence: */

    if (residual < tol)
      converged = true;

    bfVecDelete(&sLastRow);

    if (converged)
      break;
  }

  /* Construct the solution */
  X = bfMatEmptyLike(B, n, numRhs);
  for (BfSize p = 0; p < numRhs; ++p) {
    /* Extract the triangularized version of the upper Hessenberg
     * matrix H for the current RHS: */
    BfMat *Hp = bfMatZerosLike(H[0], j, j);
    for (BfSize i = 0; i < j; ++i) {
      BfVec *h = bfMatGetColView(H[i], p);
#if BF_DEBUG
      if (bfVecGetType(h) == BF_TYPE_VEC_COMPLEX) {
        BfVecComplex *_ = bfVecToVecComplex(h);
        BfComplex z = *(_->data + (i + 1)*_->stride);
        BF_ASSERT(cabs(z) <= 1e-15);
      }
#endif
      bfMatSetColRange(Hp, i, 0, i + 1, h);
      bfVecDelete(&h);
    }

    /* Extract the version of V for this RHS */
    BfMat *Vp = bfMatEmptyLike(V[0], n, j);
    for (BfSize i = 0; i < j; ++i) {
      BfVec *v = bfMatGetColView(V[i], p);
      bfMatSetCol(Vp, i, v);
      bfVecDelete(&v);
    }

    /* Solve for the coefficients representing x - x0 in the V-basis */
    BfVec *s = bfMatGetColRangeView(S, 0, j, p);
    BfVec *y = bfMatBackwardSolveVec(Hp, s);
    BfVec *x0 = bfMatGetColView(X0, p);
    BfVec *x = bfMatMulVec(Vp, y);
    bfVecAddInplace(x, x0);
    bfMatSetCol(X, p, x);

    bfMatDelete(&Hp);
    bfMatDelete(&Vp);
    bfVecDelete(&s);
    bfVecDelete(&y);
    bfVecDelete(&x0);
    bfVecDelete(&x);
  }

  if (numIter != NULL)
    *numIter = j;

  BF_ERROR_END() {
    BF_DIE();
  }

  for (BfSize i = 0; i < maxNumIter + 1; ++i)
    if (V[i] != NULL)
      bfMatDelete(&V[i]);

  for (BfSize i = 0; i < maxNumIter; ++i)
    if (H[i] != NULL)
      bfMatDelete(&H[i]);

  for (BfSize i = 0; i < numRhs*maxNumIter; ++i)
    if (J[i] != NULL)
      bfMatDelete(&J[i]);

  bfVecDelete(&RNorm);
  bfMatDelete(&R);

  bfMemFree(V);
  bfMemFree(H);
  bfMemFree(J);

  bfMatDelete(&S);

  if (X0 != NULL)
    bfMatDelete(&X0);

  return X;
}

static BfSize estimateNcv(BfSize nev, BfSize N) {
  BfSize ncv = 2*nev + 1;
  if (ncv < 20)
    ncv = 20;
  if (ncv > N)
    ncv = N;
  return ncv;
}

BfReal bfGetMaxEigenvalue(BfMat const *L, BfMat const *M) {
  /* TODO: this is a work in progress! This does NOT work for any type
   * of BfMat yet. Just real ones... */

  BF_ERROR_BEGIN();

  BfTimer timerFunc;
  bfTimerReset(&timerFunc);

  double *resid = NULL;
  double *V = NULL;
  a_int *select = NULL;
  double *workd = NULL;
  double *workl = NULL;
  BfReal *workev = NULL;

  a_int const N = bfMatGetNumRows(L);
  char const which[] = "LM";
  a_int const nev = 1;
  a_int const ncv = estimateNcv(nev, N);
  char bmat = 'G'; /* Solve (G)eneralized eigenvalue problem */
  BfReal const tol = 0;
  a_int const ldv = N;
  a_int const lworkl = 3*ncv*ncv + 6*ncv;
  a_int const rvec = 0; /* only computing eigenvalues */
  char const howmny[] = "A";

  BfLuCsrReal *M_lu = bfLuCsrRealNew();
  HANDLE_ERROR();

  bfLuCsrRealInit(M_lu, M);
  HANDLE_ERROR();

  resid = bfMemAlloc(N, sizeof(BfReal));
  if (resid == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  V = bfMemAlloc(N, ncv*sizeof(BfReal));
  if (V == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  select = bfMemAllocAndZero(ncv, sizeof(a_int));
  HANDLE_ERROR();

  a_int iparam[11] = {
    [0] = 1, /* compute exact shifts */
    [2] = 10*N, /* max number of iterations */
    // [3] = 1, /* only value allowed */
    [6] = 2, /* mode: A*x = lam*M*x */
  };

  a_int ipntr[11];

  workd = bfMemAllocAndZero(3*N, sizeof(BfReal));
  HANDLE_ERROR();

  workl = bfMemAlloc(lworkl, sizeof(BfReal));
  HANDLE_ERROR();

  workev = bfMemAlloc(3*ncv, sizeof(BfReal));
  HANDLE_ERROR();

  a_int ido = 0;
  a_int info = 0;

dnaupd:
  dnaupd_c(&ido, &bmat, N, which, nev, tol, resid, ncv, V, ldv, iparam, ipntr,
           workd, workl, lworkl, &info);

  if (ido == 1 || ido == -1) {
    BF_ASSERT(ipntr[0] > 0);
    BF_ASSERT(ipntr[1] > 0);

    BfVecReal x;
    bfVecRealInitView(&x, N, BF_DEFAULT_STRIDE, &workd[ipntr[0] - 1]);
    BfVec *tmp = bfMatMulVec(L, bfVecRealToVec(&x));
    BfVecReal *y = bfVecToVecReal(bfLuCsrRealSolveVec(M_lu, tmp));

    bfMemCopy(y->data, N, sizeof(BfReal), &workd[ipntr[1] - 1]);

    bfVecDelete(&tmp);
    bfVecRealDeinitAndDealloc(&y);

    goto dnaupd;
  } else if (ido == 2) {
    BF_ASSERT(ipntr[0] > 0);
    BF_ASSERT(ipntr[1] > 0);

    BfVecReal x;
    bfVecRealInitView(&x, N, BF_DEFAULT_STRIDE, &workd[ipntr[0] - 1]);

    BfVecReal *y = bfVecToVecReal(bfMatMulVec(M, bfVecRealToVec(&x)));

    bfMemCopy(y->data, N, sizeof(BfReal), &workd[ipntr[1] - 1]);

    bfVecRealDeinitAndDealloc(&y);

    goto dnaupd;
  }

  if (info < 0 || iparam[4] < nev)
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);

  BfReal dr[2], di[2];

  dneupd_c(
    rvec, /* == 0 -> not computing Ritz vectors */
    howmny, /* == "A" -> compute all requested eigenvalues */
    select, /* used as internal workspace since howmny == "A" */
    dr, /* will contain real part of first nev + 1 eigenvalues */
    di,       /* ... imaginary part ... */
    NULL, /* not referenced since rvec == 0 */
    0,              /* ditto */
    0.0, /* not referenced since mode == 2 */
    0.0,           /* ditto */
    workev, /* internal workspace */

    /* dnaupd parameters: don't modify before calling dseupd */
    &bmat, N, which, nev, tol, resid, ncv, V, ldv, iparam, ipntr,
    workd, workl, lworkl, &info);

  if (info != 0)
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);

  BfReal eigmax = dr[0];

  if (fabs(di[0]) > 0)
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);

  BF_ERROR_END() {
    eigmax = NAN;
  }

  bfLuCsrRealDeinitAndDealloc(&M_lu);

  bfMemFree(resid);
  bfMemFree(V);
  bfMemFree(select);
  bfMemFree(workd);
  bfMemFree(workl);
  bfMemFree(workev);

  return eigmax;
}

void bfGetShiftedEigs(BfMat const *A, BfMat const *M, BfReal sigma, BfSize k,
                      BfMat **PhiTransposePtr, BfVecReal **LambdaPtr) {
  /* TODO: this is a work in progress! This does NOT work for any type
   * of BfMat yet. Just real ones... */

  BF_ERROR_BEGIN();

  if (!isfinite(sigma))
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  BfTimer timerFunc;
  bfTimerReset(&timerFunc);

  double *resid = NULL;
  double *V = NULL;
  a_int *select = NULL;
  double *workd = NULL;
  double *workl = NULL;
  BfReal *workev = NULL;
  BfReal *dr = NULL;
  BfReal *di = NULL;
  BfReal *z = NULL;

  BfSize *J = NULL;
  BfVecReal *Lambda = NULL;

  a_int const N = bfMatGetNumRows(A);
  char const which[] = "LM";
  a_int const nev = k;
  a_int const ncv = estimateNcv(nev, N);
  char bmat = 'G'; /* Solve (G)eneralized eigenvalue problem */
  BfReal const tol = 0;
  a_int const ldv = N;
  a_int const lworkl = 3*ncv*ncv + 6*ncv;
  a_int const rvec = 1; /* computing eigenvalues and eigenvectors */
  char const howmny[] = "A";

  BfTimer timer;

  bfTimerReset(&timer);

  BfMat *A_minus_sigma_M = bfMatCopy(M);
  HANDLE_ERROR();

  bfMatScale(A_minus_sigma_M, -sigma);
  HANDLE_ERROR();

  bfMatAddInplace(A_minus_sigma_M, A);
  HANDLE_ERROR();

  BfLuCsrReal *A_minus_sigma_M_lu = bfLuCsrRealNew();
  HANDLE_ERROR();

  bfLuCsrRealInit(A_minus_sigma_M_lu, A_minus_sigma_M);
  HANDLE_ERROR();

  bfLogInfo("bfGetShiftedEigs: created and factorized A - sigma*M [%.1fs]\n",
            bfTimerGetElapsedTimeInSeconds(&timer));

  resid = bfMemAlloc(N, sizeof(BfReal));
  if (resid == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  V = bfMemAlloc(N*ncv, sizeof(BfReal));
  if (V == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  select = bfMemAllocAndZero(ncv, sizeof(a_int));
  HANDLE_ERROR();

  a_int iparam[11] = {
    [0] = 1, /* compute exact shifts */
    [2] = 10*N, /* max number of iterations */
    [6] = 3, /* shift-invert mode */
  };

  a_int ipntr[11];

  workd = bfMemAlloc(3*N, sizeof(BfReal));
  if (workd == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  for (a_int i = 0; i < 3*N; ++i)
    workd[i] = 0;

  workl = bfMemAlloc(lworkl, sizeof(BfReal));
  if (workl == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  workev = bfMemAlloc(3*ncv, sizeof(BfReal));
  if (workev == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  a_int ido = 0;
  a_int info = 0;

  BfSize numSolve = 0;
  BfSize numMvp = 0;

  BfReal totalSolveTime = 0;
  BfReal totalMvpTime = 0;

dnaupd:
  dnaupd_c(&ido, &bmat, N, which, nev, tol, resid, ncv, V, ldv, iparam, ipntr,
           workd, workl, lworkl, &info);
  if (ido == 1 || ido == -1) {
    BF_ASSERT(ipntr[0] > 0);
    BF_ASSERT(ipntr[1] > 0);

    bfTimerReset(&timer);

    BfVecReal x;
    bfVecRealInitView(&x, N, BF_DEFAULT_STRIDE, &workd[ipntr[0] - 1]);
    BfVec *tmp = bfMatMulVec(M, bfVecRealToVec(&x));
    BfVecReal *y = bfVecToVecReal(bfLuCsrRealSolveVec(A_minus_sigma_M_lu, tmp));

    totalSolveTime += bfTimerGetElapsedTimeInSeconds(&timer);

    ++numSolve;

    bfMemCopy(y->data, N, sizeof(BfReal), &workd[ipntr[1] - 1]);

    bfVecDelete(&tmp);
    bfVecRealDeinitAndDealloc(&y);

    goto dnaupd;
  } else if (ido == 2) {
    BF_ASSERT(ipntr[0] > 0);
    BF_ASSERT(ipntr[1] > 0);

    BfVecReal x;
    bfVecRealInitView(&x, N, BF_DEFAULT_STRIDE, &workd[ipntr[0] - 1]);

    bfTimerReset(&timer);

    BfVecReal *y = bfVecToVecReal(bfMatMulVec(M, bfVecRealToVec(&x)));

    totalMvpTime += bfTimerGetElapsedTimeInSeconds(&timer);

    ++numMvp;

    bfMemCopy(y->data, N, sizeof(BfReal), &workd[ipntr[1] - 1]);

    bfVecRealDeinitAndDealloc(&y);

    goto dnaupd;
  }

  bfLogInfo("bfGetShiftedEigs: finished iterating [%.1fs]\n",
            bfTimerGetElapsedTimeInSeconds(&timer));
  bfLogInfo("bfGetShiftedEigs: did %lu solves [%.1fs]\n", numSolve, totalSolveTime);
  bfLogInfo("bfGetShiftedEigs: did %lu MVPs [%.1fs]\n", numMvp, totalMvpTime);

  if (info < 0 || iparam[4] < nev)
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);

  dr = bfMemAlloc(nev + 1, sizeof(BfReal));
  if (dr == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  di = bfMemAlloc(nev + 1, sizeof(BfReal));
  if (di == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  z = bfMemAlloc(N*(nev + 1), sizeof(BfReal));
  if (z == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  BfSize ldz = N;

  BfReal sigmar = sigma;
  BfReal sigmai = 0.0;

  bfTimerReset(&timer);

  dneupd_c(
    rvec, /* == 0 -> not computing Ritz vectors */
    howmny, /* == "A" -> compute all requested eigenvalues */
    select, /* used as internal workspace since howmny == "A" */
    dr, /* will contain real part of first nev + 1 eigenvalues */
    di,       /* ... imaginary part ... */
    z,
    ldz,
    sigmar,
    sigmai,
    workev, /* internal workspace */

    /* dnaupd parameters: don't modify before calling dseupd */
    &bmat, N, which, nev, tol, resid, ncv, V, ldv, iparam, ipntr,
    workd, workl, lworkl, &info);

  bfLogInfo("bfGetShiftedEigs: extracted eigs [%.1fs]\n",
            bfTimerGetElapsedTimeInSeconds(&timer));

  if (info != 0)
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);

  /* TODO: evals could obviously be complex but right now we're only
   * interested in real evals and haven't introduced any other
   * safeguards to deal with complex case yet. So just treat complex
   * eigenvalues as an error for now. */
  for (BfSize i = 0; i < k; ++i)
    if (fabs(di[i]) > 1e-12)
      RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);

  J = bfMemAlloc(k, sizeof(BfSize));
  if (J == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  bfRealArgsort(dr, k, J);

  if (LambdaPtr == NULL)
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  if (*LambdaPtr != NULL)
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  Lambda = bfVecRealNew();
  HANDLE_ERROR();

  bfVecRealInit(Lambda, k);
  HANDLE_ERROR();

  for (BfSize j = 0; j < k; ++j)
    *(Lambda->data + J[j]*Lambda->stride) = dr[j];

  if (PhiTransposePtr == NULL)
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  if (*PhiTransposePtr != NULL)
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  BfMatDenseReal *PhiTranspose = bfMatDenseRealNew();
  HANDLE_ERROR();

  bfMatDenseRealInit(PhiTranspose, k, N);
  HANDLE_ERROR();

  for (BfSize j = 0; j < k; ++j) {
    BfVecReal *phi = bfVecRealNew();
    HANDLE_ERROR();

    bfVecRealInitView(phi, N, BF_DEFAULT_STRIDE, z + N*j);
    HANDLE_ERROR();

    bfMatDenseRealSetRow(bfMatDenseRealToMat(PhiTranspose), J[j], bfVecRealToVec(phi));

    bfVecRealDeinitAndDealloc(&phi);
  }

  *LambdaPtr = Lambda;
  *PhiTransposePtr = bfMatDenseRealToMat(PhiTranspose);

  BF_ERROR_END() {
    BF_DIE();
  }

  bfMemFree(J);
  bfMemFree(z);
  bfMemFree(dr);
  bfMemFree(di);

  bfLuCsrRealDeinitAndDealloc(&A_minus_sigma_M_lu);
  bfMatDelete(&A_minus_sigma_M);

  bfMemFree(resid);
  bfMemFree(V);
  bfMemFree(select);
  bfMemFree(workd);
  bfMemFree(workl);
  bfMemFree(workev);

  bfLogInfo("bfGetShiftedEigs: done [%.1fs]\n",
            bfTimerGetElapsedTimeInSeconds(&timerFunc));
}

static void getEigenband_doubling(BfMat const *A, BfMat const *M, BfInterval const *interval,
                                  BfMat **PhiTransposePtr, BfVecReal **LambdaPtr) {
  BF_ERROR_BEGIN();

  BfMat *PhiTranspose = NULL;
  BfVecReal *Lambda = NULL;

  BfSize k = 8;

  BfReal const sigma = bfIntervalIsFinite(interval) ?
    bfIntervalGetMidpoint(interval) :
    bfIntervalGetFiniteEndpoint(interval);

get_shifted_eigs:
  bfLogInfo("bfGetEigenband: k = %lu\n", k);

  bfGetShiftedEigs(A, M, sigma, k, &PhiTranspose, &Lambda);
  HANDLE_ERROR();

  BfReal const *lam = Lambda->data;

  /* Keep doubling the number of eigenvalues until we cover `interval`
   * with `currentInterval`: */
  BfInterval currentInterval = {
    .endpoint = {lam[0], lam[k - 1]},
    .closed = {true, true}
  };
  if (!bfIntervalContainsInterval(&currentInterval, interval)) {
    k *= 2;
    bfMatDelete(&PhiTranspose);
    bfVecRealDeinitAndDealloc(&Lambda);
    goto get_shifted_eigs;
  }

  /* Find the first eigenpair in the band */
  BfSize j0 = 0;
  while (j0 < k && !bfIntervalContainsPoint(interval, lam[j0])) ++j0;

  /* Find the last eigenpair in the band */
  BfSize j1 = k;
  while (j1 > j0 && !bfIntervalContainsPoint(interval, lam[j1 - 1])) --j1;

  if (0 < j0 || j1 < k) {
    /* Prune unnecessary eigenvectors */
    BfMat *oldPhiTranspose = PhiTranspose;
    PhiTranspose = bfMatGetRowRangeCopy(oldPhiTranspose, j0, j1);
    HANDLE_ERROR();

    /* Prune unnecessary eigenvalues */
    BfVecReal *oldLambda = Lambda;
    Lambda = bfVecToVecReal(bfVecGetSubvecCopy(bfVecRealToVec(oldLambda), j0, j1));
    HANDLE_ERROR();

    /* Free old eigenvectors and values */
    bfMatDelete(&oldPhiTranspose);
    bfVecRealDeinitAndDealloc(&oldLambda);
  }

  *PhiTransposePtr = PhiTranspose;
  *LambdaPtr = Lambda;

  BF_ERROR_END() {
    bfMatDelete(&PhiTranspose);
    bfVecRealDeinitAndDealloc(&Lambda);

    *PhiTransposePtr = NULL;
    *LambdaPtr = NULL;
  }
}

static bool doublesHaveDistinctMidpoint(double lam0, double lam1) {
  double lam_mid = (lam0 + lam1)/2;
  return lam0 != lam_mid && lam_mid != lam1;
}

static double getSigmaForInterval(BfInterval const *interval) {
  if (bfIntervalIsFinite(interval)) {
    return bfIntervalGetMidpoint(interval);
  } else if (bfIntervalHasFiniteEndpoint(interval)) {
    return bfIntervalGetFiniteEndpoint(interval);
  } else {
    return 0;
  }
}

static BfInterval getPairsCoveringInterval(BfMat const *A, BfMat const *M, BfInterval const *interval, BfRealArray *LamData, BfRealArray *PhiTransposeData) {
  BF_ERROR_BEGIN();

  BfMat *coverPhiTranspose = NULL;
  BfVecReal *coverLam = NULL;
  BfReal const sigma = getSigmaForInterval(interval);

  /* If we don't have a finite gap between the two eigenvalues on the
   * edges of the covering interval, then we can't isolate the
   * interior eigenvalues. Although it's unlikely, we could have a
   * large number of repeated eigenvalues, which forces our hand: we
   * need to increase the number of eigenvalues we gather at this
   * point, otherwise we might miss some. This means that this
   * algorithm only has the desired asymptotic complexity if there are
   * at most O(1)---that is, with respect n, the size of A and
   * M---repeated eigenvalues.
   *
   * NOTE: the way we check whether the eigenvalues have a "finite
   * gap" is operational. We need more than just the pairs of edge
   * eigenvalues to be distinct, we need their *midpoint* lie between
   * them (in floating-point arithmetic). So we use that as the
   * test. See `doublesHaveDistinctMidpoint`. */
  BfSize const k0 = 8;
  BfSize k = k0;
recompute:
  bfGetShiftedEigs(A, M, sigma, k + 2, &coverPhiTranspose, &coverLam);
  HANDLE_ERROR();
  BfReal const *lam = coverLam->data;
  if (!doublesHaveDistinctMidpoint(lam[0], lam[1]) || !doublesHaveDistinctMidpoint(lam[k], lam[k + 1])) {
    bfLogWarn("increased k from %lu to %lu\n", k, 2*k);
    k *= 2;
    bfMatDelete(&coverPhiTranspose);
    bfVecRealDeinitAndDealloc(&coverLam);
    goto recompute;
  }

  BfSize i0 = 0;
  while (i0 < k + 2 && !bfIntervalContainsPoint(interval, lam[i0])) ++i0;

  BfSize i1 = k + 2;
  while (i1 > 0 && !bfIntervalContainsPoint(interval, lam[i1 - 1])) --i1;

  BfInterval cover = {.closed = {false, false}};

  if (i0 == 0 && i1 == k + 2) {
    cover.endpoint[0] = (lam[0] + lam[1])/2;
    cover.endpoint[1] = (lam[k] + lam[k + 1])/2;
    i0 = 1;
    i1 = k + 1;
  } else if (i0 == 0 && i1 == 1) {
    cover = *interval;
  } else if (i0 == k + 1 && i1 == k + 2) {
    cover = *interval;
  } else if (i0 == 0 && i1 < k + 2) {
    cover.endpoint[0] = (lam[0] + lam[1])/2;
    i0 = 1;
    cover.endpoint[1] = interval->endpoint[1];
    cover.closed[1] = true;
  } else if (0 < i0 && i1 == k + 2) {
    cover.endpoint[0] = interval->endpoint[0];
    cover.endpoint[1] = (lam[k] + lam[k + 1])/2;
    cover.closed[0] = true;
    i1 = k + 1;
  } else if (i1 < i0) {
    cover.endpoint[0] = BF_INFINITY;
    cover.endpoint[1] = -BF_INFINITY;
  } else if (0 < i0 && i1 < k + 2) {
    cover.endpoint[0] = interval->endpoint[0];
    cover.endpoint[1] = interval->endpoint[1];
    cover.closed[0] = true;
    cover.closed[1] = true;
  } else {
    BF_DIE();
  }

  for (BfSize i = i0; i < i1; ++i) {
    BfReal lam = bfVecRealGetElt(coverLam, i);
    BF_ASSERT(bfIntervalContainsPoint(interval, lam));
    BF_ASSERT(bfIntervalContainsPoint(&cover, lam));

    bfRealArrayAppend(LamData, lam);
    HANDLE_ERROR();

    BfVecReal *phi = bfVecToVecReal(bfMatGetRowView(coverPhiTranspose, i));
    HANDLE_ERROR();

    BfRealArray *phiArray = bfVecRealGetArrayView(phi);
    HANDLE_ERROR();

    bfRealArrayExtend(PhiTransposeData, phiArray);
    HANDLE_ERROR();

    bfVecRealDeinitAndDealloc(&phi);
    bfRealArrayDeinitAndDealloc(&phiArray);
  }

  BF_ERROR_END() {
    BF_DIE();
  }

  bfMatDelete(&coverPhiTranspose);
  bfVecRealDeinitAndDealloc(&coverLam);

  return cover;
}

static void getEigenband_covering(BfMat const *A, BfMat const *M, BfInterval const *interval,
                                  BfMat **PhiPtr, BfVecReal **LambdaPtr) {
  BF_ERROR_BEGIN();

  BfRealArray *evals = bfRealArrayNewWithDefaultCapacity();
  HANDLE_ERROR();

  BfRealArray *evecs = bfRealArrayNewWithDefaultCapacity();
  HANDLE_ERROR();

  BfDisjointIntervalList *intervals = bfDisjointIntervalListNewEmpty();
  HANDLE_ERROR();

  bfDisjointIntervalListAdd(intervals, interval);
  HANDLE_ERROR();

  while (!bfDisjointIntervalListIsEmpty(intervals)) {
    BfInterval const *nextInterval = bfDisjointIntervalListGetFirstPtrConst(intervals);

    BfInterval cover = getPairsCoveringInterval(A, M, nextInterval, evals, evecs);
    HANDLE_ERROR();

    bfDisjointIntervalListRemove(intervals, bfIntervalIsEmpty(&cover) ? nextInterval : &cover);
  }

  BfSize numRows = bfMatGetNumRows(A);

  BfSize numCols = bfRealArrayGetSize(evecs);
  BF_ASSERT(numCols % numRows == 0);
  numCols /= numRows;

  BF_ASSERT(numCols == bfRealArrayGetSize(evals));
  for (BfSize j = 0; j < numCols; ++j) {
    BfReal eval = bfRealArrayGetValue(evals, j);
    BF_ASSERT(bfIntervalContainsPoint(interval, eval));
  }

  BfMatDenseReal *Phi = bfMatDenseRealNew();
  HANDLE_ERROR();

  bfMatDenseRealInit(Phi, numRows, numCols);
  HANDLE_ERROR();

  for (BfSize j = 0; j < numCols; ++j) {
    BfVec *col = bfRealArrayGetSubvecView(evecs, j*numRows, (j + 1)*numRows);
    bfMatDenseRealSetCol(Phi, j, col);
    bfVecDelete(&col);
  }

  BfPerm *perm = bfRealArrayArgsort(evals);
  HANDLE_ERROR();

  BfVecReal *Lambda = bfVecRealNewFromRealArray(evals, BF_POLICY_STEAL);
  HANDLE_ERROR();

  bfRealArrayPermute(evals, perm);
  bfMatDenseRealPermuteCols(Phi, perm);

  *PhiPtr = bfMatDenseRealToMat(Phi);
  *LambdaPtr = Lambda;

  BF_ERROR_END() {
    BF_DIE();
  }

  bfDisjointIntervalListDeinitAndDealloc(&intervals);
}

void bfGetEigenband(BfMat const *A, BfMat const *M, BfInterval const *interval,
                    BfEigenbandMethod method, BfMat **PhiTransposePtr, BfVecReal **LambdaPtr) {
  BF_ERROR_BEGIN();

  if (PhiTransposePtr == NULL)
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  if (*PhiTransposePtr != NULL)
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  if (LambdaPtr == NULL)
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  if (*LambdaPtr != NULL)
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  if (method == BF_EIGENBAND_METHOD_DOUBLING) {
    getEigenband_doubling(A, M, interval, PhiTransposePtr, LambdaPtr);
    HANDLE_ERROR();
  }

  else if (method == BF_EIGENBAND_METHOD_COVERING) {
    getEigenband_covering(A, M, interval, PhiTransposePtr, LambdaPtr);
    HANDLE_ERROR();
  }

  else RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);

  BF_ERROR_END() {
    BF_DIE();
  }
}

bool bfGetTruncatedSvd(BfMat const *mat, BfMat **UPtr, BfMatDiagReal **SPtr, BfMat **VTPtr,
                       BfTruncSpec const *truncSpec, BfBackend backend) {
  BF_ERROR_BEGIN();

  bool truncated = true;

  if (backend == BF_BACKEND_LAPACK) {
    /* Existing dense LAPACK path (unchanged) */

    /* Cast to `MatDenseReal` if we can, otherwise we'll need to convert
     * and allocate later. */
    BfMatDenseReal const *matDenseReal = NULL;
    bool shouldDeleteMatDenseReal = false;
    if (bfMatGetType(mat) == BF_TYPE_MAT_DENSE_REAL) {
      matDenseReal = bfMatConstToMatDenseRealConst(mat);
    } else {
      matDenseReal = bfMatDenseRealNewFromMatrix(mat);
      HANDLE_ERROR();

      shouldDeleteMatDenseReal = true;
    }

    BfMatDenseReal *U = NULL;
    BfMatDiagReal *S = NULL;
    BfMatDenseReal *VT = NULL;
    bfMatDenseRealSvd(matDenseReal, &U, &S, &VT);
    HANDLE_ERROR();

    /* Find the number of terms in the truncated SVD: */
    BfSize k = bfTruncSpecGetNumTerms(truncSpec, S);

    /* Should actually truncate? If so, do it: */
    truncated = k < S->numElts;
    if (truncated) {
      /** Truncate U: */

      BfMatDenseReal *Uk = bfMatToMatDenseReal(bfMatDenseRealGetColRangeCopy(U, 0, k));
      HANDLE_ERROR();

      bfMatDenseRealDeinitAndDealloc(&U);

      U = Uk;

      /** Truncate S: */

      BfMatDiagReal *Sk = bfMatDiagRealNew();
      HANDLE_ERROR();

      bfMatDiagRealInit(Sk, k, k);
      HANDLE_ERROR();

      for (BfSize i = 0; i < k; ++i)
        Sk->data[i] = S->data[i];

      bfMatDiagRealDeinitAndDealloc(&S);

      S = Sk;

      /** Truncate VT: */

      BfMatDenseReal *VkT = bfMatToMatDenseReal(bfMatDenseRealGetRowRangeCopy(VT, 0, k));
      HANDLE_ERROR();

      bfMatDenseRealDeinitAndDealloc(&VT);

      VT = VkT;
    }

    if (shouldDeleteMatDenseReal)
      bfMatDenseRealDeinitAndDealloc((BfMatDenseReal **)&matDenseReal);

    *UPtr  = bfMatDenseRealToMat(U);
    *SPtr  = S;
    *VTPtr = bfMatDenseRealToMat(VT);

    /* Ownership transferred to caller */
    U = NULL;
    VT = NULL;
    S = NULL;
  }

  else if (backend == BF_BACKEND_ARPACK) {
    /* New sparse CSR+ARPACK backend */

    if (bfMatGetType(mat) != BF_TYPE_MAT_CSR_REAL)
      RAISE_ERROR(BF_ERROR_TYPE_ERROR);

    BfSize m = bfMatGetNumRows(mat);
    BfSize n = bfMatGetNumCols(mat);

    BfMatCsrReal const *Acsr = bfMatConstToMatCsrRealConst(mat);
    HANDLE_ERROR();

    /* Maximum possible numerical rank */
    BfSize maxPossibleRank = m < n ? m : n;

    if (n <= 2 || maxPossibleRank == 0) {
      /* Too small for ARPACK: just report “no SVD compression” and
       * let caller fall back to plain CSR.
       */
      *UPtr  = NULL;
      *SPtr  = NULL;
      *VTPtr = NULL;
      truncated = false;
    } else {
      /* Heuristic NEV choice:
       *
       *  - For large tol (e.g. 0.9 like your fluxpy runs), we expect
       *    very low rank, so cap NEV aggressively (<= 64).
       *  - For more stringent tolerances, allow more singular values
       *    but never anywhere near full rank.
       *
       *  This mirrors the Python pattern “estimate_rank -> svds(A, k)”
       *  where k is small compared to min(m, n).
       */
      BfSize targetMaxRank = maxPossibleRank;

      if (truncSpec != NULL && truncSpec->usingTol) {
        BfReal t = truncSpec->tol;

        if (t >= 0.5) {
          /* Very loose tolerance: only need a handful of modes */
          if (targetMaxRank > 64) targetMaxRank = 64;
        } else if (t >= 0.1) {
          /* Moderate tolerance */
          if (targetMaxRank > 128) targetMaxRank = 128;
        } else {
          /* Tight tolerance: still cap to avoid near-full SVDs */
          if (targetMaxRank > 256) targetMaxRank = 256;
        }
      } else {
        /* No tol provided: use a conservative cap */
        if (targetMaxRank > 128) targetMaxRank = 128;
      }

      /* ARPACK eigenproblem dimension is N = n (for A^T A) and
       * requires 1 <= NEV <= N - 2.
       */
      BfSize nev = targetMaxRank;
      if (nev > n - 2)
        nev = n - 2;

      if (nev < 1) {
        *UPtr  = NULL;
        *SPtr  = NULL;
        *VTPtr = NULL;
        truncated = false;
      } else {
        truncated =
          bfMatCsrRealSparseTruncatedSvd(Acsr, nev, truncSpec,
                                         UPtr, SPtr, VTPtr);
      }
    }
  }
  else {
    RAISE_ERROR(BF_ERROR_NOT_IMPLEMENTED);
  }

  BF_ERROR_END() {
    BF_DIE();
  }

//  fprintf(stderr, "[bf] bfGetTruncatedSvd: returning (backend=%d)\n",
//          (int)backend);

  return truncated;

}

/* ---- NEW: CSR transpose MVP and ARPACK-based top-k SVD ---- */

/* y <- A^T * x  (A is m×n CSR, x is length m, y length n)
 *
 * NOTE: We do NOT reimplement A*x here: that is already provided by
 * bfMatCsrRealMulVec via bfMatMulVec. This kernel only covers A^T*x,
 * for which there is no high-level BF API yet.
 */
static void csrMulVecTransOnly(BfMatCsrReal const *A,
                               BfReal const *x,
                               BfReal       *y)
{
  BF_ERROR_BEGIN();

  BfMat const *Amat = bfMatCsrRealConstToMatConst(A);
  HANDLE_ERROR();

  BfSize m = bfMatGetNumRows(Amat);
  BfSize n = bfMatGetNumCols(Amat);

  BfSize const *rowptr = bfMatCsrRealGetRowptrConstPtr(A);
  BfSize const *colind = bfMatCsrRealGetColindConstPtr(A);
  BfReal const *data   = bfMatCsrRealGetDataConstPtr(A);

  BF_ASSERT(rowptr != NULL);
  BF_ASSERT(colind != NULL);
  BF_ASSERT(data   != NULL);

  /* zero output */
  for (BfSize j = 0; j < n; ++j)
    y[j] = 0;

  /* classic CSR transpose MVP */
  for (BfSize i = 0; i < m; ++i) {
    BfReal xi = x[i];
    if (xi == 0)
      continue;
    for (BfSize k = rowptr[i]; k < rowptr[i + 1]; ++k) {
      BfSize j = colind[k];
      y[j] += data[k]*xi;
    }
  }

  BF_ERROR_END() {
    /* Should never happen in practice; just defensive. */
    BF_DIE();
  }
}

#if BF_HAVE_PRIMME_SVDS
/* PRIMME matrixMatvec callback:
 *
 *    y <- A * x        if *transpose == 0
 *    y <- A^T * x      if *transpose != 0
 *
 * A is stored as BfMatCsrReal in primme_svds->matrix.
 *
 * We support blockSize >= 1, treating x,y as column-major (ldx, ldy)
 * with leading dimensions *ldx, *ldy.
 */
static void bfPrimmeCsrMatrixMatvec(
    void *x, PRIMME_INT *ldx,
    void *y, PRIMME_INT *ldy,
    int *blockSize,
    int *transpose,
    primme_svds_params *primme_svds,
    int *ierr)
{
  BfMatCsrReal const *Acsr = (BfMatCsrReal const *)primme_svds->matrix;
  BfMat const *Amat = bfMatCsrRealConstToMatConst(Acsr);

  BfSize m = bfMatGetNumRows(Amat);
  BfSize n = bfMatGetNumCols(Amat);

  BfReal const *xData = (BfReal const *)x;
  BfReal *yData       = (BfReal *)y;

  BfSize ldx_ = (BfSize)*ldx;
  BfSize ldy_ = (BfSize)*ldy;
  int bs      = *blockSize;

  /* Safety: PRIMME may call with bs > 1; handle general case. */
  for (int j = 0; j < bs; ++j) {
    BfReal const *xcol = xData + (BfSize)j*ldx_;
    BfReal       *ycol = yData + (BfSize)j*ldy_;

#if BF_HAVE_PRIMME_SVDS
    /* Count CSR matvecs: one per vector in the block */
    if (*transpose == 0) {
      gPrimmeCsrMatvecCalls_A += 1;
    } else {
      gPrimmeCsrMatvecCalls_AT += 1;
    }
#endif

    if (*transpose == 0) {
      /* ycol = A * xcol (x length n, y length m) */

      BfVecReal x_view;
      bfVecRealInitView(&x_view, n, BF_DEFAULT_STRIDE, (BfReal *)xcol);

      BfVecReal *y_real =
        bfVecToVecReal(bfMatMulVec(Amat, bfVecRealToVec(&x_view)));

      for (BfSize i = 0; i < m; ++i)
        ycol[i] = y_real->data[i*y_real->stride];

      bfVecRealDeinitAndDealloc(&y_real);
    }
    else {
      /* ycol = A^T * xcol (x length m, y length n) */
      csrMulVecTransOnly(Acsr, xcol, ycol);
    }
  }

  *ierr = 0;
}
#endif /* BF_HAVE_PRIMME_SVDS */

/* Internal ARPACK driver: compute top 'nev' eigenpairs of B = A^T A,
 * where A is m×n CSR, without forming B explicitly.
 *
 * - On exit, dr_out[0..nev-1] hold eigenvalues (lambda_i >= 0),
 *   and z_out is an N×nev matrix stored as columns (we flatten it as
 *   z_out[j*N + i] = j-th eigenvector component i).
 *
 * IMPORTANT: ARPACK may return fewer than 'nev' converged eigenpairs.
 * In that case we:
 *   - let nconv = iparam[4] > 0,
 *   - fill dr_out[0..nconv-1] and z_out columns 0..nconv-1,
 *   - zero-fill the remaining entries/columns so callers can safely
 *     treat them as “nonexistent / zero singular values”.
 */
static int csrAtA_top_eigs_arpack(BfMatCsrReal const *Acsr,
                                  BfSize               nev,
                                  BfReal              *dr_out,
                                  BfReal              *z_out)
{
  int status = 0; /* 0 = success, nonzero = failure */

  BfMat const *Amat = bfMatCsrRealConstToMatConst(Acsr);
  if (Amat == NULL) {
    SPARSE_SVD_LOG("[bf] sparse SVD: csrAtA_top_eigs_arpack: Amat==NULL\n");
    return -1;
  }

  a_int const N   = (a_int)bfMatGetNumCols(Amat);  /* dimension of A^T A */
  a_int const NEV = (a_int)nev;

  SPARSE_SVD_LOG("[bf] sparse SVD: csrAtA_top_eigs_arpack: N=%d NEV=%d\n",
                 (int)N, (int)NEV);

  if (N <= 2) {
    SPARSE_SVD_LOG("[bf] sparse SVD: csrAtA_top_eigs_arpack: N=%d too small for ARPACK; skipping\n",
                   (int)N);
    return 1;  /* nonzero = failure */
  }

  BF_ASSERT(NEV >= 1);
  BF_ASSERT(NEV <= N - 2);

  char const which[]  = "LM";
  char       bmat     = 'I';
  BfReal     tol      = 0;
  a_int      ncv      = (a_int)estimateNcv(NEV, N);
  a_int      ldv      = N;
  a_int      lworkl   = 3*ncv*ncv + 6*ncv;
  a_int      rvec     = 1;
  char const howmny[] = "A";

  double *resid  = NULL;
  double *V      = NULL;
  a_int  *select = NULL;
  double *workd  = NULL;
  double *workl  = NULL;
  BfReal *workev = NULL;
  BfReal *dr     = NULL;
  BfReal *di     = NULL;
  BfReal *z      = NULL;

  a_int iparam[11] = {
    [0] = 1,    /* compute exact shifts */
    [2] = 10*N, /* max iterations */
    [6] = 1     /* mode 1: standard eigenproblem */
  };

  a_int ipntr[11];
  a_int ido  = 0;
  a_int info = 0;

  /* allocate workspace */
  resid = bfMemAlloc(N, sizeof(BfReal));
  if (resid == NULL) { status = -1; goto cleanup; }
  for (a_int i = 0; i < N; ++i)
    resid[i] = 0.0;

  V = bfMemAlloc((BfSize)N*(BfSize)ncv, sizeof(BfReal));
  if (V == NULL) { status = -1; goto cleanup; }

  select = bfMemAllocAndZero(ncv, sizeof(a_int));
  if (select == NULL) { status = -1; goto cleanup; }

  workd = bfMemAlloc(3*N, sizeof(BfReal));
  if (workd == NULL) { status = -1; goto cleanup; }

  workl = bfMemAlloc(lworkl, sizeof(BfReal));
  if (workl == NULL) { status = -1; goto cleanup; }

  workev = bfMemAlloc(3*ncv, sizeof(BfReal));
  if (workev == NULL) { status = -1; goto cleanup; }

//  info = 1;  /* we provide resid as initial vector */
//
//  for (a_int i = 0; i < N; ++i)
//    resid[i] = 0.0;
//  resid[0] = 1.0;  /* simple e_1 starting vector */

  /* Debug: check initial resid and parameters */
  double res_norm0 = 0.0;
  for (a_int i = 0; i < N; ++i)
    res_norm0 += (double)resid[i] * (double)resid[i];
  res_norm0 = sqrt(res_norm0);

  SPARSE_SVD_LOG(
    "[bf] sparse SVD: csrAtA_top_eigs_arpack: "
    "about to call dnaupd: N=%d NEV=%d ncv=%d lworkl=%d info_init=%d resid_norm=%.3e\n",
    (int)N, (int)NEV, (int)ncv, (int)lworkl, (int)info, res_norm0);

dnaupd_loop:
  dnaupd_c(&ido, &bmat, N, which, NEV, tol, resid,
           ncv, V, ldv, iparam, ipntr, workd, workl, lworkl, &info);

  SPARSE_SVD_LOG(
  "[bf] sparse SVD: csrAtA_top_eigs_arpack: "
  "dnaupd done, info=%d, nconv=%d, ido=%d\n",
  (int)info, (int)iparam[4], (int)ido);

if (info == -9) {
  double res_norm = 0.0;
  for (a_int i = 0; i < N; ++i)
    res_norm += (double)resid[i] * (double)resid[i];
  res_norm = sqrt(res_norm);
  SPARSE_SVD_LOG(
    "[bf] sparse SVD: csrAtA_top_eigs_arpack: "
    "info==-9, resid_norm_after=%.3e\n", res_norm);
}

if (ido == 1 || ido == -1) {
  BF_ASSERT(ipntr[0] > 0);
  BF_ASSERT(ipntr[1] > 0);

  /* Additional sanity checks on ipntr */
  if (ipntr[0] - 1 + N > 3*N || ipntr[1] - 1 + N > 3*N) {
    SPARSE_SVD_LOG(
      "[bf] sparse SVD: csrAtA_top_eigs_arpack: "
      "ipntr out of range: ipntr[0]=%d ipntr[1]=%d N=%d\n",
      (int)ipntr[0], (int)ipntr[1], (int)N);
    status = -99;
    goto cleanup;
  }

  BfReal *x = &workd[ipntr[0] - 1];
  BfReal *y = &workd[ipntr[1] - 1];

  /* ---- ALWAYS compute y = A^T (A x) ---- */
  {
    BfVecReal x_view;
    bfVecRealInitView(&x_view, (BfSize)N, BF_DEFAULT_STRIDE, x);

    /* tmp = A * x  (length m) */
    BfVecReal *tmp_real =
      bfVecToVecReal(bfMatMulVec(Amat, bfVecRealToVec(&x_view)));

    /* y = A^T * tmp */
    csrMulVecTransOnly(Acsr, tmp_real->data, y);

#if BF_SPARSE_SVD_DEBUG
    /* Debug norms */
    double nx2 = 0.0;
    for (a_int i = 0; i < N; ++i) nx2 += (double)x[i]*(double)x[i];
    double nx = sqrt(nx2);

    BfSize m = bfMatGetNumRows(Amat);
    double nAx2 = 0.0;
    for (BfSize i = 0; i < m; ++i) {
      double v = (double)tmp_real->data[i*tmp_real->stride];
      nAx2 += v*v;
    }
    double nAx = sqrt(nAx2);

    double ny2 = 0.0;
    for (a_int i = 0; i < N; ++i) ny2 += (double)y[i]*(double)y[i];
    double ny = sqrt(ny2);

    SPARSE_SVD_LOG(
      "[bf] sparse SVD: OP-mv: ||x||=%.3e ||Ax||=%.3e ||A^T A x||=%.3e (N=%d, m=%lu)\n",
      nx, nAx, ny, (int)N, (unsigned long)m);
#endif

    bfVecRealDeinitAndDealloc(&tmp_real);
  }

  goto dnaupd_loop;
}

  SPARSE_SVD_LOG("[bf] sparse SVD: csrAtA_top_eigs_arpack: dnaupd done, info=%d, nconv=%d, ido=%d\n",
                 (int)info, (int)iparam[4], (int)ido);

  /* At this point:
   *  - info < 0  => genuine ARPACK error
   *  - info >= 0 => iparam[4] is the number of converged eigenvalues
   *
   * We accept partial convergence as long as nconv > 0.
   */
  a_int nconv = iparam[4];

if (info < 0 || nconv <= 0) {
  double res_norm_err = 0.0;
  for (a_int i = 0; i < N; ++i)
    res_norm_err += (double)resid[i] * (double)resid[i];
  res_norm_err = sqrt(res_norm_err);

  SPARSE_SVD_LOG(
    "[bf] sparse SVD: csrAtA_top_eigs_arpack: "
    "error exit: info=%d nconv=%d, ||resid||=%.3e\n",
    (int)info, (int)nconv, res_norm_err);

  status = (info != 0) ? (int)info : 1;
  goto cleanup;
}

  /* allocate for eigenvalues/eigenvectors (size NEV+1 as before) */
  dr = bfMemAlloc(NEV + 1, sizeof(BfReal));
  if (dr == NULL) { status = -1; goto cleanup; }

  di = bfMemAlloc(NEV + 1, sizeof(BfReal));
  if (di == NULL) { status = -1; goto cleanup; }

  z = bfMemAlloc((BfSize)N*(BfSize)(NEV + 1), sizeof(BfReal));
  if (z == NULL) { status = -1; goto cleanup; }

  BfSize ldz = (BfSize)N;

  BfReal sigmar = 0.0;
  BfReal sigmai = 0.0;

  dneupd_c(
    rvec,
    howmny,
    select,
    dr,
    di,
    z,
    (a_int)ldz,
    sigmar,
    sigmai,
    workev,
    &bmat,
    N,
    which,
    NEV,
    tol,
    resid,
    ncv,
    V,
    ldv,
    iparam,
    ipntr,
    workd,
    workl,
    lworkl,
    &info);

  if (info != 0) {
    SPARSE_SVD_LOG("[bf] sparse SVD: csrAtA_top_eigs_arpack: dneupd error, info=%d\n",
                   (int)info);
    status = (int)info;
    goto cleanup;
  }

  /* Use only the first nconv eigenpairs; zero out the rest. */
  BfSize K = (BfSize)nconv;
  if (K > (BfSize)NEV)
    K = (BfSize)NEV;

  SPARSE_SVD_LOG("[bf] sparse SVD: csrAtA_top_eigs_arpack: using K=%lu of NEV=%d eigenpairs\n",
                 (unsigned long)K, (int)NEV);

  /* eigenvalues */
  for (BfSize i = 0; i < K; ++i) {
    if (fabs(di[i]) > 1e-12) {
      SPARSE_SVD_LOG("[bf] sparse SVD: csrAtA_top_eigs_arpack: complex eigenvalue di[%lu]=%g\n",
                     (unsigned long)i, (double)di[i]);
      status = 2;
      goto cleanup;
    }
    dr_out[i] = dr[i];
  }
  /* zero any unused slots */
  for (BfSize i = K; i < (BfSize)NEV; ++i)
    dr_out[i] = 0.0;

  /* eigenvectors */
  for (BfSize j = 0; j < K; ++j) {
    for (BfSize i = 0; i < (BfSize)N; ++i) {
      z_out[j*(BfSize)N + i] = z[i + j*(BfSize)N];
    }
  }
  for (BfSize j = K; j < (BfSize)NEV; ++j) {
    for (BfSize i = 0; i < (BfSize)N; ++i) {
      z_out[j*(BfSize)N + i] = 0.0;
    }
  }

cleanup:
  if (status != 0) {
    SPARSE_SVD_LOG("[bf] sparse SVD: csrAtA_top_eigs_arpack: cleanup with status=%d\n",
                   status);
  }

  bfMemFree(resid);
  bfMemFree(V);
  bfMemFree(select);
  bfMemFree(workd);
  bfMemFree(workl);
  bfMemFree(workev);
  bfMemFree(dr);
  bfMemFree(di);
  bfMemFree(z);

  return status;
}

/* Compute a truncated SVD of a CSR real matrix A using ARPACK on A^T A.
 *
 * Inputs:
 *   Acsr      : m×n CSR (BF_TYPE_MAT_CSR_REAL)
 *   maxRank   : number of largest singular values to attempt (nev)
 *   truncSpec : tolerance / rank selection (same semantics as dense SVD)
 *
 * Outputs:
 *   UPtr, SPtr, VTPtr: filled with U, S, V^T where:
 *     - U is m×k, V^T is k×n, S is k×k diag, with k determined from
 *       truncSpec and the computed singular values (k <= maxRank).
 *
 * Return:
 *   true if the SVD was actually truncated (k < maxRank),
 *   false otherwise.
 */
static bool bfMatCsrRealSparseTruncatedSvd(BfMatCsrReal const *Acsr,
                                           BfSize              maxRank,
                                           BfTruncSpec const  *truncSpec,
                                           BfMat             **UPtr,
                                           BfMatDiagReal     **SPtr,
                                           BfMat             **VTPtr)
{
  BF_ERROR_BEGIN();

  /* Always initialize anything that may be freed in cleanup: */
  BfReal *lambda = NULL;
  BfReal *Z = NULL;
  BfReal *sigma = NULL;
  BfSize *permDesc = NULL;
  BfReal *sigmaSorted = NULL;
  BfReal *ZSorted = NULL;

  BfReal *resNorms = NULL;
  BfReal *resNormsSorted = NULL;

  BfReal *svals = NULL;
  BfReal *svecs = NULL;

  BfRealArray *sigmaArray = NULL;
  BfPerm *permAsc = NULL;
  BfMatDiagReal *Sfull = NULL;

  BfMatDenseReal *U = NULL;
  BfMatDiagReal  *S = NULL;
  BfMatDenseReal *VT = NULL;

  bool truncated = false;

  if (UPtr == NULL || SPtr == NULL || VTPtr == NULL) {
    SPARSE_SVD_LOG("[bf] sparse SVD: invalid NULL output pointers\n");
    RAISE_ERROR(BF_ERROR_INVALID_ARGUMENTS);
  }

  /* Default: keep CSR unless we successfully build a usable SVD */
  *UPtr  = NULL;
  *SPtr  = NULL;
  *VTPtr = NULL;

  BfMat const *Amat = bfMatCsrRealConstToMatConst(Acsr);
  HANDLE_ERROR();

  BfSize m = bfMatGetNumRows(Amat);
  BfSize n = bfMatGetNumCols(Amat);

  /* nnz check */
BfSize const *rowptr = bfMatCsrRealGetRowptrConstPtr(Acsr);
BF_ASSERT(rowptr != NULL);
BfSize nnz = rowptr[m];

SPARSE_SVD_LOG("[bf] sparse SVD: enter m=%lu n=%lu nnz=%lu maxRank(in)=%lu\n",
               (unsigned long)m,
               (unsigned long)n,
               (unsigned long)nnz,
               (unsigned long)maxRank);

if (nnz == 0) {
  SPARSE_SVD_LOG("[bf] sparse SVD: skipping zero block (nnz=0), keep sparse/zero leaf\n");
  return false;
}

/* NEW: detect “numerically zero” CSR: all data entries are zero */
BfReal const *data = bfMatCsrRealGetDataConstPtr(Acsr);
BF_ASSERT(data != NULL);

BfReal maxAbs = 0;
for (BfSize k = 0; k < nnz; ++k) {
  BfReal a = data[k];
  if (a < 0) a = -a;
  if (a > maxAbs) maxAbs = a;
}

if (maxAbs <= 10*DBL_MIN) {
  SPARSE_SVD_LOG("[bf] sparse SVD: skipping numerically-zero block (max|a_ij|=%.3e)\n",
                 (double)maxAbs);
  return false;
}

/* NEW: optional – treat very tiny blocks as not worth compressing */
if (maxAbs < 1e-11) {
  SPARSE_SVD_LOG(
    "[bf] sparse SVD: block too small to bother (max|a_ij|=%.3e), keeping CSR\n",
    (double)maxAbs);
  return false;
}

/* Quick sanity check on OP for debugging: apply A^T A to a test vector */
#if BF_SPARSE_SVD_DEBUG
{
  if (n <= 256 && m <= 256) {
    BfReal *test_x = bfMemAlloc(n, sizeof(BfReal));
    BfReal *test_y = bfMemAlloc(n, sizeof(BfReal));
    if (test_x != NULL && test_y != NULL) {
      for (BfSize j = 0; j < n; ++j)
        test_x[j] = 1.0;  /* simple nonzero vector */

      /* y = A^T A x using the same kernels as in ARPACK driver */
      BfVecReal x_view;
      bfVecRealInitView(&x_view, n, BF_DEFAULT_STRIDE, test_x);

      BfVecReal *tmp_real =
        bfVecToVecReal(bfMatMulVec(bfMatCsrRealConstToMatConst(Acsr),
                                   bfVecRealToVec(&x_view)));

      csrMulVecTransOnly(Acsr, tmp_real->data, test_y);

      double nAx2 = 0.0;
      for (BfSize j = 0; j < n; ++j)
        nAx2 += (double)test_y[j] * (double)test_y[j];
      double nAx = sqrt(nAx2);

      SPARSE_SVD_LOG(
        "[bf] sparse SVD: pre-ARPACK check: ||A^T A * 1|| = %.3e (m=%lu, n=%lu, nnz=%lu)\n",
        nAx, (unsigned long)m, (unsigned long)n, (unsigned long)nnz);

      bfVecRealDeinitAndDealloc(&tmp_real);
    }
    bfMemFree(test_x);
    bfMemFree(test_y);
  }
}
#endif

  /* SVD rank is at most min(m,n) */
  BfSize maxPossibleRank = m < n ? m : n;
  if (maxRank == 0 || maxRank > maxPossibleRank)
    maxRank = maxPossibleRank;

  /* flux-style memory budget */
  {
    double maxNbytes    = (double)nnz * (double)sizeof(BfReal);
    double bytesPerRank = (double)(m + n + 1) * (double)sizeof(BfReal);
    BfSize maxRankByBytes = maxRank;

    if (bytesPerRank > 0) {
      maxRankByBytes = (BfSize)(maxNbytes / bytesPerRank);

      if (maxRankByBytes == 0 && nnz > 0)
        maxRankByBytes = 1;

      if (maxRank > maxRankByBytes)
        maxRank = maxRankByBytes;
    }

    SPARSE_SVD_LOG("[bf] sparse SVD: mem budget: maxNbytes=%.3e bytesPerRank=%.3e -> maxRankByBytes=%lu, chosen maxRank=%lu\n",
                   maxNbytes,
                   bytesPerRank,
                   (unsigned long)maxRankByBytes,
                   (unsigned long)maxRank);
  }

  /* ARPACK constraints */
  if (n > 2 && maxRank > n - 2)
    maxRank = n - 2;

  if (maxRank == 0)
    maxRank = 1;

//  /* NEW: avoid PRIMME on small blocks to sidestep hangs */
//  {
//    const BfSize SMALL_PRIMME_DIM = 128;  /* tune as you like */
//
//    if (m <= SMALL_PRIMME_DIM && n <= SMALL_PRIMME_DIM) {
//      SPARSE_SVD_LOG(
//        "[bf] sparse SVD: small CSR block (m=%lu, n=%lu) -> "
//        "skip PRIMME and keep CSR (no truncation)\n",
//        (unsigned long)m, (unsigned long)n);
//      return false;  /* leave this block as CSR */
//    }
//  }

  SPARSE_SVD_LOG("[bf] sparse SVD: final ARPACK maxRank(nev)=%lu (m=%lu n=%lu)\n",
                 (unsigned long)maxRank,
                 (unsigned long)m,
                 (unsigned long)n);

  /* Step 1: eigen-information / singular information for A via PRIMME_SVDS
   *
   * We call dprimme_svds to get singular values and both left and right
   * singular vectors, but to reuse the rest of this routine we convert
   * them into:
   *
   *   lambda[i] = sigma[i]^2,
   *   Z(:,i)    = right singular vector v_i  (n entries).
   *
   * The subsequent code (sorting, truncation, U construction) uses
   * lambda and Z exactly as it did with the ARPACK eigen solver.
   */
#if BF_HAVE_PRIMME_SVDS

  /* --- PRIMME_SVDS backend: get singular values & right singular vectors --- */

  svals    = bfMemAlloc(maxRank, sizeof(BfReal));
  svecs    = bfMemAlloc((BfSize)(m + n)*maxRank, sizeof(BfReal));
  resNorms = bfMemAlloc(maxRank, sizeof(BfReal));

  if (svals == NULL || svecs == NULL || resNorms == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  primme_svds_params primme;
  primme_svds_initialize(&primme);

  /* Reasonable global matvec cap for this leaf */
  {
    PRIMME_INT mvBudget = (PRIMME_INT)(500 * (PRIMME_INT)maxRank);

    if (mvBudget < 2000)  mvBudget = 2000;       /* don’t be absurdly small */
    if (mvBudget > 10000) mvBudget = 10000;      /* hard cap */

    /* SVD-level budget */
    primme.maxMatvecs = mvBudget;
    /* Let the wrapper manage the inner eigensolvers */
    primme.primme.maxMatvecs       = 0;
    primme.primmeStage2.maxMatvecs = 0;

    SPARSE_SVD_LOG(
      "[bf] sparse SVD: PRIMME maxMatvecs set to %d (maxRank=%lu, m=%lu, n=%lu)\n",
      (int)mvBudget,
      (unsigned long)maxRank,
      (unsigned long)m,
      (unsigned long)n);
  }

  primme.monitorFun      = bfPrimmeSvdsCappedMonitor;
  primme.monitorFun_type = primme_op_double;
  #if BF_SPARSE_SVD_DEBUG
    primme.printLevel = 2;
  #endif

  primme.m       = (PRIMME_INT)m;
  primme.n       = (PRIMME_INT)n;
  primme.mLocal  = primme.m;
  primme.nLocal  = primme.n;
  primme.numProcs = 1;
  primme.procID   = 0;

  primme.matrix            = (void *)Acsr;
  primme.matrixMatvec      = bfPrimmeCsrMatrixMatvec;
  primme.matrixMatvec_type = primme_op_double;

  primme.numSvals     = (int)maxRank;
  if (primme.numSvals <= 0) primme.numSvals = 1;

  primme.target       = primme_svds_largest;
  primme.maxBlockSize = 1;
  primme.precondition = 0;
  primme.applyPreconditioner = NULL;
  primme.globalSumReal       = NULL;
  primme.broadcastReal       = NULL;

  primme.eps = 1e-3;
  if (truncSpec != NULL && truncSpec->usingTol) {
    BfReal t = truncSpec->tol;
    if (t < primme.eps)
      primme.eps = t;
  }

  bfSparseSvdsMonitorWarned = 0;

  /* Snapshot CSR matvec counters before calling PRIMME */
  long long csrA_before  = gPrimmeCsrMatvecCalls_A;
  long long csrAT_before = gPrimmeCsrMatvecCalls_AT;

  SPARSE_SVD_LOG(
    "[bf] sparse SVD: calling dprimme_svds for m=%lu n=%lu nnz=%lu maxRank=%lu "
    "(csrA_before=%lld, csrAT_before=%lld)\n",
    (unsigned long)m,
    (unsigned long)n,
    (unsigned long)nnz,
    (unsigned long)maxRank,
    csrA_before,
    csrAT_before);

  int primme_ret = dprimme_svds((double *)svals,
                                (double *)svecs,
                                (double *)resNorms,
                                &primme);
  (void)primme_ret;

  /* Snapshot counters after call */
  long long csrA_after  = gPrimmeCsrMatvecCalls_A;
  long long csrAT_after = gPrimmeCsrMatvecCalls_AT;

  long long csrA_delta  = csrA_after  - csrA_before;
  long long csrAT_delta = csrAT_after - csrAT_before;
  long long csr_total   = csrA_delta + csrAT_delta;

  /* Count how many singular values look “usable” (finite) */
  int numConv = 0;
  for (int j = 0; j < primme.numSvals; ++j) {
    if (!bfIsFiniteReal(svals[j])) break;
    /* allow tiny negatives later; but stop if wildly negative/zero */
    if (svals[j] <= 0) break;
    ++numConv;
  }

  SPARSE_SVD_LOG(
    "[bf] sparse SVD: PRIMME done (ret=%d, numConv=%d) for m=%lu n=%lu nnz=%lu maxRank=%lu\n",
    primme_ret, numConv,
    (unsigned long)m,
    (unsigned long)n,
    (unsigned long)nnz,
    (unsigned long)maxRank);

  SPARSE_SVD_LOG(
    "[bf] sparse SVD: PRIMME stats: "
    "svds.numMatvecs=%lld, primme.numMatvecs=%lld, stage2.numMatvecs=%lld ; "
    "CSR calls in this SVD: A*x=%lld, A^T*x=%lld (total=%lld)\n",
    (long long)primme.stats.numMatvecs,
    (long long)primme.primme.stats.numMatvecs,
    (long long)primme.primmeStage2.stats.numMatvecs,
    csrA_delta,
    csrAT_delta,
    csr_total);

  /* Also show approximate ratio CSR / “operator matvec” if available */
#if BF_SPARSE_SVD_DEBUG
  if (primme.stats.numMatvecs > 0) {
    double ratio = (double)csr_total / (double)primme.stats.numMatvecs;
    SPARSE_SVD_LOG(
      "[bf] sparse SVD: approx CSR-per-PRIMME-matvec ratio = %.3f "
      "(csr_total=%lld, svds.numMatvecs=%lld)\n",
      ratio,
      csr_total,
      (long long)primme.stats.numMatvecs);
  }
#else
  (void)csr_total;
#endif

  /* Fluxpy-style: accept partial results even if primme_ret != 0,
   * as long as we got >=1 finite singular value.
   */
  if (numConv <= 0) {
    SPARSE_SVD_LOG(
      "[bf] sparse SVD: PRIMME_SVDS produced 0 usable modes; keeping CSR block\n");
    truncated = false;
    goto cleanup;
  }

  /* Require the leading singular value to be sane */
  if (!bfIsFiniteReal(svals[0]) || svals[0] <= 0) {
    SPARSE_SVD_LOG(
      "[bf] sparse SVD: PRIMME_SVDS leading s0 invalid (s0=%g); keeping CSR block\n",
      (double)svals[0]);
    truncated = false;
    goto cleanup;
  }

  /* Work only with the converged singular triplets. */
  if ((BfSize)numConv < maxRank)
    maxRank = (BfSize)numConv;

  /* Build lambda (sigma^2) and Z (right singular vectors), as if they came
   * from an eigen-decomposition of A^T A. Only use j < maxRank. */
  lambda = bfMemAlloc(maxRank, sizeof(BfReal));
  if (lambda == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  Z = bfMemAlloc((BfSize)n*maxRank, sizeof(BfReal));
  if (Z == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  BfSize ldsv = m + n;

for (BfSize j = 0; j < maxRank; ++j) {
  BfReal sj = svals[j];

  if (!bfIsFiniteReal(sj)) {
    SPARSE_SVD_LOG(
      "[bf] sparse SVD: PRIMME returned non-finite s[%lu]=%g; keeping CSR leaf\n",
      (unsigned long)j, (double)sj);
    truncated = false;
    goto cleanup;
  }

    /* Allow tiny negative s relative to the largest one */
    BfReal s0 = svals[0];
    BfReal relTol = 1e-4;           /* tweakable */
    BfReal absTol = 100*DBL_EPSILON; /* absolute floor */
    BfReal negTol = relTol*fabs(s0) + absTol;

    if (sj < 0 && fabs(sj) <= negTol) {
      sj = 0;
    } else if (sj < 0) {
      SPARSE_SVD_LOG(
        "[bf] sparse SVD: PRIMME returned significantly negative s[%lu]=%.3e (s0=%.3e); keeping CSR leaf\n",
        (unsigned long)j, (double)sj, (double)s0);

    truncated = false;
    goto cleanup;
  }

  lambda[j] = sj*sj;

    /* Right singular vector v_j: last n entries of column j in svecs */
    const BfReal *col = &svecs[j*ldsv];
    for (BfSize i = 0; i < n; ++i)
      Z[j*n + i] = col[m + i];
  }

  SPARSE_SVD_LOG(
    "[bf] sparse SVD: PRIMME_SVDS accepted maxRank=%lu for m=%lu n=%lu nnz=%lu\n",
    (unsigned long)maxRank,
    (unsigned long)m,
    (unsigned long)n,
    (unsigned long)nnz);

#else /* !BF_HAVE_PRIMME_SVDS */

  /* Fallback: original ARPACK route */
  lambda = bfMemAlloc(maxRank, sizeof(BfReal));
  if (lambda == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  Z = bfMemAlloc((BfSize)n*maxRank, sizeof(BfReal));
  if (Z == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  int arpack_info;
#ifdef _OPENMP
#pragma omp critical(bf_sparse_svd_arpack)
  {
    arpack_info = csrAtA_top_eigs_arpack(Acsr, maxRank, lambda, Z);
  }
#else
  arpack_info = csrAtA_top_eigs_arpack(Acsr, maxRank, lambda, Z);
#endif

  if (arpack_info != 0) {
    SPARSE_SVD_LOG(
      "[bf] sparse SVD: ARPACK stage failed (status=%d) for m=%lu n=%lu nnz=%lu nev=%lu; "
      "keeping CSR leaf\n",
      arpack_info,
      (unsigned long)m,
      (unsigned long)n,
      (unsigned long)nnz,
      (unsigned long)maxRank);

    truncated = false;
    goto cleanup;
  }

  SPARSE_SVD_LOG("[bf] sparse SVD: ARPACK completed for m=%lu n=%lu nnz=%lu nev=%lu\n",
                 (unsigned long)m,
                 (unsigned long)n,
                 (unsigned long)nnz,
                 (unsigned long)maxRank);

#endif /* BF_HAVE_PRIMME_SVDS */


  /* Step 2: singular values = sqrt(lambda_i) */
  sigma = bfMemAlloc(maxRank, sizeof(BfReal));
  if (sigma == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  for (BfSize i = 0; i < maxRank; ++i) {
    BfReal lam = lambda[i];
    if (lam < 0 && fabs(lam) < 1e-14)
      lam = 0;
    if (lam < 0) {
      SPARSE_SVD_LOG("[bf] sparse SVD: negative eigenvalue lambda[%lu]=%g, clamping to 0\n",
                     (unsigned long)i, (double)lam);
      lam = 0;
    }
    sigma[i] = sqrt(lam);
  }

  /* Sort singular values in descending order as before */
  sigmaArray = bfRealArrayNewWithDefaultCapacity();
  HANDLE_ERROR();

  for (BfSize i = 0; i < maxRank; ++i) {
    bfRealArrayAppend(sigmaArray, sigma[i]);
    HANDLE_ERROR();
  }

  permAsc = bfRealArrayArgsort(sigmaArray);
  HANDLE_ERROR();

  permDesc = bfMemAlloc(maxRank, sizeof(BfSize));
  if (permDesc == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  for (BfSize i = 0; i < maxRank; ++i)
    permDesc[i] = permAsc->index[maxRank - 1 - i];

  /* NEW: sort PRIMME residual norms to match descending singular ordering */
  resNormsSorted = NULL;
  if (resNorms != NULL) {
    resNormsSorted = bfMemAlloc(maxRank, sizeof(BfReal));
    if (resNormsSorted == NULL)
      RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

    for (BfSize j = 0; j < maxRank; ++j) {
      BfSize src_j = permDesc[j];
      resNormsSorted[j] = resNorms[src_j];
    }
  }

  sigmaSorted = bfMemAlloc(maxRank, sizeof(BfReal));
  if (sigmaSorted == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  ZSorted = bfMemAlloc((BfSize)n*maxRank, sizeof(BfReal));
  if (ZSorted == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  for (BfSize j = 0; j < maxRank; ++j) {
    BfSize src_j = permDesc[j];
    sigmaSorted[j] = sigma[src_j];
    for (BfSize i = 0; i < n; ++i)
      ZSorted[j*n + i] = Z[src_j*n + i];
  }

  Sfull = bfMatDiagRealNew();
  HANDLE_ERROR();

  bfMatDiagRealInit(Sfull, maxRank, maxRank);
  HANDLE_ERROR();

  for (BfSize i = 0; i < maxRank; ++i)
    Sfull->data[i] = sigmaSorted[i];

  /* Fluxpy-style: do NOT reject based on solver residuals.
   * Residuals are useful for logging/debug, but physicality + memory gates
   * are the acceptance criteria.
   */
  if (resNormsSorted != NULL) {
    double relResMax = 0.0;
    BfSize jMax = maxRank < 3 ? maxRank : 3; /* only look at first few */
    for (BfSize j = 0; j < jMax; ++j) {
      double s = (double)sigmaSorted[j];
      double denom = (fabs(s) > 1e-30) ? fabs(s) : 1e-30;
      double rel = fabs((double)resNormsSorted[j]) / denom;
      if (rel > relResMax) relResMax = rel;
    }
    SPARSE_SVD_LOG("[bf] sparse SVD: PRIMME relResMax(first<=3)=%.3e (eps=%.3e)\n",
                   relResMax,
#if BF_HAVE_PRIMME_SVDS
                   (double)primme.eps
#else
                   0.0
#endif
                   );
  }


  if (!bfIsFiniteReal(Sfull->data[0]) || Sfull->data[0] <= 0) {
    SPARSE_SVD_LOG("[bf] sparse SVD: largest singular value ~ 0 (m=%lu n=%lu nnz=%lu maxRank=%lu) -> treat as zero block\n",
                   (unsigned long)m,
                   (unsigned long)n,
                   (unsigned long)nnz,
                   (unsigned long)maxRank);

    SPARSE_SVD_LOG(...);
    truncated = false;
    goto cleanup;
  }

#if BF_SPARSE_SVD_DEBUG
  if (truncSpec->usingTol) {
    BfSize k_dbg = bfTruncSpecGetNumTerms(truncSpec, Sfull);

    double s0 = (double)Sfull->data[0];
    double totalFrob2 = 0.0, tailFrob2 = 0.0;

    for (BfSize i = 0; i < maxRank; ++i) {
      double si = (double)Sfull->data[i];
      totalFrob2 += si*si;
      if (i >= k_dbg)
        tailFrob2 += si*si;
    }

    double relTail = (totalFrob2 > 0.0)
      ? sqrt(tailFrob2/totalFrob2)
      : 0.0;

    double skm1_rel = (k_dbg > 0)
      ? (double)Sfull->data[k_dbg - 1] / s0
      : 0.0;

    double sk_rel = (k_dbg < maxRank)
      ? (double)Sfull->data[k_dbg] / s0
      : 0.0;

    SPARSE_SVD_LOG(
      "[bf] sparse SVD: m=%lu n=%lu maxRank=%lu tol=%.3e k_dbg=%lu "
      "sigma[k-1]/sigma[0]=%.3e sigma[k]/sigma[0]=%.3e tail_frob_rel=%.3e\n",
      (unsigned long)m,
      (unsigned long)n,
      (unsigned long)maxRank,
      (double)truncSpec->tol,
      (unsigned long)k_dbg,
      skm1_rel,
      sk_rel,
      relTail);
  }
#endif

  /* Rank selection */
  BfSize k = 0;
  if (truncSpec->usingTol) {
    k = bfTruncSpecGetNumTerms(truncSpec, Sfull);
  } else {
    k = truncSpec->k <= maxRank ? truncSpec->k : maxRank;
  }

  if (k == 0)
    k = 1;

  truncated = (k < maxRank);

  SPARSE_SVD_LOG("[bf] sparse SVD: chosen rank k=%lu (truncated=%d) for m=%lu n=%lu\n",
                 (unsigned long)k,
                 (int)truncated,
                 (unsigned long)m,
                 (unsigned long)n);

  bool achievedTol = true;
  if (truncSpec->usingTol && k == maxRank) {
    double s0 = (double)Sfull->data[0];
    double sLast = (double)Sfull->data[maxRank - 1];
    double rel = (s0 > 0) ? (sLast / s0) : 0.0;

    if (rel > (double)truncSpec->tol) {
      achievedTol = false;
      SPARSE_SVD_LOG(
        "[bf] sparse SVD: did not reach tol before maxRank "
        "(maxRank=%lu, s_last/s0=%.3e > tol=%.3e); keeping SVD only if it helps\n",
        (unsigned long)maxRank, rel, (double)truncSpec->tol);
    }
  }
  SPARSE_SVD_LOG("[bf] sparse SVD: achievedTol=%d\n", (int)achievedTol);

  /* ---- Memory-benefit gate: keep CSR if SVD is not meaningfully smaller ---- */
  {
    double bytes_csr = bfSparseSvdEstimateBytesCsr(m, nnz);
    double bytes_svd = bfSparseSvdEstimateBytesSvd(m, n, k);

    SPARSE_SVD_LOG(
      "[bf] sparse SVD: bytes: csr=%.3e svd(k=%lu)=%.3e (ratio=%.3f) achievedTol=%d\n",
      bytes_csr, (unsigned long)k, bytes_svd,
      (bytes_csr > 0 ? bytes_svd/bytes_csr : 1.0),
      (int)achievedTol);

    if (bytes_svd >= BF_SPARSE_SVD_MIN_SAVINGS_FRAC * bytes_csr) {
      SPARSE_SVD_LOG(
        "[bf] sparse SVD: keeping CSR (SVD not worth it): %.3e >= %.3e * %.3e\n",
        bytes_svd, (double)BF_SPARSE_SVD_MIN_SAVINGS_FRAC, bytes_csr);
      truncated = false;
      goto cleanup;
    }
  }

  /* Build final S, VT, U as before */
  S = bfMatDiagRealNew();
  HANDLE_ERROR();
  bfMatDiagRealInit(S, k, k);
  HANDLE_ERROR();

  for (BfSize i = 0; i < k; ++i)
    S->data[i] = Sfull->data[i];

  VT = bfMatDenseRealNew();
  HANDLE_ERROR();
  bfMatDenseRealInit(VT, k, n);
  HANDLE_ERROR();

  for (BfSize j = 0; j < k; ++j) {
    BfVecReal vj_view;
    bfVecRealInitView(&vj_view, n, BF_DEFAULT_STRIDE, &ZSorted[j*n]);
    HANDLE_ERROR();
    bfMatDenseRealSetRow(bfMatDenseRealToMat(VT), j, bfVecRealToVec(&vj_view));
  }

  U = bfMatDenseRealNew();
  bfMatDenseRealInitZeros(U, m, k);
  HANDLE_ERROR();

  for (BfSize j = 0; j < k; ++j) {
    BfReal sj = S->data[j];

    BfVecReal vj_view;
    bfVecRealInitView(&vj_view, n, BF_DEFAULT_STRIDE, &ZSorted[j*n]);
    HANDLE_ERROR();

    BfVecReal *tmpReal =
      bfVecToVecReal(bfMatMulVec(Amat, bfVecRealToVec(&vj_view)));
    HANDLE_ERROR();

    if (sj > 0) {
      BfReal invSj = 1.0/sj;
      for (BfSize i = 0; i < m; ++i)
        tmpReal->data[i*tmpReal->stride] *= invSj;
    } else {
      for (BfSize i = 0; i < m; ++i)
        tmpReal->data[i*tmpReal->stride] = 0;
    }

//    bfMatDenseRealSetCol(bfMatDenseRealToMat(U), j, bfVecRealToVec(tmpReal));
    bfMatDenseRealSetCol(U, j, bfVecRealToVec(tmpReal));
    HANDLE_ERROR();

    bfVecRealDeinitAndDealloc(&tmpReal);
  }

#if BF_SPARSE_SVD_PHYSICS_PROBE
  /* ---- Physics probe gate: A_svd * x should be ~nonnegative for x>=0 ----
   * This is intentionally cheap and catches catastrophic non-physical SVDs.
   */
  {
    double eta = bfSparseSvdNegEtaFromTol(truncSpec);

    /* Two probes: x = ones, and x = pseudo-random positive (deterministic) */
    const int numProbes = 2;

    BfReal *x = bfMemAlloc(n, sizeof(BfReal));
    BfReal *y = bfMemAlloc(m, sizeof(BfReal));
    BfReal *t = bfMemAlloc(k, sizeof(BfReal)); /* t = S * (VT*x), length k */

    if (x == NULL || y == NULL || t == NULL) {
      /* If we can't probe, don't block acceptance */
      if (x) bfMemFree(x);
      if (y) bfMemFree(y);
      if (t) bfMemFree(t);
    } else {
      for (int probe = 0; probe < numProbes; ++probe) {
        /* Build x >= 0 */
        if (probe == 0) {
          for (BfSize i = 0; i < n; ++i) x[i] = 1.0;
        } else {
          /* simple LCG-ish deterministic positive values in (0,1] */
          unsigned long long seed = 1469598103934665603ULL;
          for (BfSize i = 0; i < n; ++i) {
            seed ^= (seed << 13);
            seed ^= (seed >> 7);
            seed ^= (seed << 17);
            double u = (double)(seed & 0xffffffffULL) / (double)0xffffffffULL;
            if (u <= 0) u = 1e-6;
            x[i] = (BfReal)u;
          }
        }

        /* t_j = s_j * dot(v_j, x) where v_j is row j of VT */
        for (BfSize j = 0; j < k; ++j) {
          BfVecReal *vj = bfMatDenseRealGetRowView(VT, j);
          if (vj == NULL) { truncated = false; goto cleanup; }

          BfReal dot = 0;
          for (BfSize i = 0; i < n; ++i)
            dot += vj->data[i*vj->stride] * x[i];

          t[j] = S->data[j] * dot;

          bfVecRealDeinitAndDealloc(&vj);
        }

        /* y = U * t */
        for (BfSize i = 0; i < m; ++i) {
          BfVecReal *ui = bfMatDenseRealGetRowView(U, i);
          if (ui == NULL) { truncated = false; goto cleanup; }

          BfReal sum = 0;
          for (BfSize j = 0; j < k; ++j)
            sum += ui->data[j*ui->stride] * t[j];

          y[i] = sum;

          bfVecRealDeinitAndDealloc(&ui);
        }


        /* Compute min, max, p99, and negative fraction */
        BfReal yMin = BF_INFINITY;
        BfReal yMax = 0;
        BfSize negCount = 0;
        for (BfSize i = 0; i < m; ++i) {
          BfReal yi = y[i];
          if (yi < yMin) yMin = yi;
          if (yi > yMax) yMax = yi;
        }

        /* abs negativity threshold scaled to output magnitude */
        BfReal negAbs = (BfReal)(1e-12 * (double)(yMax > 0 ? yMax : 1.0));
        for (BfSize i = 0; i < m; ++i)
          if (y[i] < -negAbs) ++negCount;

        /* p99 requires a sorted view: make a copy */
        BfReal *yCopy = bfMemAlloc(m, sizeof(BfReal));
        if (yCopy == NULL) {
          /* fallback: use max as scale */
          BfReal scale = (yMax > 0 ? yMax : 1.0);
          if (yMin < (BfReal)(-eta * (double)scale) || ((double)negCount/(double)m) > 1e-3) {
            SPARSE_SVD_LOG("[bf] sparse SVD: physics probe reject (fallback scale): "
                           "probe=%d yMin=%.3e yMax=%.3e negFrac=%.3e eta=%.3e\n",
                           probe, (double)yMin, (double)yMax,
                           (double)negCount/(double)m, eta);
            truncated = false;
            goto cleanup;
          }
        } else {
          for (BfSize i = 0; i < m; ++i) yCopy[i] = y[i];
          BfReal yP99 = bfSparseSvdQuantile(yCopy, m, 0.99);
          bfMemFree(yCopy);

          BfReal scale = (yP99 > 0 ? yP99 : (yMax > 0 ? yMax : 1.0));
          double negFrac = (double)negCount / (double)m;

          SPARSE_SVD_LOG("[bf] sparse SVD: physics probe=%d yMin=%.3e yP99=%.3e yMax=%.3e negFrac=%.3e eta=%.3e\n",
                         probe, (double)yMin, (double)yP99, (double)yMax, negFrac, eta);

          if (yMin < (BfReal)(-eta * (double)scale) || negFrac > 1e-3) {
            SPARSE_SVD_LOG("[bf] sparse SVD: physics probe reject\n");
            truncated = false;
            goto cleanup;
          }
        }
      }

      bfMemFree(x);
      bfMemFree(y);
      bfMemFree(t);
    }
  }
#endif

  *UPtr  = bfMatDenseRealToMat(U);
  *SPtr  = S;
  *VTPtr = bfMatDenseRealToMat(VT);

  /* Ownership transferred to caller */
  U = NULL;
  VT = NULL;
  S = NULL;

  BF_ERROR_END() {
    BF_DIE();
  }

  /* Cleanup */
  cleanup:
      if (U)  bfMatDenseRealDeinitAndDealloc(&U);
      if (VT) bfMatDenseRealDeinitAndDealloc(&VT);
      if (S)  bfMatDiagRealDeinitAndDealloc(&S);

      if (Sfull) bfMatDiagRealDeinitAndDealloc(&Sfull);
      if (sigmaArray) bfRealArrayDeinitAndDealloc(&sigmaArray);
      if (permAsc) bfPermDeinitAndDealloc(&permAsc);

      if (lambda) bfMemFree(lambda);
      if (Z) bfMemFree(Z);
      if (sigma) bfMemFree(sigma);
      if (permDesc) bfMemFree(permDesc);
      if (sigmaSorted) bfMemFree(sigmaSorted);
      if (ZSorted) bfMemFree(ZSorted);

      if (resNormsSorted) bfMemFree(resNormsSorted);
      if (resNorms) bfMemFree(resNorms);

      if (svals) bfMemFree(svals);
      if (svecs) bfMemFree(svecs);

      return truncated;
}

