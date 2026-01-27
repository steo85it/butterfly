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
#include <stdlib.h>  /* atexit */

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
    #define BF_SPARSE_SVD_DEBUG 1
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
 *
 * Runtime override:
 *   export BF_PRIMME_SVDS_MAX_WALLTIME=0     -> disable time-based abort
 *   export BF_PRIMME_SVDS_MAX_WALLTIME=1.0   -> abort after ~1s per leaf
 *
 * If <= 0, no time-based stopping is applied.
 */
static double bfPrimmeSvdsMaxWalltime(void) {
  static int initialized = 0;
  static double wall = 10.0; /* default */

  if (!initialized) {
    initialized = 1;
    char const *s = getenv("BF_PRIMME_SVDS_MAX_WALLTIME");
    if (s && *s) {
      char *endp = NULL;
      double v = strtod(s, &endp);
      if (endp != s && isfinite(v)) {
        wall = v;
      }
    }
    SPARSE_SVD_LOG("[bf] sparse SVD: BF_PRIMME_SVDS_MAX_WALLTIME=%.6g (env override)\n", wall);
  }

  return wall;
}

/* ---- Sparse SVD aggregate stats --------------------------------------- */
#ifndef BF_SPARSE_SVD_STATS
#define BF_SPARSE_SVD_STATS 1
#endif

/* Print summary every N leaf attempts (0 disables periodic printing) */
#ifndef BF_SPARSE_SVD_STATS_EVERY
#define BF_SPARSE_SVD_STATS_EVERY 500
#endif

/* Counters (best-effort; in OpenMP builds, use atomic updates) */
static long long gSparseSvd_leafTotal        = 0;
static long long gSparseSvd_acceptSvd        = 0;
static long long gSparseSvd_reject_notAchTol = 0;
static long long gSparseSvd_reject_rowSum    = 0;
static long long gSparseSvd_reject_bytes     = 0;
static long long gSparseSvd_reject_physics   = 0;
static long long gSparseSvd_reject_solver    = 0;
static long long gSparseSvd_skip_nnz0        = 0;
static long long gSparseSvd_skip_numZero     = 0;
static long long gSparseSvd_skip_tiny        = 0;

#ifdef _OPENMP
  #define BF_SVDSTAT_INC(x) do { _Pragma("omp atomic") x++; } while (0)
  #define BF_SVDSTAT_ADD(x, v) do { _Pragma("omp atomic") x += (v); } while (0)
#else
  #define BF_SVDSTAT_INC(x) do { (x)++; } while (0)
  #define BF_SVDSTAT_ADD(x, v) do { (x) += (v); } while (0)
#endif

static void bfSparseSvdPrintStatsNow(char const *tag) {
#if BF_SPARSE_SVD_STATS
  long long t = gSparseSvd_leafTotal;
  long long acc   = gSparseSvd_acceptSvd;
  long long rejT  = gSparseSvd_reject_notAchTol;
  long long rejRS = gSparseSvd_reject_rowSum;
  long long rejB  = gSparseSvd_reject_bytes;
  long long rejP  = gSparseSvd_reject_physics;
  long long rejS  = gSparseSvd_reject_solver;
  long long sk0   = gSparseSvd_skip_nnz0;
  long long skZ   = gSparseSvd_skip_numZero;
  long long skTi  = gSparseSvd_skip_tiny;

  double denom = (t > 0) ? (double)t : 1.0;

  bfLogInfo(
    "[bf][sparse_svd][stats]%s leaves=%lld  acceptSVD=%lld(%.1f%%) "
    "rejTol=%lld  rejRowSum=%lld  rejBytes=%lld  rejPhys=%lld  rejSolver=%lld  "
    "skip(nnz0=%lld numZero=%lld tiny=%lld)\n",
    tag ? tag : "",
    t,
    acc, 100.0*(double)acc/denom,
    rejT, rejRS, rejB, rejP, rejS,
    sk0, skZ, skTi);
#else
  (void)tag;
#endif
}

static void bfSparseSvdMaybePrintStats(void) {
#if BF_SPARSE_SVD_STATS
  long long t = gSparseSvd_leafTotal;
  if (BF_SPARSE_SVD_STATS_EVERY <= 0) return;
  if (t > 0 && (t % (long long)BF_SPARSE_SVD_STATS_EVERY) == 0)
    bfSparseSvdPrintStatsNow("");
#endif
}

/* ---- Public API: print/reset sparse SVD stats ------------------------ */
/* Put prototypes in bf/linalg.h (see Patch 2). */

void bfSparseSvdPrintStats(char const *tag) {
#if BF_SPARSE_SVD_STATS
  bfSparseSvdPrintStatsNow(tag);
#else
  (void)tag;
#endif
}

void bfSparseSvdResetStats(void) {
#if BF_SPARSE_SVD_STATS
  gSparseSvd_leafTotal        = 0;
  gSparseSvd_acceptSvd        = 0;
  gSparseSvd_reject_notAchTol = 0;
  gSparseSvd_reject_rowSum    = 0;
  gSparseSvd_reject_bytes     = 0;
  gSparseSvd_reject_physics   = 0;
  gSparseSvd_reject_solver    = 0;
  gSparseSvd_skip_nnz0        = 0;
  gSparseSvd_skip_numZero     = 0;
  gSparseSvd_skip_tiny        = 0;
#endif
}

/* One-time atexit hook to print final summary (optional) */
#ifndef BF_SPARSE_SVD_STATS_ATEXIT
#define BF_SPARSE_SVD_STATS_ATEXIT 0
#endif

static int gSparseSvdAtexitInstalled = 0;

static void bfSparseSvdPrintStatsAtExit(void) {
  bfSparseSvdPrintStatsNow(" [final]");
}

static void bfSparseSvdInstallAtexitOnce(void) {
#if BF_SPARSE_SVD_STATS
  if (!BF_SPARSE_SVD_STATS_ATEXIT) return;
  if (!gSparseSvdAtexitInstalled) {
    gSparseSvdAtexitInstalled = 1;
    atexit(bfSparseSvdPrintStatsAtExit);
  }
#endif
}

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
  {
    double wall = bfPrimmeSvdsMaxWalltime();
    if (wall > 0.0 &&
        primme_svds->stats.elapsedTime > wall &&
        primme_svds->maxMatvecs != 0) {

    SPARSE_SVD_LOG(
      "[bf] sparse SVD: PRIMME monitor: elapsedTime=%.3e > %.3e s, "
      "forcing maxMatvecs=0 to abort\n",
      primme_svds->stats.elapsedTime,
      wall);

    /* Outer SVDS matvec budget */
    primme_svds->maxMatvecs = 0;

    /* Inner eigensolvers as well, for safety */
    primme_svds->primme.maxMatvecs       = 0;
    primme_svds->primmeStage2.maxMatvecs = 0;
    }
  }
}
#endif

static bool bfIsFiniteReal(BfReal x) {
  return isfinite(x);
}

/* ---- Sparse SVD accept/reject heuristics ---------------------------- */

/* Keep CSR if SVD isn't at least this fraction smaller (0.9 = save >=10%) */
#ifndef BF_SPARSE_SVD_MIN_SAVINGS_FRAC
#define BF_SPARSE_SVD_MIN_SAVINGS_FRAC 0.98
#endif

/* If 1: enforce tol strictly (reject SVD when tol not achieved before maxRank).
 * If 0: treat tol as a target; still allow acceptance via bytes + row-sum + phys gates.
 * Fluxpy behavior is closer to 0.
 */
#ifndef BF_SPARSE_SVD_STRICT_TOL
#define BF_SPARSE_SVD_STRICT_TOL 0
#endif

/* Enable a simple nonnegativity/physics probe for nonnegative operators */
#ifndef BF_SPARSE_SVD_PHYSICS_PROBE
#define BF_SPARSE_SVD_PHYSICS_PROBE 1
#endif

/* B) Cheap PRIMME sanity checks (recommended) */
#ifndef BF_SPARSE_SVD_SOLVER_SANITY
#define BF_SPARSE_SVD_SOLVER_SANITY 1
#endif

/* ||v|| should be ~1; ||A v|| should be ~sigma */
#ifndef BF_SPARSE_SVD_SANITY_VNORM_TOL
#define BF_SPARSE_SVD_SANITY_VNORM_TOL 1e-2
#endif

#ifndef BF_SPARSE_SVD_SANITY_SIG_REL_TOL
#define BF_SPARSE_SVD_SANITY_SIG_REL_TOL 1e-2
#endif

/* small orthogonality check for first few right singular vectors */
#ifndef BF_SPARSE_SVD_SANITY_ORTHO_TOL
#define BF_SPARSE_SVD_SANITY_ORTHO_TOL 5e-2
#endif

#ifndef BF_SPARSE_SVD_SANITY_MAXK
#define BF_SPARSE_SVD_SANITY_MAXK 4
#endif

/* Physics-probe helpers:
 * - scale floor prevents absurdly strict negativity tests on weak leaves
 * - y_floor skips the physics probe when outputs are numerically negligible
 */
#ifndef BF_SPARSE_SVD_PHYS_SCALE_FLOOR
#define BF_SPARSE_SVD_PHYS_SCALE_FLOOR 1e-12
#endif

#ifndef BF_SPARSE_SVD_PHYS_Y_FLOOR
#define BF_SPARSE_SVD_PHYS_Y_FLOOR 1e-14
#endif

/* If yMax is below this, the leaf output is effectively zero -> skip phys test */
#ifndef BF_SPARSE_SVD_PHYS_YMAX_FLOOR
#define BF_SPARSE_SVD_PHYS_YMAX_FLOOR 1e-14
#endif

/* Optional runtime override for eta_abs (set <=0 to ignore).
 *   export BF_SPARSE_SVD_NEG_ETA_ABS=0.05
 */
static double bfSparseSvdNegEtaAbsOverride(void) {
  static int initialized = 0;
  static double v = 0.0;
  if (!initialized) {
    initialized = 1;
    char const *s = getenv("BF_SPARSE_SVD_NEG_ETA_ABS");
    if (s && *s) {
      char *endp = NULL;
      double x = strtod(s, &endp);
      if (endp != s && isfinite(x)) v = x;
    }
  }
  return v;
}

/* Allowed negativity magnitude relative to p99 scale, as a function of tol */
static double bfSparseSvdNegEtaFromTol(BfTruncSpec const *truncSpec) {
  /* Runtime override wins (useful for quick experiments) */
  double ov = bfSparseSvdNegEtaAbsOverride();
  if (ov > 0.0 && isfinite(ov)) return ov;

  if (truncSpec && truncSpec->usingTol) {
    double t = (double)truncSpec->tol;

    /* Slightly looser defaults than before to avoid rejPhys on weak leaves */
    if (t >= 1.0)  return 1e-1;  /* was 5e-2 */
    if (t >= 0.1)  return 5e-2;  /* was 1e-2 */
    if (t >= 0.01) return 1e-2;  /* was 5e-3 */
    return 5e-3;                 /* was 1e-3 */
  }
  return 5e-2; /* was 1e-2 */
}

/* Allowed *negative mass* ratio, as a function of tol:
 *   neg_mass = sum(-min(y,0)) / sum(max(y,0))
 *
 * This prevents rejecting leaves where negatives exist but are tiny in magnitude.
 */
static double bfSparseSvdNegMassEtaFromTol(BfTruncSpec const *truncSpec) {
  if (truncSpec && truncSpec->usingTol) {
    double t = (double)truncSpec->tol;
    /* Looser tol -> tolerate a bit more “mass” leakage */
    if (t >= 1.0)  return 2e-2;
    if (t >= 0.1)  return 5e-3;
    if (t >= 0.01) return 1e-3;
    return 5e-4;
  }
  return 5e-3;
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

/* ---- NEW: Row-sum check ------------------------------------------------ */

/* Enable row-sum check (recommended for nonnegative operators like VF) */
#ifndef BF_SPARSE_SVD_ROWSUM_CHECK
#define BF_SPARSE_SVD_ROWSUM_CHECK 0
#endif

#ifndef BF_SPARSE_SVD_ROWSUM_REPAIR
#define BF_SPARSE_SVD_ROWSUM_REPAIR 0
#endif

/* If 1, only repair deficits (energy loss): d = max(rs_csr - rs_svd, 0).
 * This guarantees the repair term is nonnegative (won’t create negatives).
 */
#ifndef BF_SPARSE_SVD_ROWSUM_REPAIR_ONLY_DEFICIT
#define BF_SPARSE_SVD_ROWSUM_REPAIR_ONLY_DEFICIT 0
#endif

/* Base tolerances for row-sum drift */
#ifndef BF_SPARSE_SVD_ROWSUM_REL_TOL
#define BF_SPARSE_SVD_ROWSUM_REL_TOL 5e-3
#endif

#ifndef BF_SPARSE_SVD_ROWSUM_ABS_TOL
#define BF_SPARSE_SVD_ROWSUM_ABS_TOL 1e-12
#endif

/* Robust acceptance:
 * - allow a small fraction of rows to violate the relative tolerance
 * - use a high quantile of relative error (not max) to avoid 1-row outliers
 */
#ifndef BF_SPARSE_SVD_ROWSUM_BAD_FRAC_TOL
#define BF_SPARSE_SVD_ROWSUM_BAD_FRAC_TOL 0.01   /* allow 1% rows to be “bad” */
#endif

#ifndef BF_SPARSE_SVD_ROWSUM_REL_Q
#define BF_SPARSE_SVD_ROWSUM_REL_Q 0.99          /* check 99th percentile rel err */
#endif

/* Denominator floor for relative error: prevents tiny rows from dominating */
#ifndef BF_SPARSE_SVD_ROWSUM_DENOM_FLOOR
#define BF_SPARSE_SVD_ROWSUM_DENOM_FLOOR 1e-6
#endif

/* Optionally scale row-sum tolerance with truncSpec->tol */
static double bfSparseSvdRowSumRelTol(BfTruncSpec const *truncSpec) {
  (void)truncSpec;
  /* Keep this independent of trunc tol for VF operators:
   * trunc tol controls spectral truncation; row-sum gate protects energy.
   */
  return (double)BF_SPARSE_SVD_ROWSUM_REL_TOL; /* e.g. 5e-3 = 0.5% */
}

/* Forward decl: used by row-sum + physics checks */
static BfReal bfSparseSvdQuantile(BfReal const *y, BfSize n, double p);

/* Returns true if row-sums match within tolerance; false => reject SVD */
static bool bfSparseSvdCheckRowSums(
    BfMatCsrReal const *Acsr,
    BfMatDenseReal const *U,     /* m x k */
    BfMatDiagReal  const *S,     /* k diag */
    BfMatDenseReal const *VT,    /* k x n */
    BfTruncSpec const *truncSpec,
    BfSize k)
{
  BfMat const *Amat = bfMatCsrRealConstToMatConst(Acsr);
  BfSize m = bfMatGetNumRows(Amat);
  BfSize n = bfMatGetNumCols(Amat);

  BfSize const *rowptr = bfMatCsrRealGetRowptrConstPtr(Acsr);
  BfReal const *data   = bfMatCsrRealGetDataConstPtr(Acsr);
  BF_ASSERT(rowptr && data);

  BfReal *rs_csr = bfMemAlloc(m, sizeof(BfReal));
  BfReal *rs_svd = bfMemAlloc(m, sizeof(BfReal));
  BfReal *v      = bfMemAlloc(k, sizeof(BfReal)); /* v = VT * 1 */
  BfReal *t      = bfMemAlloc(k, sizeof(BfReal)); /* t = S * v */

  if (!rs_csr || !rs_svd || !v || !t) {
    if (rs_csr) bfMemFree(rs_csr);
    if (rs_svd) bfMemFree(rs_svd);
    if (v) bfMemFree(v);
    if (t) bfMemFree(t);
    /* If we can't check, don't block acceptance */
    return true;
  }

  /* 1) CSR row sums */
  for (BfSize i = 0; i < m; ++i) {
    BfReal s = 0;
    for (BfSize p = rowptr[i]; p < rowptr[i + 1]; ++p)
      s += data[p];
    rs_csr[i] = s;
  }

  /* 2) v = VT * 1  (avoid per-row GetRowView allocations) */
  {
    BfReal *ones = bfMemAlloc(n, sizeof(BfReal));
    if (ones == NULL) {
      bfMemFree(rs_csr); bfMemFree(rs_svd); bfMemFree(v); bfMemFree(t);
      /* If we can't check, don't block acceptance */
      return true;
    }

    for (BfSize i = 0; i < n; ++i) ones[i] = 1.0;

    BfVecReal ones_view;
    bfVecRealInitView(&ones_view, n, BF_DEFAULT_STRIDE, ones);

    BfVecReal *vvec =
      bfVecToVecReal(
        bfMatMulVec(bfMatDenseRealToMat((BfMatDenseReal *)VT),
                    bfVecRealToVec(&ones_view)));

    bfMemFree(ones);

    if (vvec == NULL) {
      bfMemFree(rs_csr); bfMemFree(rs_svd); bfMemFree(v); bfMemFree(t);
      return false;
    }

    for (BfSize j = 0; j < k; ++j)
      v[j] = vvec->data[j*vvec->stride];

    bfVecRealDeinitAndDealloc(&vvec);
  }

  /* 3) t = S * v */
  for (BfSize j = 0; j < k; ++j)
    t[j] = S->data[j] * v[j];

  /* 4) rs_svd = U * t  (avoid per-row GetRowView allocations) */
  {
    BfVecReal t_view;
    bfVecRealInitView(&t_view, k, BF_DEFAULT_STRIDE, t);

    BfVecReal *rsvec =
      bfVecToVecReal(
        bfMatMulVec(bfMatDenseRealToMat((BfMatDenseReal *)U),
                    bfVecRealToVec(&t_view)));

    if (rsvec == NULL) {
      bfMemFree(rs_csr); bfMemFree(rs_svd); bfMemFree(v); bfMemFree(t);
      return false;
    }

    for (BfSize i = 0; i < m; ++i)
      rs_svd[i] = rsvec->data[i*rsvec->stride];

    bfVecRealDeinitAndDealloc(&rsvec);
  }

  /* 5) Compare: per-row combined tolerance
   *    diff <= absTol + relTol*|a|
   *
   * This is crucial for VF-like operators where |rowsum| << 1:
   * using denom=max(1,|a|) turns a “relative” test into an absolute one.
   */
  double relTol = bfSparseSvdRowSumRelTol(truncSpec);
  double absTol = (double)BF_SPARSE_SVD_ROWSUM_ABS_TOL;

  /* Compute robust denom floor: avoid tiny rows dominating rel error */
  double denomFloor = (double)BF_SPARSE_SVD_ROWSUM_DENOM_FLOOR;
  if (denomFloor < absTol) denomFloor = absTol;

  /* Build rel error array, track maxAbs, and count “bad” rows */
  BfReal *relErr = bfMemAlloc(m, sizeof(BfReal));
  if (relErr == NULL) {
    /* If we can't allocate, don't block acceptance */
    bfMemFree(rs_csr); bfMemFree(rs_svd); bfMemFree(v); bfMemFree(t);
    return true;
  }

  double maxAbs = 0.0;
  BfSize worstAbsI = 0;
  long long bad = 0;

  for (BfSize i = 0; i < m; ++i) {
    double a = (double)rs_csr[i];
    double b = (double)rs_svd[i];
    double diff = fabs(b - a);

    double denom = fabs(a);
    if (denom < denomFloor) denom = denomFloor;

    double rel = diff / denom;
    relErr[i] = (BfReal)rel;

    if (diff > maxAbs) { maxAbs = diff; worstAbsI = i; }
    if (rel > relTol) bad++;
  }

  double badFrac = (m > 0) ? ((double)bad / (double)m) : 0.0;
  double q = (double)BF_SPARSE_SVD_ROWSUM_REL_Q;
  if (q < 0.0) q = 0.0;
  if (q > 1.0) q = 1.0;

  BfReal relQ = bfSparseSvdQuantile(relErr, m, q);

  /* Accept if:
   *  (A) absolute error is tiny, OR
   *  (B) high-quantile relative error is within tol AND only a small fraction are bad.
   */
  bool ok = (maxAbs <= absTol) ||
            ((double)relQ <= relTol && badFrac <= (double)BF_SPARSE_SVD_ROWSUM_BAD_FRAC_TOL);

  SPARSE_SVD_LOG(
    "[bf] sparse SVD: row-sum check: ok=%d relQ(%.2f)=%.3e tol=%.3e badFrac=%.3e badTol=%.3e "
    "maxAbs=%.3e worstAbsRow=%lu denomFloor=%.3e\n",
    (int)ok, q, (double)relQ, relTol, badFrac,
    (double)BF_SPARSE_SVD_ROWSUM_BAD_FRAC_TOL,
    maxAbs, (unsigned long)worstAbsI, denomFloor);

  bfMemFree(relErr);

  (void)worstAbsI;

  bfMemFree(rs_csr);
  bfMemFree(rs_svd);
  bfMemFree(v);
  bfMemFree(t);

  return ok;
}

/* Compute p-quantile of y (0<=p<=1). Uses full argsort via bfRealArgsort. */
static BfReal bfSparseSvdQuantile(BfReal const *y, BfSize n, double p) {
  if (n == 0) return 0;

  /* clamp */
  if (p < 0.0) p = 0.0;
  if (p > 1.0) p = 1.0;

  BfSize *perm = bfMemAlloc(n, sizeof(BfSize));
  if (perm == NULL) {
    /* fallback: return something safe without sorting */
    return y[n - 1];
  }

  bfRealArgsort(y, n, perm);

  /* endpoints */
  if (p <= 0.0) {
    BfReal q = y[perm[0]];
    bfMemFree(perm);
    return q;
  }
  if (p >= 1.0) {
    BfReal q = y[perm[n - 1]];
    bfMemFree(perm);
    return q;
  }

  /* nearest-rank index in [0, n-1] */
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

static bool bfSparseSvdUseFrobTol(void) {
  static int init = 0;
  static int use  = 0;
  if (!init) {
    init = 1;
    char const *s = getenv("BF_TRUNC_TOL_FROBENIUS");
    if (s && *s) {
      use = (atoi(s) != 0);
    }
    SPARSE_SVD_LOG("[bf] sparse SVD: BF_TRUNC_TOL_FROBENIUS=%d\n", use);
  }
  return use != 0;
}

/* Choose smallest k such that sqrt(tail/total) <= tol, where
 * total = sum_i sigma_i^2, tail = sum_{i>=k} sigma_i^2.
 * Returns k in [0..S->numElts].
 */
static BfSize bfTruncSpecGetNumTermsFrob(BfTruncSpec const *truncSpec,
                                        BfMatDiagReal const *S) {
  if (S == NULL || S->numElts == 0) return 0;
  if (truncSpec == NULL || !truncSpec->usingTol) return 0;

  double tol = (double)truncSpec->tol;
  if (!(tol > 0.0) || !isfinite(tol)) return S->numElts;

  double total = 0.0;
  for (BfSize i = 0; i < S->numElts; ++i) {
    double si = (double)S->data[i];
    if (!isfinite(si) || si < 0) break;
    total += si*si;
  }
  if (!(total > 0.0)) return 0;

  double tail = total;
  for (BfSize k = 0; k < S->numElts; ++k) {
    double sk = (double)S->data[k];
    tail -= sk*sk;
    if (tail < 0.0) tail = 0.0;
    double relTail = sqrt(tail/total);
    if (relTail <= tol) return k + 1;
  }

  return S->numElts;
}

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

  V = bfMemAlloc((BfSize)N*(BfSize)ncv, sizeof(BfReal));
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
#if !BF_HAVE_PRIMME_SVDS
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
#endif /* !BF_HAVE_PRIMME_SVDS */

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

  BF_ASSERT(truncSpec != NULL);

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

#if BF_SPARSE_SVD_STATS
bfSparseSvdInstallAtexitOnce();
BF_SVDSTAT_INC(gSparseSvd_leafTotal);
bfSparseSvdMaybePrintStats();
#endif

if (nnz == 0) {
  SPARSE_SVD_LOG("[bf] sparse SVD: skipping zero block (nnz=0), keep sparse/zero leaf\n");
#if BF_SPARSE_SVD_STATS
  BF_SVDSTAT_INC(gSparseSvd_skip_nnz0);
  bfSparseSvdMaybePrintStats();
#endif
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
#if BF_SPARSE_SVD_STATS
  BF_SVDSTAT_INC(gSparseSvd_skip_numZero);
  bfSparseSvdMaybePrintStats();
#endif
  return false;
}

if (maxAbs < 1e-11) {
  SPARSE_SVD_LOG(
    "[bf] sparse SVD: block too small to bother (max|a_ij|=%.3e), keeping CSR\n",
    (double)maxAbs);
#if BF_SPARSE_SVD_STATS
  BF_SVDSTAT_INC(gSparseSvd_skip_tiny);
  bfSparseSvdMaybePrintStats();
#endif
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

  /* Memory budget based on FULL CSR bytes (data + indices).
   * This aligns the maxRank cap with the later accept/reject bytes gate.
   */
  {
    double csrBytesFull = bfSparseSvdEstimateBytesCsr(m, nnz);
    double bytesPerRank = (double)(m + n + 1) * (double)sizeof(BfReal);
    BfSize maxRankByBytes = maxRank;

    if (bytesPerRank > 0) {
      maxRankByBytes = (BfSize)(csrBytesFull / bytesPerRank);

      if (maxRankByBytes == 0 && nnz > 0)
        maxRankByBytes = 1;

      if (maxRank > maxRankByBytes)
        maxRank = maxRankByBytes;
    }

    SPARSE_SVD_LOG("[bf] sparse SVD: mem budget (full CSR): csrBytesFull=%.3e bytesPerRank=%.3e -> maxRankByBytes=%lu, chosen maxRank=%lu\n",
                   csrBytesFull,
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

  /* --- PRIMME_SVDS backend: get singular values & right singular vectors --- */

  /* Use an explicit initSize so we can parse svecs deterministically as:
   *   leftBlock  = U  (m x initSize_in)
   *   rightBlock = V  (n x initSize_in)
   * stored as two contiguous column-major blocks.
   */
  PRIMME_INT initSize_in = (PRIMME_INT)maxRank;
  if (initSize_in <= 0) initSize_in = 1;

  svals    = bfMemAlloc((BfSize)initSize_in, sizeof(BfReal));
  resNorms = bfMemAlloc((BfSize)initSize_in, sizeof(BfReal));
  svecs    = bfMemAlloc((BfSize)initSize_in * (BfSize)(m + n), sizeof(BfReal));

  if (svals == NULL || svecs == NULL || resNorms == NULL)
    RAISE_ERROR(BF_ERROR_MEMORY_ERROR);

  primme_svds_params primme;
  primme_svds_initialize(&primme);

  /* A) Fix PRIMME svecs parsing first: be explicit */
  primme.numOrthoConst = 0;
  primme.initSize      = initSize_in;

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

  /* Prefer PRIMME's reported count (if it returns it via initSize),
   * otherwise fall back to scanning svals for finiteness.
   */
  int numConv = 0;

  /* Determine how many usable modes we actually got.
   * Do NOT infer this from primme.initSize (input parameter); instead
   * scan returned outputs for sanity.
   */
  for (int j = 0; j < primme.numSvals; ++j) {
    if (!bfIsFiniteReal(svals[j])) break;
    if (svals[j] <= 0) break;
    if (resNorms != NULL && !bfIsFiniteReal(resNorms[j])) break;
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
#if BF_SPARSE_SVD_STATS
    BF_SVDSTAT_INC(gSparseSvd_reject_solver);
    bfSparseSvdMaybePrintStats();
#endif
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

  /* A) Parse svecs as two contiguous blocks:
   *   leftBlock:  m x initSize_in (U)
   *   rightBlock: n x initSize_in (V)
   * both stored column-major (each vector contiguous).
   */
  const BfReal *leftBlock  = (const BfReal *)svecs;
  const BfReal *rightBlock = (const BfReal *)svecs + (BfSize)initSize_in * (BfSize)m;

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
    BfReal relTol = 1e-4;
    BfReal absTol = 100*DBL_EPSILON;
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

    /* Right singular vector v_j is column j of rightBlock (length n) */
    const BfReal *vj = rightBlock + (BfSize)j*(BfSize)n;
    for (BfSize i = 0; i < n; ++i)
      Z[j*n + i] = vj[i];
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

#if BF_SPARSE_SVD_STATS
    BF_SVDSTAT_INC(gSparseSvd_reject_solver);
    bfSparseSvdMaybePrintStats();
#endif
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

#if BF_SPARSE_SVD_SOLVER_SANITY
  /* Optional: quick orthogonality sanity check for first few right vectors */
  {
    BfSize kchk = maxRank;
    if (kchk > (BfSize)BF_SPARSE_SVD_SANITY_MAXK) kchk = (BfSize)BF_SPARSE_SVD_SANITY_MAXK;

    for (BfSize a = 0; a < kchk; ++a) {
      for (BfSize b = 0; b < a; ++b) {
        double dot = 0.0;
        const BfReal *va = &ZSorted[a*n];
        const BfReal *vb = &ZSorted[b*n];
        for (BfSize i = 0; i < n; ++i) dot += (double)va[i] * (double)vb[i];

        if (fabs(dot) > (double)BF_SPARSE_SVD_SANITY_ORTHO_TOL) {
          SPARSE_SVD_LOG(
            "[bf] sparse SVD: solver sanity reject (V-ortho): a=%lu b=%lu dot=%.3e tol=%.3e\n",
            (unsigned long)a, (unsigned long)b, dot, (double)BF_SPARSE_SVD_SANITY_ORTHO_TOL);
#if BF_SPARSE_SVD_STATS
          BF_SVDSTAT_INC(gSparseSvd_reject_solver);
          bfSparseSvdMaybePrintStats();
#endif
          truncated = false;
          goto cleanup;
        }
      }
    }
  }
#endif

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

    SPARSE_SVD_LOG("[bf] sparse SVD: rejecting because largest singular value is non-finite or <= 0\n");
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
    if (bfSparseSvdUseFrobTol())
      k = bfTruncSpecGetNumTermsFrob(truncSpec, Sfull);
    else
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
    if (bfSparseSvdUseFrobTol()) {
      /* Frobenius-tail tol: if we used all available modes and still
       * don't meet tail/total <= tol, then we didn't “achieve tol”.
       */
      double total = 0.0;
      for (BfSize i = 0; i < maxRank; ++i) {
        double si = (double)Sfull->data[i];
        total += si*si;
      }
      /* With Frobenius-tail tol, if we used all computed modes (k==maxRank),
       * we cannot certify the remaining tail beyond maxRank. Be conservative
       * and mark tol as not achieved unless the block is effectively zero.
       */
      if (total > 0.0) {
        achievedTol = false;
        SPARSE_SVD_LOG(
          "[bf] sparse SVD: cannot certify Frobenius-tail tol with k==maxRank "
          "(maxRank=%lu, tol=%.3e); treating as not achieved\n",
          (unsigned long)maxRank, (double)truncSpec->tol);
      }
    } else {
      /* Spectral tol: s_last/s0 <= tol */
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
  }

  SPARSE_SVD_LOG("[bf] sparse SVD: achievedTol=%d\n", (int)achievedTol);

#if BF_SPARSE_SVD_STRICT_TOL
  /* Strict semantics: reject if tol not achieved before maxRank */
  if (truncSpec != NULL && truncSpec->usingTol && !achievedTol) {
    SPARSE_SVD_LOG(
      "[bf] sparse SVD: rejecting SVD because tol not achieved (maxRank=%lu, tol=%.3e)\n",
      (unsigned long)maxRank, (double)truncSpec->tol);
#if BF_SPARSE_SVD_STATS
    BF_SVDSTAT_INC(gSparseSvd_reject_notAchTol);
    bfSparseSvdMaybePrintStats();
#endif
    truncated = false;
    goto cleanup;
  }
#else
  /* Soft semantics (fluxpy-like): log, but allow acceptance if it passes
   * bytes + row-sum + physics gates.
   */
  if (truncSpec != NULL && truncSpec->usingTol && !achievedTol) {
    SPARSE_SVD_LOG(
      "[bf] sparse SVD: tol not achieved before maxRank (maxRank=%lu, tol=%.3e); "
      "continuing with bytes/row-sum/physics gates\n",
      (unsigned long)maxRank, (double)truncSpec->tol);
  }
#endif

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
#if BF_SPARSE_SVD_STATS
      BF_SVDSTAT_INC(gSparseSvd_reject_bytes);
      bfSparseSvdMaybePrintStats();
#endif
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

#if BF_SPARSE_SVD_SOLVER_SANITY
    /* Sanity: ||v||~1 and ||A v||~sigma (before scaling) */
    {
      double nv2 = 0.0;
      for (BfSize i = 0; i < n; ++i) {
        double vi = (double)ZSorted[j*n + i];
        nv2 += vi*vi;
      }
      double nv = sqrt(nv2);

      double nAv2 = 0.0;
      for (BfSize i = 0; i < m; ++i) {
        double yi = (double)tmpReal->data[i*tmpReal->stride];
        nAv2 += yi*yi;
      }
      double nAv = sqrt(nAv2);

      if (fabs(nv - 1.0) > (double)BF_SPARSE_SVD_SANITY_VNORM_TOL) {
        SPARSE_SVD_LOG(
          "[bf] sparse SVD: solver sanity reject: ||v||=%.6e (tol=%.3e) j=%lu\n",
          nv, (double)BF_SPARSE_SVD_SANITY_VNORM_TOL, (unsigned long)j);
#if BF_SPARSE_SVD_STATS
        BF_SVDSTAT_INC(gSparseSvd_reject_solver);
        bfSparseSvdMaybePrintStats();
#endif
        truncated = false;
        goto cleanup;
      }

      if (sj > 0) {
        double rel = fabs(nAv - (double)sj) / (double)sj;
        if (rel > (double)BF_SPARSE_SVD_SANITY_SIG_REL_TOL) {
          SPARSE_SVD_LOG(
            "[bf] sparse SVD: solver sanity reject: ||A v||=%.6e sigma=%.6e rel=%.3e tol=%.3e j=%lu\n",
            nAv, (double)sj, rel, (double)BF_SPARSE_SVD_SANITY_SIG_REL_TOL, (unsigned long)j);
#if BF_SPARSE_SVD_STATS
          BF_SVDSTAT_INC(gSparseSvd_reject_solver);
          bfSparseSvdMaybePrintStats();
#endif
          truncated = false;
          goto cleanup;
        }
      }
    }
#endif

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

#if BF_SPARSE_SVD_ROWSUM_CHECK
  /* ---- Row-sum gate: preserve A*1 (energy-ish) ------------------------- */
  if (!bfSparseSvdCheckRowSums(Acsr, U, S, VT, truncSpec, k)) {

#if BF_SPARSE_SVD_ROWSUM_REPAIR
    SPARSE_SVD_LOG("[bf] sparse SVD: row-sum drift detected -> attempting rank-1 repair\n");

    /* Build d = rs_csr - rs_svd (optionally clamp to deficits only),
     * then append a rank-1 term:  A += d * (1/n) * 1^T
     *
     * Implementation: add one extra “mode”
     *   - new column in U: u0 = d
     *   - new diagonal entry in S: s0 = 1
     *   - new row in VT: v0^T = (1/n) * 1^T
     *
     * NOTE: this is not orthonormal SVD anymore, but BF’s matvec doesn’t
     * require orthonormal factors; it just multiplies U*S*VT.
     */

    /* 1) Recompute row sums here (cheap vs rejecting) */
    {
      BfMat const *Amat2 = bfMatCsrRealConstToMatConst(Acsr);
      BfSize m2 = bfMatGetNumRows(Amat2);
      BfSize n2 = bfMatGetNumCols(Amat2);

      BfReal *d = bfMemAlloc(m2, sizeof(BfReal));
      if (d == NULL) {
        SPARSE_SVD_LOG("[bf] sparse SVD: repair failed (alloc d)\n");
        goto rowsum_reject;
      }

      /* Get rs_csr and rs_svd via existing checker’s internal math:
       * easiest: compute A*1 for CSR and SVD right here.
       */

      /* CSR rs = A * 1 */
      BfSize const *rp = bfMatCsrRealGetRowptrConstPtr(Acsr);
      BfReal const *ad = bfMatCsrRealGetDataConstPtr(Acsr);
      for (BfSize i = 0; i < m2; ++i) {
        BfReal s = 0;
        for (BfSize p = rp[i]; p < rp[i + 1]; ++p) s += ad[p];
        d[i] = s;
      }

      /* SVD rs = U*S*VT*1 */
      BfReal *v1 = bfMemAlloc(k, sizeof(BfReal));
      BfReal *t1 = bfMemAlloc(k, sizeof(BfReal));
      if (v1 == NULL || t1 == NULL) {
        bfMemFree(d);
        if (v1) bfMemFree(v1);
        if (t1) bfMemFree(t1);
        SPARSE_SVD_LOG("[bf] sparse SVD: repair failed (alloc v1/t1)\n");
        goto rowsum_reject;
      }

      for (BfSize j = 0; j < k; ++j) {
        BfVecReal *row = bfMatDenseRealGetRowView(VT, j);
        if (row == NULL) { bfMemFree(d); bfMemFree(v1); bfMemFree(t1); goto rowsum_reject; }
        BfReal sum = 0;
        for (BfSize i = 0; i < n2; ++i) sum += row->data[i*row->stride];
        v1[j] = sum;
        bfVecRealDeinitAndDealloc(&row);
      }

      for (BfSize j = 0; j < k; ++j) t1[j] = S->data[j] * v1[j];

      for (BfSize i = 0; i < m2; ++i) {
        BfVecReal *ui = bfMatDenseRealGetRowView(U, i);
        if (ui == NULL) { bfMemFree(d); bfMemFree(v1); bfMemFree(t1); goto rowsum_reject; }
        BfReal sum = 0;
        for (BfSize j = 0; j < k; ++j) sum += ui->data[j*ui->stride] * t1[j];
        /* d <- rs_csr - rs_svd */
        d[i] = d[i] - sum;
#if BF_SPARSE_SVD_ROWSUM_REPAIR_ONLY_DEFICIT
        if (d[i] < 0) d[i] = 0;
#endif
        bfVecRealDeinitAndDealloc(&ui);
      }

      bfMemFree(v1);
      bfMemFree(t1);

      /* 2) Allocate expanded factors */
      BfMatDenseReal *U2  = bfMatDenseRealNew();
      BfMatDiagReal  *S2  = bfMatDiagRealNew();
      BfMatDenseReal *VT2 = bfMatDenseRealNew();

      bfMatDenseRealInit(U2, m2, k + 1);
      bfMatDiagRealInit(S2, k + 1, k + 1);
      bfMatDenseRealInit(VT2, k + 1, n2);

      /* copy old factors */
      for (BfSize j = 0; j < k; ++j) {
        /* copy col j of U */
        BfVec *colU = bfMatDenseRealGetColView(bfMatDenseRealToMat(U), j);
        bfMatDenseRealSetCol(U2, j, colU);
        bfVecDelete(&colU);

        S2->data[j] = S->data[j];

        /* copy row j of VT */
        BfVecReal *rowVT = bfMatDenseRealGetRowView(VT, j);
        bfMatDenseRealSetRow(bfMatDenseRealToMat(VT2), j, bfVecRealToVec(rowVT));
        bfVecRealDeinitAndDealloc(&rowVT);
      }

      /* append repair mode */
      S2->data[k] = 1.0;

      /* U2 last column = d */
      {
        BfVecReal dv;
        bfVecRealInitView(&dv, m2, BF_DEFAULT_STRIDE, d);
        bfMatDenseRealSetCol(U2, k, bfVecRealToVec(&dv));
      }

      /* VT2 last row = (1/n) * 1^T */
      {
        BfReal *ones = bfMemAlloc(n2, sizeof(BfReal));
        if (ones == NULL) {
          /* Avoid leaking expanded factors on this failure path */
          bfMemFree(d);
          bfMatDenseRealDeinitAndDealloc(&U2);
          bfMatDenseRealDeinitAndDealloc(&VT2);
          bfMatDiagRealDeinitAndDealloc(&S2);
          goto rowsum_reject;
        }
        BfReal invn = (n2 > 0) ? (1.0/(BfReal)n2) : 0.0;
        for (BfSize i = 0; i < n2; ++i) ones[i] = invn;
        BfVecReal ov;
        bfVecRealInitView(&ov, n2, BF_DEFAULT_STRIDE, ones);
        bfMatDenseRealSetRow(bfMatDenseRealToMat(VT2), k, bfVecRealToVec(&ov));
        bfMemFree(ones);
      }

      bfMemFree(d);

      /* swap in repaired factors */
      bfMatDenseRealDeinitAndDealloc(&U);
      bfMatDenseRealDeinitAndDealloc(&VT);
      bfMatDiagRealDeinitAndDealloc(&S);

      U  = U2;
      VT = VT2;
      S  = S2;
      k  = k + 1;

      /* re-check row sums: if still bad, reject */
      if (!bfSparseSvdCheckRowSums(Acsr, U, S, VT, truncSpec, k)) {
        SPARSE_SVD_LOG("[bf] sparse SVD: repair failed to satisfy row-sum gate\n");
        goto rowsum_reject;
      }
    }

    SPARSE_SVD_LOG("[bf] sparse SVD: repair succeeded\n");

    /* NEW: re-run bytes gate after row-sum repair.
     * The repair increments k, so bytes_svd can cross the “worth it” threshold.
     * If it does, revert to CSR (i.e., reject this SVD leaf).
     */
    {
      double bytes_csr = bfSparseSvdEstimateBytesCsr(m, nnz);
      double bytes_svd = bfSparseSvdEstimateBytesSvd(m, n, k);

      SPARSE_SVD_LOG(
        "[bf] sparse SVD: bytes after repair: csr=%.3e svd(k=%lu)=%.3e (ratio=%.3f)\n",
        bytes_csr, (unsigned long)k, bytes_svd,
        (bytes_csr > 0 ? bytes_svd/bytes_csr : 1.0));

      if (bytes_svd >= BF_SPARSE_SVD_MIN_SAVINGS_FRAC * bytes_csr) {
        SPARSE_SVD_LOG(
          "[bf] sparse SVD: rejecting repaired SVD (not worth it after repair): %.3e >= %.3e * %.3e\n",
          bytes_svd, (double)BF_SPARSE_SVD_MIN_SAVINGS_FRAC, bytes_csr);

#if BF_SPARSE_SVD_STATS
        BF_SVDSTAT_INC(gSparseSvd_reject_bytes);
        bfSparseSvdMaybePrintStats();
#endif
        truncated = false;
        goto cleanup; /* frees repaired U/S/VT; caller keeps CSR */
      }
    }

    goto rowsum_ok;


rowsum_reject:
    SPARSE_SVD_LOG("[bf] sparse SVD: rejecting SVD due to row-sum drift\n");
#if BF_SPARSE_SVD_STATS
    BF_SVDSTAT_INC(gSparseSvd_reject_rowSum);
    bfSparseSvdMaybePrintStats();
#endif
    truncated = false;
    goto cleanup;

rowsum_ok:
    ; /* continue */

#else
    SPARSE_SVD_LOG("[bf] sparse SVD: rejecting SVD due to row-sum drift\n");
#if BF_SPARSE_SVD_STATS
    BF_SVDSTAT_INC(gSparseSvd_reject_rowSum);
    bfSparseSvdMaybePrintStats();
#endif
    truncated = false;
    goto cleanup;
#endif
  }
#endif

#if BF_SPARSE_SVD_PHYSICS_PROBE
  /* ---- Physics probe gate: A_svd * x should be ~nonnegative for x>=0 ----
   * This is intentionally cheap and catches catastrophic non-physical SVDs.
   */
  {
    /* eta computed later as eta_abs; keep single source of truth */

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
          if (vj == NULL) { truncated = false; goto probe_cleanup; }

          BfReal dot = 0;
          for (BfSize i = 0; i < n; ++i)
            dot += vj->data[i*vj->stride] * x[i];

          t[j] = S->data[j] * dot;

          bfVecRealDeinitAndDealloc(&vj);
        }

        /* y = U * t */
        for (BfSize i = 0; i < m; ++i) {
          BfVecReal *ui = bfMatDenseRealGetRowView(U, i);
          if (ui == NULL) { truncated = false; goto probe_cleanup; }

          BfReal sum = 0;
          for (BfSize j = 0; j < k; ++j)
            sum += ui->data[j*ui->stride] * t[j];

          y[i] = sum;

          bfVecRealDeinitAndDealloc(&ui);
        }


        /* Compute min/max and negative/positive “mass” */
        BfReal yMin = BF_INFINITY;
        BfReal yMax = 0;
        double sumPos = 0.0;
        double sumNeg = 0.0;

        for (BfSize i = 0; i < m; ++i) {
          double yi = (double)y[i];
          if ((BfReal)yi < yMin) yMin = (BfReal)yi;
          if ((BfReal)yi > yMax) yMax = (BfReal)yi;

          if (yi >= 0.0) sumPos += yi;
          else           sumNeg += -yi; /* accumulate magnitude of negatives */
        }

        double negMass = 0.0;
        if (sumPos > 0.0) negMass = sumNeg / sumPos;
        else if (sumNeg > 0.0) negMass = BF_INFINITY;

        /* If output is (near-)zero, don't reject on “negativity fractions”.
         * This avoids insane ratios when sumPos≈0 and also avoids pointless
         * rejections on numerically-null leaves.
         */
        double yAbsMax = fmax(fabs((double)yMin), (double)yMax);
        double sumTot  = sumPos + sumNeg;

        /* Option C: near-zero leaf auto-accept */
        if ((double)yMax < (double)BF_SPARSE_SVD_PHYS_Y_FLOOR ||
            yAbsMax      < (double)BF_SPARSE_SVD_PHYS_Y_FLOOR ||
            sumTot       < (double)BF_SPARSE_SVD_PHYS_Y_FLOOR) {
          SPARSE_SVD_LOG(
            "[bf] sparse SVD: physics probe=%d near-zero output (yMax=%.3e yAbsMax=%.3e sumTot=%.3e) -> auto-pass\n",
            probe, (double)yMax, yAbsMax, sumTot);
          continue;
        }

        /* Option A: robust scale so it doesn’t go tiny on weak-coupling leaves */
        BfReal yP99 = 0;

        /* default fallback scale from max magnitude */
        BfReal scale = (yMax > 0 ? yMax : (BfReal)yAbsMax);
        if (scale <= 0) scale = 1.0;

        BfReal *yCopy = bfMemAlloc(m, sizeof(BfReal));
        if (yCopy != NULL) {
          for (BfSize i = 0; i < m; ++i) yCopy[i] = y[i];
          yP99 = bfSparseSvdQuantile(yCopy, m, 0.99);
          bfMemFree(yCopy);
        }

        /* scale = max(yP99, 0.1*yMax, scale_floor) */
        {
          BfReal s = scale;

          if (yP99 > 0) s = yP99;

          BfReal s2 = (BfReal)(0.1 * (double)yMax);
          if (s2 > s) s = s2;

          BfReal floorS = (BfReal)BF_SPARSE_SVD_PHYS_SCALE_FLOOR;
          if (floorS > s) s = floorS;

          scale = s;
        }

        /* Thresholds:
         *  - eta_abs: min-value floor relative to scale (magnitude-aware)
         *  - eta_mass: allowed total negative “mass” fraction
         */
        double eta_abs  = bfSparseSvdNegEtaFromTol(truncSpec);
        double eta_mass = bfSparseSvdNegMassEtaFromTol(truncSpec);

        SPARSE_SVD_LOG(
          "[bf] sparse SVD: physics probe=%d "
          "yMin=%.3e yP99=%.3e yMax=%.3e scale=%.3e "
          "negMass=%.3e (eta_mass=%.3e) eta_abs=%.3e\n",
          probe,
          (double)yMin, (double)yP99, (double)yMax, (double)scale,
          negMass, eta_mass, eta_abs);

        /* Reject only if negatives are meaningful in magnitude */
        if (negMass > eta_mass || (double)yMin < -eta_abs*(double)scale) {
          SPARSE_SVD_LOG("[bf] sparse SVD: physics probe reject\n");
#if BF_SPARSE_SVD_STATS
          BF_SVDSTAT_INC(gSparseSvd_reject_physics);
          bfSparseSvdMaybePrintStats();
#endif
          truncated = false;
          goto cleanup;
        }
      }

probe_done:
      bfMemFree(x);
      bfMemFree(y);
      bfMemFree(t);
      goto probe_after;

probe_cleanup:
      /* Ensure no leaks on early abort inside the probe */
      if (x) bfMemFree(x);
      if (y) bfMemFree(y);
      if (t) bfMemFree(t);
      goto cleanup;

probe_after:
      ;
    }
  }
#endif


  *UPtr  = bfMatDenseRealToMat(U);
  *SPtr  = S;
  *VTPtr = bfMatDenseRealToMat(VT);

#if BF_SPARSE_SVD_STATS
  BF_SVDSTAT_INC(gSparseSvd_acceptSvd);
  bfSparseSvdMaybePrintStats();
#endif

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

