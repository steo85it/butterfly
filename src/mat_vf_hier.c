#include <bf/mat_vf_hier.h>   /* struct + prototypes */
#include <bf/vf_hier.h>   /* for BfVfHier, bfVfHierNewFromQuadtree, ... */
#include <bf/mat.h>
#include <bf/vec.h>
#include <bf/vec_real.h>
#include <bf/mem.h>
#include <bf/assert.h>

static BfMatVtable MAT_VTABLE;

/* Up/down-cast helpers */

BfMat *bfMatVfHierToMat(BfMatVfHier *mat) {
  return &mat->super;
}

BfMatVfHier *bfMatToMatVfHier(BfMat *mat) {
  if (!bfMatInstanceOf(mat, BF_TYPE_MAT_VF_HIER)) {
    bfSetError(BF_ERROR_TYPE_ERROR);
    return NULL;
  }
  return (BfMatVfHier *)mat;
}

/* --- Internal helpers --- */

static BfMat *MatVfHierGetView(BfMat *mat);
static void   MatVfHierDelete(BfMat **mat);
static BfType MatVfHierGetType(BfMat const *mat);
static BfSize MatVfHierGetNumRows(BfMat const *mat);
static BfSize MatVfHierGetNumCols(BfMat const *mat);
static BfVec *MatVfHierMulVec(BfMat const *mat, BfVec const *vec);

/* --- Constructor / init --- */

BfMatVfHier *bfMatVfHierNew(void) {
  BfMatVfHier *matVfHier = bfMemAlloc(1, sizeof(BfMatVfHier));
  matVfHier->vfHier = NULL;
  return matVfHier;
}

void bfMatVfHierInitFromQuadtree(BfMatVfHier   *matVfHier,
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
  BF_ASSERT(matVfHier != NULL);

  /* Build the hierarchical VF operator */
  BfVfHier *vf = bfVfHierNew();
  bfVfHierInitFromQuadtree(vf, trimesh, quadtree, eta, leafMax, leafMin, minArea, tol, minSvdSize, maxSvdRankFrac);

  matVfHier->vfHier = vf;

  /* Initialize the Mat base: square n x n */
  BfSize n = vf->n;
  bfMatInit(&matVfHier->super, &MAT_VTABLE, n, n);
}

/* Convenience builder returning BfMat* */
BfMat *bfMatVfHierNewFromQuadtree(BfTrimesh const *trimesh,
                                  BfQuadtree      *quadtree,
                                  BfReal           eta,
                                  BfSize           leafMax,
                                  BfSize           leafMin,
                                 BfReal           minArea,
                              BfReal           tol,
                              BfSize           minSvdSize,
                              BfReal           maxSvdRankFrac)
{
  BfMatVfHier *matVfHier = bfMatVfHierNew();
  bfMatVfHierInitFromQuadtree(matVfHier, trimesh, quadtree,
                              eta, leafMax, leafMin, minArea, tol, minSvdSize, maxSvdRankFrac);
  return bfMatVfHierToMat(matVfHier);
}

void bfMatVfHierInitFromCsrAndQuadtree(BfMatVfHier  *matVfHier,
                                       BfMatCsrReal *Afull,
                                       BfQuadtree   *quadtree,
                                       BfReal        eta,
                                       BfSize        leafMax,
                                       BfSize        leafMin,
                                       BfReal           minArea,
                                       BfReal        tol,
                                       BfSize        minSvdSize,
                                       BfReal        maxSvdRankFrac)
{
  BF_ASSERT(matVfHier != NULL);
  BF_ASSERT(Afull != NULL);

  BfVfHier *vf = bfVfHierNew();
  bfVfHierInitFromCsrAndQuadtree(vf, Afull, quadtree,
                                 eta, leafMax, leafMin, minArea,
                                 tol, minSvdSize, maxSvdRankFrac);

  matVfHier->vfHier = vf;

  BfSize n = vf->n;
  bfMatInit(&matVfHier->super, &MAT_VTABLE, n, n);
}

BfMat *bfMatVfHierNewFromCsrAndQuadtree(BfMatCsrReal *Afull,
                                        BfQuadtree   *quadtree,
                                        BfReal        eta,
                                        BfSize        leafMax,
                                        BfSize        leafMin,
                                        BfReal           minArea,
                                        BfReal        tol,
                                        BfSize        minSvdSize,
                                        BfReal        maxSvdRankFrac)
{
  BfMatVfHier *matVfHier = bfMatVfHierNew();
  bfMatVfHierInitFromCsrAndQuadtree(matVfHier, Afull, quadtree,
                                    eta, leafMax, leafMin, minArea,
                                    tol, minSvdSize, maxSvdRankFrac);
  return bfMatVfHierToMat(matVfHier);
}

/* --- Vtable impl --- */

static BfMat *MatVfHierGetView(BfMat *mat) {
  /* Shallow view: share the underlying vfHier, mark as view */
  BfMatVfHier *self = bfMatToMatVfHier(mat);
  BF_ASSERT(self != NULL);

  BfMatVfHier *view = bfMatVfHierNew();
  view->vfHier = self->vfHier; /* share; do NOT duplicate */

  BfMat *matView = bfMatVfHierToMat(view);
  matView->props |= BF_MAT_PROPS_VIEW;
  return matView;
}

static void MatVfHierDelete(BfMat **mat) {
  if (mat == NULL || *mat == NULL) return;
  BfMatVfHier *self = (BfMatVfHier *)(*mat);

  /* Only free the underlying hierarchy if this is not a view */
  if (!bfMatIsView(*mat) && self->vfHier != NULL) {
    bfVfHierDeinitAndDealloc(&self->vfHier);
  }

  bfMatDeinit(*mat);
  bfMemFree(self);
  *mat = NULL;
}

static BfType MatVfHierGetType(BfMat const *mat) {
  (void)mat;
  return BF_TYPE_MAT_VF_HIER;
}

static BfSize MatVfHierGetNumRows(BfMat const *mat) {
  BfMatVfHier const *self = (BfMatVfHier const *)mat;
  BF_ASSERT(self->vfHier != NULL);
  return self->vfHier->n;
}

static BfSize MatVfHierGetNumCols(BfMat const *mat) {
  BfMatVfHier const *self = (BfMatVfHier const *)mat;
  BF_ASSERT(self->vfHier != NULL);
  return self->vfHier->n;
}

static BfVec *MatVfHierMulVec(BfMat const *mat, BfVec const *vec) {
  BfMatVfHier const *self = (BfMatVfHier const *)mat;
  BF_ASSERT(self->vfHier != NULL);

  BfSize n = self->vfHier->n;
  BF_ASSERT(vec->size == n);
  BF_ASSERT(bfVecGetType(vec) == BF_TYPE_VEC_REAL);

  BfVecReal const *vxReal = bfVecConstToVecRealConst(vec);
  BfReal const *xData = bfVecRealGetDataPtr(vxReal);

  /* Allocate output vector y (real, size n) */
  BfVecReal *vyReal = bfVecRealNewWithValue(n, 0);
  BfReal *yData = bfVecRealGetDataPtr(vyReal);

  /* Actual hierarchical apply */
  bfVfHierApply(self->vfHier, xData, yData);

  return bfVecRealToVec(vyReal);
}

/* Optionally: a right-multiply (x^T A) if ever needed */
static BfVec *MatVfHierRmulVec(BfMat const *mat, BfVec const *vec) {
  (void)mat;
  (void)vec;
  bfSetError(BF_ERROR_NOT_IMPLEMENTED);
  return NULL;
}

/* --- Vtable definition --- */

static BfMatVtable MAT_VTABLE = {
  .GetView = MatVfHierGetView,
  .Copy = NULL,          /* optional; can be implemented later */
  .Steal = NULL,
  .GetRowCopy = NULL,
  .GetRowView = NULL,
  .GetColView = NULL,
  .GetColRangeView = NULL,
  .Delete = MatVfHierDelete,
  .EmptyLike = NULL,
  .ZerosLike = NULL,
  .GetType = MatVfHierGetType,
  .NumBytes = NULL,
  .Save = NULL,
  .Dump = NULL,
  .Print = NULL,
  .GetNumRows = MatVfHierGetNumRows,
  .GetNumCols = MatVfHierGetNumCols,
  .SetRow = NULL,
  .SetCol = NULL,
  .SetColRange = NULL,
  .GetRowRange = NULL,
  .GetColRange = NULL,
  .GetColRangeConst = NULL,
  .GetRowRangeCopy = NULL,
  .GetColRangeCopy = NULL,
  .SetRowRange = NULL,
  .PermuteRows = NULL,
  .PermuteCols = NULL,
  .RowDists = NULL,
  .ColDists = NULL,
  .ColDots = NULL,
  .ColNorms = NULL,
  .Scale = NULL,
  .ScaleRows = NULL,
  .ScaleCols = NULL,
  .SumCols = NULL,
  .Add = NULL,
  .AddInplace = NULL,
  .AddDiag = NULL,
  .Sub = NULL,
  .SubInplace = NULL,
  .Mul = NULL,
  .MulVec = MatVfHierMulVec,
  .MulInplace = NULL,
  .Rmul = NULL,
  .RmulVec = MatVfHierRmulVec,
  .Solve = NULL,
  .SolveLU = NULL,
  .LstSq = NULL,
  .IsUpperTri = NULL,
  .ForwardSolveVec = NULL,
  .BackwardSolveVec = NULL,
  .IsZero = NULL,
  .Negate = NULL,
  .ToType = NULL,
  .Cholesky = NULL,
  .GetNonzeroColumnRanges = NULL,
  .PrintBlocksDeep = NULL,
  .GetBlockView = NULL,
  .GetLu = NULL,
  .GetInverse = NULL,
  .DivideCols = NULL,
  .GetSubmatByMask = NULL,
  .Transpose = NULL,
  .NormMax = NULL,
  .DistMax = NULL,
};
