#include <bf/def.h>
#include <bf/size_array.h>
#include <bf/ptr_array.h>
#include <bf/quadtree.h>
#include <bf/trimesh.h>
#include <bf/mat_csr_real.h>
#include <bf/vec_real.h>
#include <bf/vf_hier.h>

/* A Mat wrapper around the hierarchical view-factor operator */

typedef struct BfMatVfHier {
  BfMat  super;   /* must be first: upcasting works */
  BfVfHier *vfHier;
} BfMatVfHier;

/* Creation / destruction */
BfMatVfHier *bfMatVfHierNew(void);

void bfMatVfHierInitFromQuadtree(BfMatVfHier   *mat,
                                 BfTrimesh const *trimesh,
                                 BfQuadtree      *quadtree,
                                 BfReal           eta,
                                 BfSize           leafMax,
                                 BfSize           leafMin,
                                 BfReal           tol,
                                 BfSize           minSvdSize,
                                 BfReal           maxSvdRankFrac);

BfMat *bfMatVfHierNewFromQuadtree(BfTrimesh const *trimesh,
                                  BfQuadtree      *quadtree,
                                  BfReal           eta,
                                  BfSize           leafMax,
                                  BfSize           leafMin,
                                  BfReal           tol,
                                  BfSize           minSvdSize,
                                  BfReal           maxSvdRankFrac);

/* New: CSR + quadtree constructors */

void bfMatVfHierInitFromCsrAndQuadtree(BfMatVfHier  *mat,
                                       BfMatCsrReal *Afull,
                                       BfQuadtree   *quadtree,
                                       BfReal        eta,
                                       BfSize        leafMax,
                                       BfSize        leafMin,
                                       BfReal        tol,
                                       BfSize        minSvdSize,
                                       BfReal        maxSvdRankFrac);

BfMat *bfMatVfHierNewFromCsrAndQuadtree(BfMatCsrReal *Afull,
                                        BfQuadtree   *quadtree,
                                        BfReal        eta,
                                        BfSize        leafMax,
                                        BfSize        leafMin,
                                        BfReal        tol,
                                        BfSize        minSvdSize,
                                        BfReal        maxSvdRankFrac);

/* Up/down-cast helpers */
BfMat       *bfMatVfHierToMat(BfMatVfHier *mat);
BfMatVfHier *bfMatToMatVfHier(BfMat *mat);