#include <bf/lbo.h>

#include <bf/assert.h>
#include <bf/const.h>
#include <bf/error.h>
#include <bf/error_macros.h>
#include <bf/interval_tree.h>
#include <bf/interval_tree_node.h>
#include <bf/linalg.h>
#include <bf/util.h>
#include <bf/vec_real.h>

#include <math.h>

BfPoints1 *bfLboEigsToFreqs(BfVecReal const *Lam) {
  BF_ERROR_BEGIN();

  BfPoints1 *freqs = bfPoints1New();
  HANDLE_ERROR();

  bfPoints1InitEmpty(freqs, BF_ARRAY_DEFAULT_CAPACITY);
  HANDLE_ERROR();

  for (BfSize i = 0; i < Lam->super.size; ++i) {
    /* Clamp eigenvalues to be nonnegative if we find any that are
     * negative due to roundoff */
    if (Lam->data[i] < 0)
      Lam->data[i] = 0;

    bfPoints1Append(freqs, sqrt(Lam->data[i]));
    HANDLE_ERROR();
  }

  BF_ERROR_END() {
    BF_DIE();
  }

  return freqs;
}

static BfInterval getBracketFromNode(BfTreeNode const *treeNode) {
  BF_ERROR_BEGIN();

  BfIntervalTreeNode const *intervalTreeNode = bfTreeNodeConstToIntervalTreeNodeConst(treeNode);
  HANDLE_ERROR();

  bool const left = intervalTreeNode->isLeftmost;
  bool const right = intervalTreeNode->isRightmost;

  /* Set up the bracket for computing the next eigenband: */
  BfInterval bracket = {
    .endpoint = {
      pow(intervalTreeNode->a, 2.0),
      pow(intervalTreeNode->b, 2.0)
    },
    .closed = {false, true}
  };

  /* TODO: do this less dumb */
  const double PADDING = 1e-5;
  if (left) bracket.endpoint[0] -= PADDING;
  if (right) bracket.endpoint[1] += PADDING;

  BF_ERROR_END() {
    BF_DIE();
  }

  return bracket;
}

BfLboFeedResult bfLboFeedFacStreamerNextEigenband(BfFacStreamer *facStreamer, BfPoints1 *freqs, BfMat const *L, BfMat const *M) {
  BF_ERROR_BEGIN();

  BfReal t0_total = bfTime();

  BfLboFeedResult result;

  BfMat *Phi = NULL;
  BfVecReal *Lam = NULL;
  BfPoints1 *newFreqs = NULL;

  BfTreeNode *treeNode = bfFacStreamerGetCurrentColumnNode(facStreamer);
  if (!bfTreeNodeIsLeaf(treeNode))
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);

  BfTree *tree = bfTreeNodeGetTree(treeNode);

  BfIntervalTree *intervalTree = bfTreeToIntervalTree(tree);
  HANDLE_ERROR();

  /* The current node should be empty since we're incrementally
   * constructing the column tree as we stream the band
   * eigenvalues. */
  if (!bfTreeNodeIsEmpty(treeNode))
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);

  BfInterval bracket = getBracketFromNode(treeNode);
  HANDLE_ERROR();

  /* Compute the next eigenband using Lanczos */
  BfReal t0_eigs = bfTime();
  bfGetEigenband(L, M, &bracket, BF_EIGENBAND_METHOD_COVERING, &Phi, &Lam);
  result.eigenbandTime = bfTime() - t0_eigs;
  HANDLE_ERROR();

  /* Permute the rows of Phi, putting them into row tree order */
  bfMatPermuteRows(Phi, bfFacStreamerGetRowTreeReversePerm(facStreamer));

  /* Convert the new eigenvalues to frequencies: */
  newFreqs = bfLboEigsToFreqs(Lam);
  HANDLE_ERROR();

  /* Sort the new frequencies into the existing set of frequencies
   *
   * TODO: because of the way we're doing this, we can guarantee that
   * we only need to append the new frequencies to the end of `freqs`,
   * simplifying things a bit. */
  bfPoints1InsertPointsSorted(freqs, newFreqs);
  HANDLE_ERROR();

  /* Update the tree's point set without rebuilding the tree. This has
   * the effect of adjusting the range of points each tree node points
   * to without actually changing the nodes in the tree. */
  bfIntervalTreeSetPoints(intervalTree, freqs, /* rebuildTree: */ false);
  HANDLE_ERROR();

  /* The current column tree node should now have exactly the same
   * number of points as the number of newly streamed eigenvalues. */
  if (bfTreeNodeGetNumPoints(treeNode) != Lam->super.size)
    RAISE_ERROR(BF_ERROR_RUNTIME_ERROR);

  /* Feed the streamed band of eigenvectors: */
  result.success = bfFacStreamerFeed(facStreamer, Phi);
  HANDLE_ERROR();

  BF_ERROR_END() {
    BF_DIE();
  }

  bfPoints1Delete(&newFreqs);
  bfMatDelete(&Phi);
  bfVecRealDeinitAndDealloc(&Lam);

  result.totalTime = bfTime() - t0_total;

  result.freqBand = bracket;
  result.freqBand.endpoint[0] = sqrt(result.freqBand.endpoint[0]);
  result.freqBand.endpoint[1] = sqrt(result.freqBand.endpoint[1]);

  return result;
}
