#include <bf/assert.h>
#include <bf/const.h>
#include <bf/error.h>
#include <bf/error_macros.h>
#include <bf/fac_span.h>
#include <bf/fac_streamer.h>
#include <bf/fiedler_tree.h>
#include <bf/interval_tree.h>
#include <bf/interval_tree_node.h>
#include <bf/lbo.h>
#include <bf/linalg.h>
#include <bf/logging.h>
#include <bf/mat_csr_real.h>
#include <bf/octree.h>
#include <bf/rand.h>
#include <bf/util.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "argtable3.h"

int const MAX_NUM_ARG_ERRORS = 20;

typedef struct {
  BfSize seed;
  BfLogLevel logLevel;

  char const *objPath;
  char const *matDirPath;

  bool computeMaxFreq;

  bool useOctree;
  bool useFiedlerTree;

  BfReal tol;
  BfReal freqMax;
  BfSize numLeafNodes;
  BfSize rowTreeInitDepth;
  BfSize freqTreeMaxDepth;
  BfSize freqTreeOffset;

  bool compareRelativeErrors;
  bool bailIfBottomedOut;
} Opts;

bool parseArgs(Opts *opts, int argc, char *argv[]) {
  bool success = true;

  struct arg_lit *help;

  struct arg_int *seed;
  struct arg_str *logLevel;

  struct arg_str *objPath;
  struct arg_str *matDirPath;

  struct arg_lit *computeMaxFreq;

  struct arg_lit *useOctree;
  struct arg_lit *useFiedlerTree;

  struct arg_dbl *tol;
  struct arg_dbl *freqMax;
  struct arg_int *numLeafNodes;
  struct arg_int *rowTreeInitDepth;
  struct arg_int *freqTreeMaxDepth;

  struct arg_lit *compareRelativeErrors;
  struct arg_lit *bailIfBottomedOut;

  struct arg_end *end;

  void *argtable[] = {
    help = arg_lit0(NULL, "help", "Display help and exit"),

    seed = arg_int0(NULL, "seed", NULL, "Seed for random number generator (default: 0)"),
    logLevel = arg_str0(NULL, "logLevel", NULL, "Log level (must be TODO, DEBUG, INFO, WARN, or ERROR)"),

    objPath = arg_str0(NULL, "objPath", NULL, "Path to Wavefront .obj file to load (must be watertight triangle mesh)"),
    matDirPath = arg_str0(NULL, "matDirPath", NULL, "Path to directory containing {L,M}_{rowptr,colind,data}.bin defining stiffness (L) and mass (M) matrices in CSR format"),

    computeMaxFreq = arg_lit0(NULL, "computeMaxFreq", "Compute the maximum frequency of the mesh and terminate"),

    useOctree = arg_lit0(NULL, "useOctree", "Use an octree for the row/space tree"),
    useFiedlerTree = arg_lit0(NULL, "useFiedlerTree", "Use a Fiedler tree for the row/space tree (experimental)"),

    tol = arg_dbl0(NULL, "tol", NULL, "Relative tolerance used for SVD truncation"),
    freqMax = arg_dbl0(NULL, "freqMax", NULL, "Build column/frequency tree on [0, freqMax] (instead of using maximum eigenvalue)"),
    numLeafNodes = arg_int0(NULL, "numLeafNodes", NULL, "Manually specify the number of leaf nodes to stream"),
    rowTreeInitDepth = arg_int0(NULL, "rowTreeInitDepth", NULL, "The level (>= 0, where 0 == root) to start from in the row/space tree when doing an adaptive column butterfly"),
    freqTreeMaxDepth = arg_int0(NULL, "freqTreeMaxDepth", NULL, "The maximum depth to which the frequency interval tree should be built"),

    compareRelativeErrors = arg_lit0(NULL, "compareRelativeErrors", NULL),
    bailIfBottomedOut = arg_lit0(NULL, "bailIfBottomedOut", NULL),

    end = arg_end(MAX_NUM_ARG_ERRORS)
  };

  *seed->ival = 0;
  *logLevel->sval = "error";
  *objPath->sval = "";
  *matDirPath->sval = "";

  computeMaxFreq->count = 0;

  useOctree->count = 1;
  useFiedlerTree->count = 0;

  *tol->dval = 1e-3;
  *freqMax->dval = BF_NAN;
  *numLeafNodes->ival = -1;
  *rowTreeInitDepth->ival = 0;
  *freqTreeMaxDepth->ival = -1;

  compareRelativeErrors->count = 0;
  bailIfBottomedOut->count = 0;

  BfSize numErrors = arg_parse(argc, argv, argtable);

  if (help->count > 0) {
    printf("Usage: %s\n", argv[0]);
    arg_print_syntax(stdout, argtable, "\n");
    printf("Test driver for butterfly compression of LBO\n");
    arg_print_glossary(stdout, argtable, "  %-25s %s\n");
    success = false;
    goto cleanup;
  }

  if (numErrors > 0) {
    arg_print_errors(stdout, end, argv[0]);
    printf("Try '%s --help' for more information.\n", argv[0]);
    success = false;
    goto cleanup;
  }

  if (computeMaxFreq->count == 0 && !(useOctree->count ^ useFiedlerTree->count)) {
    printf("Pass exactly one of --useOctree or --useFiedlerTree\n");
    success = false;
    goto cleanup;
  }

  if (computeMaxFreq->count > 0 && useOctree->count > 0) {
    printf("WARNING: Passed --computeMaxFreq: --useOctree will be ignored");
  }

  if (computeMaxFreq->count > 0 && useFiedlerTree->count > 0) {
    printf("WARNING: Passed --computeMaxFreq: --useOctree will be ignored");
  }

  if (!(strcmp(*objPath->sval, "") ^ strcmp(*matDirPath->sval, ""))) {
    printf("Exactly one of --objPath or --matDirPath should be set\n");
    success = false;
    goto cleanup;
  }

  if (strcmp(*matDirPath->sval, "") && useFiedlerTree->count) {
    printf("Argument --useFiedlerTree is only implemented with --objPath, not --matDirPath\n");
    success = false;
    goto cleanup;
  }

  opts->seed = *seed->ival;

  if (!strcmp(*logLevel->sval, "todo")) {
    opts->logLevel = BF_LOG_LEVEL_TODO;
  } else if (!strcmp(*logLevel->sval, "debug")) {
    opts->logLevel = BF_LOG_LEVEL_DEBUG;
  } else if (!strcmp(*logLevel->sval, "info")) {
    opts->logLevel = BF_LOG_LEVEL_INFO;
  } else if (!strcmp(*logLevel->sval, "warn")) {
    opts->logLevel = BF_LOG_LEVEL_WARN;
  } else if (!strcmp(*logLevel->sval, "error")) {
    opts->logLevel = BF_LOG_LEVEL_ERROR;
  } else {
    printf("--logLevel must be one of: \"todo\", \"debug\", \"info\", \"warn\", \"error\"\n");
    success = false;
    goto cleanup;
  }

  opts->objPath = *objPath->sval;
  opts->matDirPath  = *matDirPath->sval;

  opts->computeMaxFreq = computeMaxFreq->count > 0;

  opts->useOctree = useOctree->count > 0;
  opts->useFiedlerTree = useFiedlerTree->count > 0;

  opts->tol = *tol->dval;
  opts->freqMax = *freqMax->dval;
  opts->numLeafNodes = *numLeafNodes->ival;
  opts->rowTreeInitDepth = *rowTreeInitDepth->ival;

  if (*freqTreeMaxDepth->ival == -1) {
    opts->freqTreeMaxDepth = BF_SIZE_BAD_VALUE;
  } else {
    BF_ASSERT(*freqTreeMaxDepth->ival >= 0);
    opts->freqTreeMaxDepth = *freqTreeMaxDepth->ival;
  }

  opts->compareRelativeErrors = compareRelativeErrors->count > 0;
  opts->bailIfBottomedOut = bailIfBottomedOut->count > 0;

cleanup:
  arg_freetable(argtable, sizeof(argtable)/sizeof(argtable[0]));

  return success;
}

void printNode(BfTree const *tree, BfTreeNode const *treeNode, FILE *fp) {
  (void)tree;

  BfIntervalTreeNode const *intervalTreeNode = bfTreeNodeConstToIntervalTreeNodeConst(treeNode);

  fprintf(fp, "depth = %lu", bfTreeNodeGetDepth(treeNode));
  if (bfTreeNodeIsRoot(treeNode))
    fprintf(fp, ", root");
  else
    fprintf(fp, ", index = %lu", treeNode->index);
  fprintf(fp, ": [%g, %g)", intervalTreeNode->a, intervalTreeNode->b);
  fprintf(fp, " -> %lu freqs\n", bfTreeNodeGetNumPoints(treeNode));
}

int main(int argc, char *argv[]) {
  Opts opts;
  if (!parseArgs(&opts, argc, argv))
    return EXIT_FAILURE;

  bfSeed(opts.seed);
  bfSetLogLevel(opts.logLevel);

  BF_ERROR_BEGIN() {}

  BfTrimesh *trimesh = NULL;
  BfPoints3 *verts = NULL;
  BfSize numVerts = 0;
  BfTree *rowTree = NULL;
  BfMat *L = NULL, *M = NULL;
  if (strcmp(opts.objPath, "")) {
    trimesh = bfTrimeshNewFromObjFile(opts.objPath);
    HANDLE_ERROR();

    /* Compute a finite element discretization of the Laplace-Beltrami
    * operator on `trimesh` using linear finite elements. The stiffness
    * matrix is returned in L and the mass matrix is returned in M. The
    * mass matrix isn't diagonal but this isn't too important. */
    bfTrimeshGetLboFemDiscretization(trimesh, &L, &M);
    HANDLE_ERROR();

    verts = bfTrimeshGetVerts(trimesh);

    numVerts = bfTrimeshGetNumVerts(trimesh);
    printf("using linear FEM on triangle mesh from obj file %s (%lu verts)\n", opts.objPath, numVerts);
  } else if (strcmp(opts.matDirPath, "")) {
    /* Load mass matrix M and stiffness matrix L from binaries */
    char rowptrPath[200];
    char colindPath[200];
    char dataPath[200];
    strcpy(rowptrPath, opts.matDirPath);
    strcat(rowptrPath, "/L_rowptr.bin");
    strcpy(colindPath, opts.matDirPath);
    strcat(colindPath, "/L_colind.bin");
    strcpy(dataPath, opts.matDirPath);
    strcat(dataPath, "/L_data.bin");
    L = bfMatCsrRealToMat(bfMatCsrRealNewFromBinaryFiles(rowptrPath, colindPath, dataPath));
    if (bfMatGetNumRows(L) != bfMatGetNumCols(L)) {
      printf("L must have same number of rows and columns (got %lu x %lu matrix)\n",
             bfMatGetNumRows(L), bfMatGetNumCols(L));
      exit(EXIT_FAILURE);
    }

    rowptrPath[0] = '\0';
    colindPath[0] = '\0';
    dataPath[0] = '\0';
    strcpy(rowptrPath, opts.matDirPath);
    strcat(rowptrPath, "/M_rowptr.bin");
    strcpy(colindPath, opts.matDirPath);
    strcat(colindPath, "/M_colind.bin");
    strcpy(dataPath, opts.matDirPath);
    strcat(dataPath, "/M_data.bin");
    M = bfMatCsrRealToMat(bfMatCsrRealNewFromBinaryFiles(rowptrPath, colindPath, dataPath));
    if (bfMatGetNumRows(M) != bfMatGetNumCols(M)) {
      printf("M must have same number of rows and columns\n");
      exit(EXIT_FAILURE);
    }

    char vertsPath[200];
    strcpy(vertsPath, opts.matDirPath);
    strcat(vertsPath, "/nodes.bin");
    verts = bfPoints3NewFromBinaryFile(vertsPath);

    numVerts = bfPoints3GetSize(verts);
    printf("using FEM matrices loaded from binaries in directory %s (%lu verts)\n", opts.matDirPath, numVerts);
  }

  /* Find the largest eigenvalue. We need this to determine the
   * interval on which we'll build the frequency tree. */
  bfToc();
  BfReal lamMax = bfGetMaxEigenvalue(L, M);
  HANDLE_ERROR();
  printf("maximum eigenvalue: %g [%.1fs]\n", lamMax, bfToc());
  printf("maximum frequency: %g [%.1fs]\n", sqrt(lamMax), bfToc());
  if (opts.computeMaxFreq) {
    exit(EXIT_SUCCESS);
  }

  /* The natural frequency of each eigenvector is the square root of
   * the associated eigenvalue. */
  if (isfinite(opts.freqMax)) {
    printf("- passed user-defined maximum frequency of %1.2f\n", opts.freqMax);
    printf("- would have used %1.2f\n", sqrt(lamMax));
  } else {
    printf("- using maximum frequency of %1.2f = sqrt(%1.2f)\n", sqrt(lamMax), lamMax);
  }
  BfReal freqMax = isfinite(opts.freqMax) ? opts.freqMax : sqrt(lamMax);

  if (opts.useOctree) {
    BfOctree *octree = bfOctreeNew();
    HANDLE_ERROR();

    bfOctreeInit(octree, verts, NULL, /* maxLeafSize: */ 64);
    HANDLE_ERROR();

    char const *octreeBoxesPath = "octree_boxes.txt";
    bfOctreeSaveBoxesToTextFile(octree, octreeBoxesPath);
    HANDLE_ERROR();
    printf("wrote octree cells to %s\n", octreeBoxesPath);

    rowTree = bfOctreeToTree(octree);
  } else if (opts.useFiedlerTree) {
    BfFiedlerTree *fiedlerTree = bfFiedlerTreeNewFromTrimesh(
      trimesh, /* tol: */ 1e-15, /* keepNodeTrimeshes: */ false);
    printf("built Fiedler tree\n");

    rowTree = bfFiedlerTreeToTree(fiedlerTree);
  }

  BfSize rowTreeMaxDepth = bfTreeGetMaxDepth(rowTree);
  printf("row tree has depth %lu\n", rowTreeMaxDepth);

  /* Set the valence of the frequency tree to match the valence of the
   * row tree: */
  BfSize k = opts.useOctree ? 8 : 2;

  /* Figure out the depth of the frequency tree: */
  BfSize freqTreeMaxDepth = BF_SIZE_BAD_VALUE;
  if (opts.freqTreeMaxDepth != BF_SIZE_BAD_VALUE) {
    freqTreeMaxDepth = opts.freqTreeMaxDepth;
    if (freqTreeMaxDepth > rowTreeMaxDepth) {
      printf("user-defined frequency tree depth (%lu) exceeds row tree depth\n", freqTreeMaxDepth);
      exit(EXIT_FAILURE);
    }
  } else {
    freqTreeMaxDepth = rowTreeMaxDepth;
  }
  BF_ASSERT(freqTreeMaxDepth <= rowTreeMaxDepth);

  printf("building frequency tree with depth %lu (k = %lu)\n", freqTreeMaxDepth, k);

  /* Set up the frequency tree. Note: we build the tree on the
   * frequency scale as opposed to the eigenvalue scale to preserve
   * the time-frequency product in the butterfly factorization. */
  BfIntervalTree *freqTree = bfIntervalTreeNew();
  bfIntervalTreeInitEmpty(freqTree, 0, freqMax, k, freqTreeMaxDepth);
  HANDLE_ERROR();

  /* Upcast frequency tree to get the column tree */
  BfTree *colTree = bfIntervalTreeToTree(freqTree);

  FILE *fp = fopen("freqTree.txt", "w");
  bfTreeMapConst(colTree, NULL, BF_TREE_TRAVERSAL_LR_LEVEL_ORDER, (BfTreeMapConstFunc)printNode, fp);
  fclose(fp);

  BfPoints1 *freqs = bfPoints1New();
  HANDLE_ERROR();

  bfPoints1InitEmpty(freqs, BF_ARRAY_DEFAULT_CAPACITY);
  HANDLE_ERROR();

  BfFacSpec spec = {
    .rowTree = rowTree,
    .colTree = colTree,
    .rowTreeInitDepth = opts.rowTreeInitDepth,
    .tol = opts.tol,
    .minNumRows = 20,
    .minNumCols = 20,
    .compareRelativeErrors = opts.compareRelativeErrors,
    .bailIfBottomedOut = opts.bailIfBottomedOut,
  };

  /* Set up the depth-first butterfly factorization streamer. We'll
   * use this below to construct the butterfly factorization
   * incrementally. */
  BfFacStreamer *facStreamer = bfFacStreamerNew();
  bfFacStreamerInit(facStreamer, &spec);
  HANDLE_ERROR();

  if (opts.numLeafNodes != BF_SIZE_BAD_VALUE) {
    printf("streaming %lu leaf nodes\n", opts.numLeafNodes);
  } else if (opts.bailIfBottomedOut) {
    printf("streaming until we bottom out in the space tree\n");
  } else if (isfinite(opts.freqMax)) {
    printf("streaming until we cover [0, %1.2f]\n", opts.freqMax);
  } else {
    printf("streaming the entire eigenvector matrix (PROBABLY A BAD IDEA!)\n");
  }

  BfReal t_total = 0;
  BfReal t_eigs = 0;

  BfFacSpan *facSpan = NULL;
  BfMat *mat = NULL;

  BfSize numStreamed = 0;
  while (!bfFacStreamerIsDone(facStreamer)) {
    BfReal t0_column = bfTime();

    BfLboFeedResult result = bfLboFeedFacStreamerNextEigenband(facStreamer, freqs, L, M);
    printf("streamed frequency band %c%1.2f, %1.2f%c:\n",
           result.freqBand.closed[0] ? '[' : '(',
           result.freqBand.endpoint[0],
           result.freqBand.endpoint[1],
           result.freqBand.closed[1] ? ']' : ')');

    BfSize numBytesUncompressed = sizeof(BfReal)*numVerts*freqs->size;
    printf("- streamed %lu eigs total (%1.1f%% of total) in %1.2fs\n",
           freqs->size,
           (100.0*freqs->size)/numVerts,
           result.eigenbandTime);
    printf("- uncompressed size: %.3f MB\n", numBytesUncompressed/pow(1024, 2));

    BfReal t1_column = bfTime();
    printf("- total time: %1.2fs\n", t1_column - t0_column);

    t_total += t1_column - t0_column;
    t_eigs += result.eigenbandTime;

    if (!result.success) {
      printf("* bottomed out!\n");
      break;
    }

    facSpan = bfFacStreamerGetFacSpan(facStreamer);
    mat = bfFacSpanGetMat(facSpan, BF_POLICY_VIEW);
    BfSize numBytesCompressed = bfMatNumBytes(mat);
    bfMatDelete(&mat);
    bfFacSpanDelete(&facSpan);

    printf("- compressed size:   %.3f MB\n", numBytesCompressed/pow(1024, 2));
    printf("- compression rate:  %.3f\n", (double)numBytesUncompressed/numBytesCompressed);

    HANDLE_ERROR();

    if (++numStreamed >= opts.numLeafNodes) break;
  }

  printf("finished streaming butterfly factorization\n");
  printf("- total time: %1.2fs\n", t_total);
  printf("- eigs time: %1.2fs (%0.1f%%)\n", t_eigs, 100.0*t_eigs/t_total);

  BF_ERROR_END() {}

  /** Clean up: */

  bfMatDelete(&mat);
  bfFacSpanDelete(&facSpan);
  bfFacStreamerDelete(&facStreamer);
  bfPoints1Delete(&freqs);
  bfTreeDelete(&colTree);
  bfMatDelete(&M);
  bfMatDelete(&L);
  bfTreeDelete(&rowTree);
  bfTrimeshDeinitAndDealloc(&trimesh);
}
