#pragma once

typedef enum {
  BF_BACKEND_LAPACK,
  /* Sparse SVD/SVDS backend:
   * - PRIMME_SVDS preferred when available
   * - optional ARPACK_SVDS fallback only if explicitly enabled
   */
  BF_BACKEND_SVDS,

  /* Deprecated alias kept for source compatibility (historical naming). */
  BF_BACKEND_ARPACK = BF_BACKEND_SVDS,
} BfBackend;

