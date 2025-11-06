//#pragma once
//
//#if defined BF_LINUX
//#  include <openblas/cblas.h>
//#  include <openblas/lapacke.h>
//#elif defined BF_DARWIN
//#  include <cblas.h>
//#  include <lapacke.h>
//#endif

// include/bf/blas.h
#pragma once

#if defined(__has_include)
  #if __has_include(<openblas/cblas.h>)
    #include <openblas/cblas.h>
  #elif __has_include(<cblas.h>)
    #include <cblas.h>
  #else
    #error "No cblas.h found. Add -I to your OpenBLAS/Netlib include dir."
  #endif

  #if __has_include(<openblas/lapacke.h>)
    #include <openblas/lapacke.h>
  #elif __has_include(<lapacke.h>)
    #include <lapacke.h>
  #else
    #error "No lapacke.h found. Install Netlib LAPACKE or OpenBLAS with LAPACKE."
  #endif
#else
  /* Older cpp: fall back to generic names; your -I must point at them */
  #include <cblas.h>
  #include <lapacke.h>
#endif


