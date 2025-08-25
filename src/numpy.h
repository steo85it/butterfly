#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
/* Each translation unit sees the same unique symbol for the NumPy C-API */
#define PY_ARRAY_UNIQUE_SYMBOL bf_ARRAY_API

/* src/numpy.h — ensure macOS exposes consistent signal macros */
#if defined(__APPLE__)
  /* Feature macros (you already added these earlier) */
  #ifndef _DARWIN_C_SOURCE
  #define _DARWIN_C_SOURCE 1
  #endif
  #ifndef __DARWIN_UNIX03
  #define __DARWIN_UNIX03 1
  #endif

  /* Ensure the right signal macros/types are visible before NumPy pulls <signal.h> */
  #include <sys/param.h>
  #include <sys/signal.h>        /* sometimes defines _NSIG and/or NSIG */
  #include <signal.h>            /* declares sys_signame/sys_siglist using NSIG */
  /* Bridge variants → NSIG */
  #if !defined(NSIG) && defined(_NSIG)
  #  define NSIG _NSIG
  #endif
  /* Some SDKs may still not provide either; define a conservative fallback */
  #if !defined(NSIG)
  #  define NSIG 32
  #endif

#endif

#define PY_SSIZE_T_CLEAN 1
#include <Python.h>

/* Now it’s safe to include NumPy headers (import macros become visible) */
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>

/* NOTE: in all translation units except bf.c, the correct way to
 * include this file is:
 *
 *   #define NO_IMPORT_ARRAY
 *   #include "numpy.h"
 *
 * see the link to the NumPy docs in the comment below for more
 * explanation about NumPy's special needs. */

/* NumPy has some wacky requirements to be initialized correctly. See
 * this link:
 *
 *   https://numpy.org/devdocs/reference/c-api/array.html#importing-the-api
 *
 * for an explanation... */
//#define PY_ARRAY_UNIQUE_SYMBOL bf_ARRAY_API
//#include <numpy/arrayobject.h>

#ifdef BF_DOUBLE
static int const BF_COMPLEX_TYPENUM = NPY_COMPLEX128;
static int const BF_REAL_TYPENUM = NPY_FLOAT64;
#else
#  error "Not implemented yet!"
#endif
