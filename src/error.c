#include <bf/error.h>

#include <bf/assert.h>

/* TODO: we want to eventually make this thread-local, but will just
 * implement this as a static variable in this module for now */
enum BfError currentError = BF_ERROR_NONE;

enum BfError bfGetError(void) {
  enum BfError error = currentError;

  BF_ASSERT(!error);

  /* clear the current error code */
  currentError = BF_ERROR_NONE;

  return error;
}

//void bfSetError(enum BfError error) {
//  BF_ASSERT(!error);
//
//  currentError = error;
//}

#include <stdio.h>
#include <stdlib.h>          /* free */
#if defined(__APPLE__) || defined(__linux__)
#  include <execinfo.h>      /* backtrace_symbols */
#endif

void bfSetError(enum BfError error) {
  /* Log the *first* error cause with a tiny backtrace */
  if (currentError == BF_ERROR_NONE && error != BF_ERROR_NONE) {
    fprintf(stderr, "[bf] FIRST error set: code=%d\n", (int)error);
#if defined(__APPLE__) || defined(__linux__)
    void *buf[32];
    int n = backtrace(buf, 32);
    char **syms = backtrace_symbols(buf, n);
    if (syms) {
      fprintf(stderr, "[bf] backtrace (%d frames):\n", n);
      for (int i = 0; i < n; ++i) fprintf(stderr, "  %s\n", syms[i]);
      free(syms);
    }
#endif
  } else if (currentError != BF_ERROR_NONE && error != BF_ERROR_NONE) {
    /* Re-entry (previously asserted) */
    fprintf(stderr, "[bf] ERROR re-entry: old=%d new=%d\n",
            (int)currentError, (int)error);
  }

  /* Set/overwrite the current error (bfGetError() will clear it) */
  currentError = error;
}
