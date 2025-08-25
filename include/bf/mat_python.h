#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize NumPy C-API once. Returns 0 on success, -1 on failure. */
int bfInit(void);

#ifdef __cplusplus
}
#endif

/* Avoid including Python.h here — forward declare PyObject instead */
typedef struct _object PyObject;

#include "mat.h"

/** Interface: Mat */

BfMat *bfMatPythonGetView(BfMatPython *matPython);
BfType bfMatPythonGetType(BfMatPython const *matPython);
BfSize bfMatPythonGetNumRows(BfMatPython const *matPython);
BfSize bfMatPythonGetNumCols(BfMatPython const *matPython);
BfMat *bfMatPythonMul(BfMatPython const *matPython, BfMat const *otherMat);
BfMat *bfMatPythonRmul(BfMatPython const *matPython, BfMat const *otherMat);

/** Upcasting: MatPython -> Mat */

BfMat *bfMatPythonToMat(BfMatPython *matPython);
BfMat const *bfMatPythonConstToMatConst(BfMatPython const *matPython);

/** Downcasting: Mat -> MatPython */

BfMatPython *bfMatToMatPython(BfMat *mat);

/** Implementation: MatPython */

struct BfMatPython {
  BfMat super;

  /* A pointer to the Python extension class backing this instance. */
  PyObject *obj;
};

BfMatPython *bfMatPythonAlloc(void);
BfMatPython *bfMatPythonNewFromPyObject(PyObject *obj, BfSize numRows, BfSize numCols);
void bfMatPythonInitFromPyObject(BfMatPython *matPython, PyObject *obj, BfSize numRows, BfSize numCols);
