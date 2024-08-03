#!/usr/bin/env python

import numpy as np
import sys

from pathlib import Path

data_path = Path(sys.argv[1])
rowptr_path = Path(sys.argv[2])
colind_path = Path(sys.argv[3])

data = np.fromfile(data_path, dtype=np.float64)
rowptr = np.fromfile(rowptr_path, dtype=np.intc)
colind = np.fromfile(colind_path, dtype=np.intc)

for k0, k1 in zip(rowptr[:-1], rowptr[1:]):
    P = np.argsort(colind[k0:k1])
    data[k0:k1] = data[k0:k1][P]
    colind[k0:k1] = colind[k0:k1][P]

data.astype(np.float64).tofile(data_path)
rowptr.astype(np.uintp).tofile(rowptr_path)
colind.astype(np.uintp).tofile(colind_path)
