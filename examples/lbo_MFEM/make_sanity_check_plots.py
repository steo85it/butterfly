#!/usr/bin/env python

import colorcet as cc
import numpy as np
import scipy.sparse
import pyvista as pv
import sys

from pathlib import Path

path = Path(sys.argv[1])

L_data = np.fromfile(path/'L_data.bin', dtype=np.float64)
L_rowptr = np.fromfile(path/'L_rowptr.bin', dtype=np.uintp)
L_colind = np.fromfile(path/'L_colind.bin', dtype=np.uintp)

M_data = np.fromfile(path/'M_data.bin', dtype=np.float64)
M_rowptr = np.fromfile(path/'M_rowptr.bin', dtype=np.uintp)
M_colind = np.fromfile(path/'M_colind.bin', dtype=np.uintp)

nodes = np.fromfile(path/'nodes.bin').reshape(-1, 3)

L = scipy.sparse.csr_matrix((L_data, L_colind, L_rowptr))
M = scipy.sparse.csr_matrix((M_data, M_colind, M_rowptr))

assert nodes.shape[0] == L.shape[0] == L.shape[1]
assert nodes.shape[0] == M.shape[0] == M.shape[1]

lam_max = scipy.sparse.linalg.eigsh(L, k=1, M=M)[0][0]
print(f'{lam_max = }')

Lam, U = scipy.sparse.linalg.eigsh(L, k=2, M=M, sigma=-0.001)

poly_data = pv.PolyData(nodes)
poly_data['u1'] = U[:, 1]

_ = pv.Plotter(off_screen=True)
_.add_mesh(poly_data, scalars='u1', cmap=cc.cm.gouldian)
_.screenshot('blah.png')
