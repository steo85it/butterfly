#!/usr/bin/env python

import sys
sys.path.insert(-1, '..') # for util
sys.path.insert(-1, '../../build/wrappers/python') # for butterfly

import colorcet as cc
import numpy as np
import pyvistaqt as pvqt
import pyvista as pv
import time

from pathlib import Path

import butterfly as bf

if __name__ == '__main__':
    path = Path('67p.obj')

    t0 = time.time()

    trimesh = bf.Trimesh.from_obj(str(path))

    if trimesh.vertex_normals is None:
        print(f'obj file {path} lacks vertex normals')
        exit(1)

    if trimesh.vertex_normals is not None and trimesh.face_normals is None:
        trimesh.compute_face_normals_matching_vertex_normals()

    trimesh.init_embree()

    print(f'loaded mesh with {trimesh.num_verts} vertices and {trimesh.num_faces} faces [{time.time() - t0:0.2f}s]')

    t0 = time.time()
    F = bf.MatCsrReal.new_view_factor_matrix_from_trimesh(trimesh)
    print(f'computed view factor matrix [{time.time() - t0:0.2f}s]')

    i = 0
    e_i = np.zeros(trimesh.num_faces)
    e_i[i] = 1

    F_i = F@e_i

    grid = pv.make_tri_mesh(trimesh.verts, trimesh.faces)
    grid['F_i'] = F_i.to_array()

    plotter = pvqt.BackgroundPlotter()
    plotter.add_mesh(grid, scalars='F_i', cmap=cc.cm.gray)
