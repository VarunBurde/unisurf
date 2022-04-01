import trimesh
import numpy as np
import os
import matplotlib.pyplot as plt
path_cam = "scan65/scan65/cameras.npz"

cam = np.load(path_cam)
scale_mat = cam['scale_mat_inv_1']

mesh = trimesh.load('~/Projects/colmap_dtu_out/scan65/dense/meshed-poisson.ply')
mesh.apply_transform(scale_mat)
vert = mesh.vertices
radius = np.sqrt(np.sum(vert**2, axis=1))
print(vert.max())
print(vert.min())
n, bins, patches = plt.hist(radius, 50, density=True, facecolor='g', alpha=0.75)
plt.show()
import ipdb; ipdb.set_trace()
mesh.export("./scan65_unit.ply")

