import numpy as np
import os

directory = 'data/DTU/scan24/scan/cam_lin'
for dir in os.listdir(directory):
    data = np.load(os.path.join(directory,dir))
    print(dir)
    print(data)
    print("    ")

directory = 'data/DTU/scan24/scan/camera'
for dir in os.listdir(directory):
    data = np.load(os.path.join(directory,dir))
    print(dir)
    print(data)
    print("    ")


# camera = 'data/DTU/scan24/scan/camera_mat_0.npy'
# scale_mat = 'data/DTU/scan24/scan/scale_mat_0.npy'
#
# cam = np.load(camera)
# sca = np.load(scale_mat)
# print(cam)
#
# abc = np.matmul(sca,cam)
# print(abc)