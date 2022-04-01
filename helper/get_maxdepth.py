import imageio
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
listmax = []
listmin= []
for obj in ["scan65", "scan106", "scan118"]:
	print(obj)
	files = glob.glob(os.path.join("./",obj, obj,'depth','*.exr'))
	for filei in files:
		print(filei)
		depth = imageio.imread(filei)
		max_i = depth[depth!=np.inf].max()
		listmax.append(max_i)
		min_i = depth[depth!=np.inf].min()
		listmin.append(min_i)
	print(max(listmax))

lm=np.array(listmax)
lmi=np.array(listmin)  

n, bins, patches = plt.hist(listmax, 50, density=True, facecolor='g', alpha=0.75)
plt.show()
import ipdb; ipdb.set_trace()
print(listmax)
print(listmin)
print(max(listmax))
print(min(listmin))

