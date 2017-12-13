from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from sklearn.cluster import KMeans


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


plt.rcParams['figure.figsize'] = (8, 6)
plt.style.use('ggplot')

data = np.loadtxt("data/Y_matrix.txt")

V1 = data[:,0].tolist()
V2 = data[:,1].tolist()

Coords = np.array(list(zip(V1, V2)))
# plt.scatter(V1, V2, c='black', s=7)
# # plt.show()

k = 2
C_lat = [random.uniform(-1.0, 1.0) for i in xrange(0,k)]
C_long = [random.uniform(-1.0, 1.0) for i in xrange(0,k)]
C = np.array(list(zip(C_lat, C_long)), dtype=np.float32)

C_old = np.zeros(C.shape)
clusters = np.zeros(len(Coords))
error = dist(C, C_old, None)

while error != 0:

	for i in range(len(Coords)):
		distances = dist(Coords[i], C)
		cluster = np.argmin(distances)
		clusters[i] = cluster

	C_old = deepcopy(C)

	for i in range(k):
		points = [Coords[j] for j in range(len(Coords)) if clusters[j] == i]
		C[i] = np.mean(points, axis=0)

	error = dist(C, C_old, None)

print(C)


colors = ['r', 'g', 'b', 'y', 'c', 'm', 'g']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([Coords[j] for j in range(len(Coords)) if clusters[j] == i])
    print(points.shape)
    ax.scatter(points[:, 0], points[:, 1], s=100, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
plt.show()

my_list = []
for j in range(len(Coords)):
	my_list.append(clusters[j])

print(len(my_list))
	
import csv
with open("data/spectral_clustering_results.csv", 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(my_list)





