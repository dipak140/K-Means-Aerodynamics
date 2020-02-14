# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:39:52 2019

@author: DIPAK
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread   ## ???
import pandas as pd
import numpy as np
import seaborn as sns         ## ???
#from sklearn.datasets.sample_generator import (make_blobs, make_circles, make_moons)         ## ???
from sklearn.cluster import KMeans, SpectralClustering      ## ???
from sklearn.preprocessing import StandardScaler             
from sklearn.metrics import silhouette_samples, silhouette_score   ## ???

dataRaw = pd.read_csv(r"C:\Users\DIPAK\COMP167\K-Means-Aerodynamics\eruptionData.csv")

plt.figure(figsize=(6, 6))
plt.scatter(dataRaw.iloc[:, 0], dataRaw.iloc[:, 1])
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of raw data')

X_std = StandardScaler().fit_transform(dataRaw)

km = KMeans(n_clusters = 2, max_iter = 100 )
km.fit(X_std)
centroids = km.cluster_centers_



fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(X_std[km.labels_ == 0, 0], X_std[km.labels_ == 0, 1],
            c='green', label='cluster 1')
plt.scatter(X_std[km.labels_ == 1, 0], X_std[km.labels_ == 1, 1],
            c='blue', label='cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,
            c='r', label='centroid')

plt.legend()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of clustered data', fontweight='bold')
ax.set_aspect('equal')



