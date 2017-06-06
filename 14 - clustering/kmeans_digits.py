"""
@author: Hannes Rothe
@Original: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
@Description: K-Means algorithm for digits data to learn handwriting numbers; reduced to two dimensions using PCA. Representative images of digits per clusters is added to original code.
"""

import numpy as np
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import pairwise_distances_argmin_min

"""
Load and Prepare Data
"""

digits = load_digits()
n_digits = len(np.unique(digits.target)) #number of different digit values (here 10)
reduced_data = PCA(n_components=2).fit_transform(scale(digits.data)) #run pca for data preparation

"""
Calculate KMeans with n_digits clusters
"""
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10) #initiate kmeans with 10 clusters
kmeans.fit(reduced_data) #run kmeans

          
"""
Visualization
"""
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

# Get instance closest to the centroid

centroids = kmeans.cluster_centers_
closest, _ = pairwise_distances_argmin_min(centroids, reduced_data)

# Plot the centroids as a white X
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker=n_digits , s=169, linewidths=3,
            color='w', zorder=10)

plt.title('K-means clustering on the digits dataset (PCA-reduced data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

print('Show images of data points closest to centroids')
for cl in closest:
    plt.matshow(digits.images[cl])
    print("True value:"+str(digits.target[cl]))
    plt.show()
