# -*- coding: utf-8 -*-
"""
@author: hrothe

@Adapted code from:
    http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py

@Description: Comparison of different SVM Classifiers using raw data and principal components

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

iris = pd.read_csv('../00 - data/irisData.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

""" PCA Analysis from our class on data understanding """

#select only metric data for pca
raw_iris = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
pca_iris = scale(raw_iris)
pca = PCA(n_components=2)
pca_iris = pca.fit_transform(pca_iris)


""" Choose data for PCA """

# either one of the following two dimensions for graphical representation
X = iris[['sepal_length', 'sepal_width']].as_matrix() #two dimensions from raw data
#X = pca_iris # alternatively, we choose two dimensions from pca
y = iris['species'].astype('category').cat.codes

        
""" Fit the data to an SVM instance """

# SVM regularization parameter
C = 1.0

#compare different kernels
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

""" Prepare illustration"""

h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    """Predict the SVM"""
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
