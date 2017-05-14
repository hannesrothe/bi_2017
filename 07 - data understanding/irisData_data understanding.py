# -*- coding: utf-8 -*-
"""
@course: Business Intelligence summer term 2017

@author: Hannes Rothe

Desc: Data Preparation

2017-05-09
"""

#pandas is used for creating DataFrames for more elaborate datasets and analysis
import pandas as pd
#matplotlib is used to plot and manipulate illustrations (i.e., in IPython)
import matplotlib.pyplot as plt
#Seaborn is a powerful data visualization library
import seaborn as sns

# Create DataFrame using Pandas and set Column names
iris = pd.read_csv('irisData.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
# alternative: iris2 = sns.load_dataset("iris")
#iris_list = iris.values.tolist()
#iris_dict = iris.to_dict()

# Show descriptive statistics on dimensional distributions
#print(iris.describe())

# Describe relationships amoung variables in scatter plot
# hue: Variable used for color mapping
#sns.pairplot(iris, hue="species", palette="husl")
#plt.show()
#plt.clf()


"""
Principal Component Analysis
2017-05-10
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

#select only metric data
raw_iris = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

#center data to mean
norm_iris = scale(raw_iris)

#create pca with 2-dimensions
pca = PCA(n_components=2)

# pca data
pca_iris = pca.fit_transform(norm_iris)
print("Show PCA results:")
print(norm_iris.shape)
print(pca_iris.shape)


vis_iris = pd.DataFrame(pca_iris, columns=['pc1', 'pc2'])
vis_iris['species'] = iris['species']
g = sns.FacetGrid(vis_iris, hue='species', size=5)
g.map(plt.scatter, 'pc1', 'pc2')
g.set_xlabels('principal component 1')
g.set_ylabels('principal component 2')

plt.show()
plt.clf()

"""
Parallel Coordinates
2017-05-10
"""

#from pandas.tools.plotting import parallel_coordinates
#
#parallel_coordinates(iris, 'species')
#plt.show()
#plt.clf()
#
#
#"""
#Correlation Analysis
#2017-05-10
#"""
#
#print("Show Pearson's correlation:")
#print(iris.corr())
#
#print("Show Spearman's rho correlation:")
#print(iris.corr('spearman'))
#
#print("Show Kendal's tau correlation:")
#print(iris.corr('kendall'))
#
## Heatmap visualization of correlations
#sns.heatmap(iris.corr('kendall'))
#plt.show()