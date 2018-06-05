# -*- coding: utf-8 -*-
"""
@author: hrothe
"""

from sklearn import tree
import pandas as pd
import seaborn as sns

#iris = pd.read_csv('../00 - data/irisData.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
iris = sns.load_dataset("iris")

iris_b = iris.sample(100) #random build sample n=100
iris_t = iris.loc[~iris.index.isin(iris_b.index)] #test sample n=50

clf = tree.DecisionTreeClassifier()
cl_fit = clf.fit(iris_b[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], iris_b['species'])

print("Model Accuracy:")
print(clf.score(iris_t[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], iris_t['species']))
    
"""
Export alternative 1:
    Export as json and visualize using D3JS (https://d3js.org/)
    Attention: load json with irisData_decisiontree_vis.html
    
    Source: http://planspace.org/20151129-see_sklearn_trees_with_d3/
"""
#import json
#
#def rules(clf, features, labels, node_index=0):
#    
#    node = {}
#    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
#        count_labels = zip(clf.tree_.value[node_index, 0], labels)
#        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
#                                  for count, label in count_labels))
#    else:
#        feature = features[clf.tree_.feature[node_index]]
#        threshold = clf.tree_.threshold[node_index]
#        node['name'] = '{} > {}'.format(feature, threshold)
#        left_index = clf.tree_.children_left[node_index]
#        right_index = clf.tree_.children_right[node_index]
#        node['children'] = [rules(clf, features, labels, right_index),
#                            rules(clf, features, labels, left_index)]
#    return node
#
#r = rules(clf, iris.columns.values.tolist(), iris['species'])
#with open('structure.json', 'w') as f:
#    f.write(json.dumps(r))
##

"""
Export alternative 2:
    Export as as dot file and visualize with graphviz

A library called pydot and Graphviz (http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html) are needed
graphviz: You'll find it on http://www.graphviz.org/Download..php (install graphviz before pydot!) or install it with
"conda install graphviz"
pydot:
install it in the console with "conda install -c conda-forge pydotplus" 
(maybe admin rights are needed: in Windows right click on "Eingabeaufforderung"/"cmd" -> start as admin
"""

import pydotplus 

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)


dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.columns.values.tolist(),
                                class_names=iris['species'],
                                filled=True, rounded=True,
                                special_characters=True)  

graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png("decTree.png")
graph.write_pdf("iris.pdf")