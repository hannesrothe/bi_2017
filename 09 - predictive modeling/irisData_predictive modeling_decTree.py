# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:00:38 2017

@author: hrothe
"""

def rules(clf, features, labels, node_index=0):
    """Source: http://planspace.org/20151129-see_sklearn_trees_with_d3/
    Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} > {}'.format(feature, threshold)
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node



from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

    
"""
Export alternative 1:
    Export as json and visualize using D3JS (https://d3js.org/)
    Attention: load json with irisData_decisiontree_vis.html
"""
import json

r = rules(clf, iris.feature_names, iris.target_names)
with open('structure.json', 'w') as f:
    f.write(json.dumps(r))


"""
Export alternative 2:
    Export as as dot file and visualize with graphviz

A library called pydot and Graphviz are needed
graphviz: You'll find it on http://www.graphviz.org/Download..php (install graphviz before pydot!) or install it with
"conda install graphviz"
pydot:
install it in the console with "conda install -c conda-forge pydotplus" 
(maybe admin rights are needed: in Windows right click on "Eingabeaufforderung"/"cmd" -> start as admin
"""

import pydotplus 

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf")


dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)  

graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png("decTree.png")
