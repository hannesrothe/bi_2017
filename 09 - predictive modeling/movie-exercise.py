# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:16:18 2017

@author: hrothe
"""

from sklearn import tree
from sklearn import preprocessing
import pandas as pd

movies = pd.read_csv('../00 - data/imdbData.csv', header=0)
movies = movies.drop_duplicates().dropna()
movies['profit'] = movies['gross'] - movies['budget']

movies.loc[movies['imdb_score'] >= 8, 'score_class'] = 'high'
movies.loc[movies['imdb_score'] < 8, 'score_class'] = 'medium'
movies.loc[movies['imdb_score'] < 4, 'score_class'] = 'low'

lab_enc = preprocessing.LabelEncoder()
movies_enc = lab_enc.fit_transform(movies)
          
clf = tree.DecisionTreeClassifier()
cl_fit = clf.fit(movies_enc[['profit', 'director_facebook_likes']], movies_enc['imdb_score'])
