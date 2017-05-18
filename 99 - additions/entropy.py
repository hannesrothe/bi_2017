# -*- coding: utf-8 -*-
"""
@author: hrothe

@Description: Comparison between Entropy and Cross-entropy
"""

import scipy
import pandas as pd
import numpy as np

data = pd.DataFrame()
data['data'] = [0,0,0,0,0,1,1,1,1,1]

#get frequency of classes in the array
freq = scipy.stats.itemfreq(data['data'])[:,1]

#calculate probabilities for classes in the array
prob = freq / np.sum(freq)

#print simple entropy
print("entropy")
print(scipy.stats.entropy(prob,base=2))



# Let's compare this to the distribution of "True Values"
true_data = pd.DataFrame()
true_data['data'] = [1,0,0,0,0,1,1,1,1,1]
#get frequency of classes in the array
freq_true = scipy.stats.itemfreq(true_data['data'])[:,1]

#calculate probabilities for classes in the array
prob_true = freq_true / np.sum(freq_true)

#print Cross entropy
print("Cross-entropy, using Kullback-Leibler distance")
print(scipy.stats.entropy(prob,qk=prob_true,base=2))