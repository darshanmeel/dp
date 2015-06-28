# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:50:31 2015
@author: dsing001
"""
'''
from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn import datasets 
import math
import datetime
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import NearestNeighbors as nn


import random 
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC

iris_data = datasets.load_digits()
print iris_data
dt = iris_data.data
lbls = iris_data.target
'''

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.decomposition.pca import PCA
from numpy import linalg

dgts_data = pd.read_csv("abcd_2.csv",index_col=0)
dgts_data =  dgts_data.head(10)
print dgts_data.shape
dgts_data = np.array(dgts_data)[:,200:209]
print dgts_data.shape
#print dgts_data

dgts_lbl = pd.read_csv("abcd_2_l.csv",index_col=0)
dgts_lbl = dgts_lbl.head(10)
#print dgts_lbl.head()
lbls = np.array(dgts_lbl)


dt = dgts_data.T

#remove mean values from each row
mn = np.mean(dt,axis=0).reshape(1,dt.shape[1])
print dt.shape
print mn.shape

# now subtract the mean
dt = dt - mn

sigma = np.dot(dt,dt.T)/dt.shape[0]

print sigma

u,s,v = linalg.svd(sigma)
dt_rot = np.dot(u.T,dt)

sigma1 = np.cov(dt_rot)
pc = PCA()
pc.fit(dt)
ab = pc.transform(dt)

print ab
print sigma1
print tyu
abc =np.divide(s,np.sqrt(s+0.000001))
pcawhite = np.dot(abc,np.dot(u.T,dt))
print pcawhite