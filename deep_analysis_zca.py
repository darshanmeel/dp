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

a = open('pcaData.csv','r')
ln = []
for line in a:
    ln1 = line.strip('\n').split(',')
    ln1 = [float(ln2.strip("\n").strip()) for ln2 in ln1]
    ln.append(ln1)
dt = np.array(ln)
print dt.shape

plt.scatter(dt[0,:],dt[1,:])
plt.show()
#remove mean values from each row
mn = np.mean(dt,axis=0).reshape(1,dt.shape[1])
print dt.shape
print mn.shape

# now subtract the mean
dt = dt - mn

sigma = np.dot(dt,dt.T)/dt.shape[0]
sigma = np.cov(dt)


u,s,v = linalg.svd(sigma)

dt_rot = np.dot(u.T,dt)

plt.scatter(dt_rot[0,:],dt_rot[1,:])
plt.show()

sigma1 = np.cov(dt_rot)

abc =np.array(np.divide(1,np.sqrt(s))).reshape(s.shape[0],1)
pcawhite = np.multiply(abc,np.dot(u.T,dt))


plt.scatter(pcawhite[0,:],pcawhite[1,:])
plt.show()

zcawhite = np.dot(u,pcawhite)

plt.scatter(zcawhite[0,:],zcawhite[1,:])
plt.show()