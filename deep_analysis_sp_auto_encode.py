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
import random

import scipy.io
mat = scipy.io.loadmat('IMAGES.mat')

imgs = mat['IMAGES']
print imgs.shape


#size of image 
imgsize = 512
patchsize = 10

rng = imgsize-patchsize - 1

sampleimages= []
for i in range(100):
    # randomly select the image
    
    img_num = random.randint(0,9)    
    rw = random.randint(0,rng)    
    cl = random.randint(0,rng)
    
    patch = imgs[rw:rw+patchsize,cl:cl+patchsize,img_num].reshape(patchsize*patchsize,1)
    sampleimages.append(patch)
sampleimages =  np.array(sampleimages)
sampleimages = sampleimages.reshape(patchsize*patchsize,sampleimages.shape[0])
print sampleimages.shape
si = pd.DataFrame(sampleimages)
si.to_csv('si.csv')

    