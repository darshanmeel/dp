# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 22:19:20 2015

@author: Inpiron
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

si = pd.read_csv('si.csv',index_col=0)
print si.shape

n_hidden = 10
imgsize = 100
wght= []
in_to_hid = [np.random.normal() for i in range(n_hidden) for j in range(imgsize)]
in_to_hid = np.array(in_to_hid).reshape(n_hidden,imgsize)
wght.append(in_to_hid)

in_to_hid = [np.random.normal() for i in range(n_hidden) for j in range(imgsize)]
in_to_hid = np.array(in_to_hid).reshape(imgsize,n_hidden)
print in_to_hid.shape
wght.append(in_to_hid)

wght = np.array(wght)
print wght.shape

