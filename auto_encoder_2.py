# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:18:39 2015

@author: dsing001
"""
from numpy import linalg
from NN_3 import NN
import numpy as np
import numpy 
import math
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn import datasets
from matplotlib.backends.backend_pdf import PdfPages


def sigmoid(vl):
    return (1/(1+np.exp(-vl)))

def tanh(vl):
    e2z = np.exp(2*vl)
    return ((e2z-1)/(e2z + 1))
    
def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 5,learning_rate=1,epochs = 10,fnc='sigmoid',wgt_decay=0.0,pp=None): 
    
    n_in_layer = train_data.shape[1]
    n_out_layer = n_in_layer

    
    lyrs = [n_in_layer,n_hidden_layer,n_out_layer]
    print
    print lyrs
    print
    n = NN(lyrs,fnc=fnc,learning_rate=learning_rate,epochs=epochs,Normalize=False,batch_size = 1000,outer_fnc='tanh',wgt_decay=wgt_decay,bias=1.0) 
    print 
    print
    print n.wghts[0]
    print
    print
    train_err,test_err,wghts_after_each_epoch = n.fit(train_data,train_cls,None,test_cls)
 

    hdwghts= np.array(wghts_after_each_epoch[-1])
    print '1'
    print hdwghts
    wts = hdwghts[-2]
    pl= train_data[0,:]
    print '2'
    print pl
    imgsize = math.sqrt(n_in_layer)
    pl = np.reshape(pl,(imgsize,imgsize))
    plt.imshow(pl)
    plt.show()
    pl = np.reshape(pl,(imgsize*imgsize,1))
    print wts
    bias = np.ones(pl.shape[1]).reshape(pl.shape[1],1)
    print pl
    print bias
    pl_changed = np.vstack((pl,bias))
    print pl_changed
    pl_changed = np.dot(wts,pl_changed) 
    print 3
    print pl_changed
    print pl_changed.shape
    fnc = np.vectorize(sigmoid)
    fnc = np.vectorize(tanh)
    pl_changed = fnc(pl_changed)
    print pl_changed
    hsize = math.sqrt(n_hidden_layer)
    print hsize
    pl = np.reshape(pl_changed,(hsize,hsize))
    print pl
    plt.imshow(pl)
    plt.show()
    plt.imshow(pl,cmap=cm.gray)
    plt.show()
    print train_err






ts= datasets.load_digits()
dt = ts.data
lbl = ts.target


print dt.shape

mn = np.mean(dt,axis=0).reshape(1,dt.shape[1])
print dt.shape
print mn.shape

# now subtract the mean
#dt = dt - mn

sigma = np.dot(dt,dt.T)/dt.shape[1]
#sigma = np.cov(dt)


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

print zcawhite
print zcawhite.shape
ts_f = zcawhite
train_class = ts_f
test_class = ts_f
train_data = ts_f
test_data = ts_f
smthing=False
# I havekept the number of epochs to be 100 and I have not used any early stopping.
print (datetime.datetime.now())
pp = PdfPages('Output_figures.pdf')
# you can change the number of hidden layers for example range(6,9,2) means that try with 6 hidden layers and then increase it by 2 untill you cross 9 this means it will run for hidden layers 6 and 8
for nl in range(6,5,-1):
    n_hidden_layer= nl*nl
    # eta ia learning rate and if you see that it is range between 0 and 100 and this is because range function takes integers only. I have divided this by 100 to make it proper learning rates as learning
    # rate should be between 0 and 1.
    for eta in range(1,2,15):
        learning_rate= eta/100.0
        if (smthing):
            smt = range(0,5,3)
        else:
            smt = range(0,1,1)
        print smt
        # smoothing rate 0 means no regularization otherwise it means that apply regularization. Here as well values are between 0 and 5 but these are divided by 10000.
        for smts in smt:      
            train_neural_net(train_data,train_class,test_data,test_class,n_hidden_layer=n_hidden_layer,learning_rate=learning_rate,epochs =500,wgt_decay=smts/10000.0,pp=pp)
pp.close()      
print (datetime.datetime.now())