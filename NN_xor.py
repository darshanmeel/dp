# -*- coding: utf-8 -*-
"""
Created on Sun Nov 02 19:04:21 2014

@author: Darshan Singh

This is single hidden layer model only. It can be made generic to have as many as hidden layer and finally sum of signals at each node can be calculated 
using numpy matrix multiplication to make things bit easy.
"""
import numpy as ma
import numpy
#from numpy import ma 
import math
import datetime

from NN_5 import NN

def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 5,learning_rate=1,epochs = 10,fnc='sigmoid',wgt_decay=0.0,pp=None): 

    n_in_layer = train_data.shape[1]
    n_out_layer = train_cls.shape[1]
    n_hidden_layer = 3
    print n_out_layer

    lyrs = [n_in_layer,3,n_out_layer]
    
    learning_rate = 1.0
    wgt_decay=0.1
    n = NN(lyrs,fnc=fnc,learning_rate=learning_rate,epochs=epochs,Normalize=True,batch_size = 1,outer_fnc='sigmoid',wgt_decay=wgt_decay,bias= True) 

    train_err,test_err,wghts_after_each_epoch = n.fit(train_data,train_cls,None,None)
    print
    print 'train_err'
    print train_err
    print
    print 'wghts'
    print wghts_after_each_epoch[-1]
    print
    print 'predict'
    print n.predict(test_data)
    print
    
     

  

train_dt = numpy.array([[1,0],[1,1],[0,1],[0,0]])
train_cls = numpy.array([0,1,0,1]).reshape(4,1)
#train_cls = train_dt
test_dt = train_dt
test_cls = train_cls
train_neural_net(train_dt,train_cls,test_dt,test_cls,epochs =10)
        
    
        
        
            
        
            