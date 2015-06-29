# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:18:39 2015

@author: dsing001
"""

from NN_1 import NN
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

def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 5,learning_rate=1,epochs = 10,fnc='sigmoid',wgt_decay=0.0,pp=None): 
    
    n_in_layer = train_data.shape[1]
    n_out_layer = n_in_layer
    n_hidden_layer = 4
    
    lyrs = [n_in_layer,n_hidden_layer,n_out_layer]
    n = NN(lyrs,fnc=fnc,learning_rate=learning_rate,epochs=epochs,Normalize=True,batch_size = 1,outer_fnc='linear',wgt_decay=wgt_decay,bias=1.0) 

    train_err,test_err,wghts_after_each_epoch,biases_after_each_epoch = n.fit(train_data,train_cls,test_data,test_cls)
 

    hdwghts= np.array(wghts_after_each_epoch[-1])
    print '1'
    print hdwghts
    wts = hdwghts[-2]
    bswghts = np.array(biases_after_each_epoch[-1])
    print bswghts
    bss = bswghts[-2]
    pl= train_data[0,:]
    print '2'
    print pl
    imgsize = 7
    pl = np.reshape(pl,(imgsize,imgsize))
    plt.imshow(pl)
    plt.show()
    pl = np.reshape(pl,(imgsize*imgsize,1))
    print wts.shape
    print bss.shape
    pl_changed = np.dot(wts,pl) + bss
    print 3
    print pl_changed
    print pl_changed.shape
    
    hsize = math.sqrt(n_hidden_layer)
    print hsize
    pl = np.reshape(pl_changed,(hsize,hsize))
    print pl
    plt.imshow(pl)
    plt.show()
    plt.imshow(pl,cmap=cm.gray)
    plt.show()



ts_f = pd.read_csv('si.csv',index_col=0)

#rdm = numpy.random.normal(0,0.05,ln)
#ts = numpy.array(ts)+ rdm
ts_f = np.array(ts_f)
ts_f = ts_f.T


ts= datasets.load_digits()
dt = ts.data
lbl = ts.target
ts_f = dt[0:10,0:49]

print ts_f
train_class = ts_f
test_class = ts_f
train_data = ts_f
test_data = ts_f
smthing=False
# I havekept the number of epochs to be 100 and I have not used any early stopping.
print (datetime.datetime.now())
pp = PdfPages('Output_figures.pdf')
# you can change the number of hidden layers for example range(6,9,2) means that try with 6 hidden layers and then increase it by 2 untill you cross 9 this means it will run for hidden layers 6 and 8
for nl in range(3,4,2):
    n_hidden_layer= nl*nl
    # eta ia learning rate and if you see that it is range between 0 and 100 and this is because range function takes integers only. I have divided this by 100 to make it proper learning rates as learning
    # rate should be between 0 and 1.
    for eta in range(75,76,15):
        learning_rate= eta/100.0
        if (smthing):
            smt = range(0,5,3)
        else:
            smt = range(0,1,1)
        print smt
        # smoothing rate 0 means no regularization otherwise it means that apply regularization. Here as well values are between 0 and 5 but these are divided by 10000.
        for smts in smt:      
            train_neural_net(train_data,train_class,test_data,test_class,n_hidden_layer=n_hidden_layer,learning_rate=learning_rate,epochs =1,wgt_decay=smts/10000.0,pp=pp)
pp.close()      
print (datetime.datetime.now())