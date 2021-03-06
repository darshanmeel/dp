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


def sigmoid(vl):
    return (1/(1+ma.exp(-vl)))

def sigmoid_diff(vl):
    return sigmoid(vl)* (1-sigmoid(vl))
    
def tanh(vl):
    return (1/(1+ma.exp(-vl)))

def tanh_diff(vl):
    return sigmoid(vl)* (1-sigmoid(vl))
    
def linear(x):
    return x
    
def linear_diff(x):
    return 1
    
class NN:
    def __init__(self,lyrs,fnc='sigmoid',learning_rate= 1.0,epochs= 1000,outer_fnc= None,Normalize= True,batch_size = 1,wgt_decay = 0,bias= 1):
        self.lyrs = lyrs
        if len(self.lyrs) < 3:
            raise('There should be more than 2 layers')
        self.bias = bias 
        self.n_in_layer = 2
        self.Normalize=Normalize
        self.n_outer_layer = 1
        self.mn= 0
        self.mx = 1
        self.min=None
        self.max = None
        self.wgt_decay = wgt_decay
        
        self.fnc = [] 
        self.fnc_diff = []
        if fnc == 'tanh':
            for i in range(len(self.lyrs) -2):
                self.fnc.append(ma.vectorize(tanh))
                self.fnc_diff.append(ma.vectorize(tanh_diff))
        else:
            for i in range(len(self.lyrs) -2):
                self.fnc.append(ma.vectorize(sigmoid))
                self.fnc_diff.append(ma.vectorize(sigmoid_diff))

        if outer_fnc=='linear':
            self.fnc.append(ma.vectorize(linear))
            self.fnc_diff.append(ma.vectorize(linear_diff))

        else:
            self.fnc.append(ma.vectorize(sigmoid))
            self.fnc_diff.append(ma.vectorize(sigmoid_diff))
        


        self.eta = learning_rate            
        self.epochs = epochs
        self.training_error= []
        self.batch_size = batch_size
        
        #wghts
        self.wghts = []
        self.biases = []
        self.init_weights()
        self.wghts_delta = self.wghts[:]
        self.bias_delta = self.biases[:]
        print len(self.wghts)
        print len(self.biases)
        
            

        
    def init_weights(self):
        
        for i in range(len(self.lyrs) -1):
            ed = math.sqrt(6/(self.lyrs[i] + self.lyrs[i+1] + 1))
            st = -1*ed
            wght = numpy.random.uniform(st,ed,self.lyrs[i+1]*self.lyrs[i]).reshape(self.lyrs[i+1],self.lyrs[i])
            self.wghts.append(wght)
            bias = numpy.zeros(self.lyrs[i+1]).reshape(self.lyrs[i+1],1)
            self.biases.append(bias)
      

    def nrmlz(self,X):
       
        return numpy.multiply(numpy.divide(numpy.subtract(X,self.min),(self.max-self.min)),(self.mx-self.mn))

    def forward_pass(self,inpt):

        outs = []
        diffs = []
        print inpt.T
     
        outs.append(inpt.T)
        diffs.append(inpt.T)
        print outs[0]
      
        for i in range(len(self.lyrs)-1):
            print i
            print outs[i].shape
            print self.wghts[i].shape
            print self.wghts[i]
            print self.biases[i].shape
            out = self.wghts[i].dot(outs[i]) + self.biases[i]
            out = ma.array(self.fnc[i](out))
            print out.shape
            outs.append(out)
            diff = ma.array(self.fnc_diff[i](out))
            diffs.append(diff)
       
        return (outs,diffs) 
        
        
          
    def backpropagate(self,outs,diffs,inpts,err):
        
        # betas for output layer neuron to calculate the weights 
        print 'o'
        betas = []
        print diffs[-1]
        bt = ma.multiply(diffs[-1],err)
        print err.shape
        print bt.shape
 
        
        print self.wghts_delta[-1]
        betas.append(bt)
        print outs[0]
        
        print 'p'
        print len(self.wghts)
        print self.wghts_delta[-1]
        print 'q'
        print len(outs)
        print 'r'
        print outs
     
        for i in range(len(self.lyrs)-2,-1,-1):
            print i
            bt = betas[-1]
            print betas
            beta_sums =  ma.sum(self.wghts[i].T.dot(bt))
            print bt
            print outs[i]
            print outs[i].shape,bt.shape
            print outs
          
            wght_delta = ma.multiply(self.eta,bt.dot(outs[i].T))
            print wght_delta.shape
            self.wghts_delta[i] = wght_delta
            bias_delta = ma.multiply(self.eta,bt)
            self.bias_delta[i] = bias_delta
            bt = ma.multiply(diffs[i],beta_sums)
            betas.append(bt)
            print betas
        #print ghyu
    



    def update_weights(self):
        #update in to hidden weights 
 
        for i in range(len(self.lyrs) -1):
            self.wghts[i] = self.wghts[i] + self.wghts_delta[i]
            self.biases[i] = self.biases[i] + self.bias_delta[i]

       

    def fit(self,X,Y,test_data=None,test_class=None):          
        trgts = Y  
        print X

     
        '''
        if self.Normalize:
            if self.min==None:
                self.min = X.min(axis=0)
            if self.max==None:
                self.max = X.max(axis=0)
            print self.max,self.min
            
            X = self.nrmlz(X)
        '''
        trgts = X
    
        bs = self.batch_size
        wghts_after_each_epoch = [] 
        biases_after_each_epoch = [] 
    
        tst_error = []
        for epoch in range(self.epochs):  
            #print epoch
            
            
          
            error= ma.zeros(bs,dtype='float64').reshape(bs,1)
            
            #self.learning_eta *= 0.9
            for i in range(int(math.ceil(X.shape[0]*1.0/bs))): 
                print i
                inputs = X[i*bs:(i+1)*bs,:]   
                targets = trgts[i*bs:(i+1)*bs,:]     
                
                print inputs
                print targets
                print 'kui'
                
                outs,diffs = self.forward_pass(inputs) 
      
                print 'j'
                print outs[0]
                print 'k'
                print diffs
                print 'l'
                print self.wghts
                print 'm'
                print outs[-1]
                print '9999'
                
                
                err = ma.subtract(targets.T,outs[-1])
                print err
                self.backpropagate(outs,diffs,inputs,err)      
                self.update_weights()                             
                error = error + ma.sum(err**2,axis=0)  
                
            train_error = error/(X.shape[0])
  
            #print 'train_error',train_error
            self.training_error.append(train_error)
            #print (epoch, str(datetime.datetime.now()) ,'end')
            if test_data.shape > 0:
                 predicted = self.predict(test_data)
                
                 #test_cls= numpy.array(test_class).reshape(test_data.shape[0],self.n_outer_layer)
                 
                 test_error = ma.sum(ma.power(test_class-predicted,2),axis=0)/(test_data.shape[0])
                 #print 'test_error',test_error
                 tst_error.append(test_error)
            wghts_after_each_epoch.append(self.wghts)
            biases_after_each_epoch.append(self.biases)
          
                       
                
                
        train_err = self.training_error
        return(train_err,tst_error,wghts_after_each_epoch,biases_after_each_epoch)

    def predict(self,X):

       '''
       if self.Normalize:
            X = self.nrmlz(X)
       '''
       outs,diffs = self.forward_pass(X)  
   
       
     
       tst_cls_out = outs[-1].T

       return tst_cls_out
       

        
            
if __name__ == "__main__":
    def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 5,learning_rate=1,epochs = 10,fnc='sigmoid',wgt_decay=0.0,pp=None): 
    
        n_in_layer = train_data.shape[1]
        n_out_layer = n_in_layer
        n_hidden_layer = 4
        
        lyrs = [n_in_layer,n_hidden_layer,n_out_layer]
        n = NN(lyrs,fnc=fnc,learning_rate=learning_rate,epochs=epochs,Normalize=False,batch_size = 2,outer_fnc='linear',wgt_decay=wgt_decay,bias=1.0) 
    
        train_err,test_err,wghts_after_each_epoch,biases_after_each_epoch = n.fit(train_data,train_cls,test_data,test_cls)
     

  
    
    train_dt = numpy.array([[1,0],[1,1],[0,1],[0,0]])
    train_cls = [0,1,0,1]
    train_cls = train_dt
    test_dt = train_dt
    test_cls = train_cls
    train_neural_net(train_dt,train_cls,test_dt,test_cls,epochs =2)
        
    
        
        
            
        
            