# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def foreground_noise(x,sigma):
    # print(x.shape)
    m,n,_= x.shape
    x = x.reshape(m,n,1)
    y = np.random.rand(m,n,1)*sigma
    y = y+ x
            
    return y


def background_noise(x,sigma):
    m,n,_= x.shape
    y = np.random.rand(m,n,1)*sigma
    for i in range(m):
        for j in range(n):
            if x[i,j]>0.:
                y[i,j,0] = 1.
            
    return y

def rescale(a,b):
    mina = np.min(a[:,:,:,0])
    maxa = np.max(a[:,:,:,0])
    
    minb = np.min(b[:,:,:,0])
    maxb = np.max(b[:,:,:,0])
    b_ =(b-minb)/(maxb-minb)*(maxa-mina)-np.abs(mina)
        
    return b_
