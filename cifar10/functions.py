# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#foreground
def noisy_image(x,sigma=.1,negative=0):
    m,n,c = x.shape
    y = np.zeros((m,n,c))
    for i in range(c):
        if negative:
            noise = np.random.rand(m,n)-.5
        else:
            noise = np.random.rand(m,n)
        y[:,:,i] = x[:,:,i]*(1+noise*sigma)
            
    return y

def noisy_Poisson_image(x, sigma=.1):
    m,n,c = x.shape
    y = np.zeros((m,n,c))
    for i in range(c):
        noise = np.random.poisson(sigma,(m,n))
        y[:,:,i] = x[:,:,i]*(1+noise)
            
    return y

def noisy_MBN_image(x,sigma = .1):
    m,n,c = x.shape
    y = np.zeros((m,n,c))
    m = np.random.choice([0, 1], size=(m,n,1), p=[sigma, 1-sigma])
    y = x*np.repeat(m, c, axis=2)
    return y


image_shape = (32, 32,3)
def plot_gallery(title, images, n_col, n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i +1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,            
                   vmin=-vmax, vmax=vmax)
#         plt.imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    plt.show()

def rescale(a,b):
    #b_ = np.zeros((b.shape[0],32,32,3))
    mina = np.array([np.min(a[:,:,:,0]),np.min(a[:,:,:,1]),np.min(a[:,:,:,2])])
    maxa = np.array([np.max(a[:,:,:,0]),np.max(a[:,:,:,1]),np.max(a[:,:,:,2])])
    
    minb = np.array([np.min(b[:,:,:,0]),np.min(b[:,:,:,1]),np.min(b[:,:,:,2])])
    maxb = np.array([np.max(b[:,:,:,0]),np.max(b[:,:,:,1]),np.max(b[:,:,:,2])])
    b_ =(b-minb)/(maxb-minb)*(maxa-mina)-np.abs(mina)
        
    return b_

