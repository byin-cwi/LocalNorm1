# -*- coding: utf-8 -*-

### 
# useful functions
### various noise
### abd oplt functions

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cv2

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
            
    return rescale(y)

def noisy_Poisson_image_add(x, sigma=.1):
    img_scale = np.max(x,axis=(0,1,2))
    m,n,c = x.shape
    y = np.zeros((m,n,c))
    for i in range(c):
        noise = np.random.poisson(sigma*255,(m,n))
        y[:,:,i] = x[:,:,i]+noise
            
    return rescale(y,img_scale).astype('int32')

def noisy_MBN_image(x,sigma = .1):
    m,n,c = x.shape
    y = np.zeros((m,n,c))
    m = np.random.choice([0, 1], size=(m,n,1), p=[sigma, 1-sigma])
    y = x*np.repeat(m, c, axis=2)
    return y


image_shape = (224,224,3)
def plot_gallery(title, images, n_col, n_row,isSave=0):
    fig = plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i +1)
        # vmax = max(comp.max(), -comp.min())
        # plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,            
        #            vmin=-vmax, vmax=vmax)
        # plt.imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
        plt.imshow(rescale(comp))
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    if isSave: 
        fig.savefig('noisy_image_example/'+title+'.png')
    else:
        plt.show()
    plt.close()


def rescale(b,s=np.ones(3)):
    #b_ = np.zeros((b.shape[0],32,32,3))
    mina = 0
    maxa = s
    
    minb = np.array([np.min(b[:,:,0]),np.min(b[:,:,1]),np.min(b[:,:,2])])
    maxb = np.array([np.max(b[:,:,0]),np.max(b[:,:,1]),np.max(b[:,:,2])])
    b_ =(b-minb)/(maxb-minb)*(maxa-mina)-np.abs(mina)
        
    return b_

