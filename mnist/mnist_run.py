# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Flatten, Activation, Conv2D, MaxPool2D, AvgPool2D, Dense, Dropout, BatchNormalization, Lambda,Input, MaxPooling2D, Flatten, Activation, Conv2D, AvgPool2D, Dense, Dropout, concatenate, AveragePooling2D
from keras.optimizers import Adam, SGD
from keras.models import Sequential
import keras.backend as K
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import model_from_json, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import numpy as np
from localNorm import LocalNormalization
from switchNorm import SwitchNormalization
from keras.layers.core import Lambda
from keras import backend as K
import pandas as pd
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from functions import foreground_noise,background_noise,rescale
import argparse
import pickle

parser = argparse.ArgumentParser(description='Choose Norm type')

np.random.seed(451)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

parser.add_argument('--Norm_type', type=str, default='local',
                    help='LocalNorm,BatchNorm,SwitchNorm')
parser.add_argument('--groups', type=int, default=10,
                    help='number of groups')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch_size')
FLAGS = parser.parse_args()
norm_type = FLAGS.Norm_type
group_number = FLAGS.groups
bz = FLAGS.batch_size
group_size = int(bz/group_number)


def norm(norm_type):
        if norm_type=='batch':
            return BatchNormalization()
        elif norm_type=='local':
            return LocalNormalization(groupsize=group_size,batch_size=bz)
        elif norm_type=='switch':
            return SwitchNormalization()

act = 'relu'

######################
###   build model  ###
######################
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation=act,
                 input_shape = (28, 28, 1)))
model.add(norm(norm_type))
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation=act))
model.add(norm(norm_type))

model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation=act))
model.add(norm(norm_type))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation=act))
model.add(norm(norm_type))

model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation=act))
model.add(Dropout(0.25))
model.add(Dense(1024, activation=act))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

######################
### training model ###
######################

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

max_epoch = 50
with tf.device('/gpu:0'):
    hist = model.fit(x_train, y_train, batch_size=bz, 
    epochs=max_epoch, validation_split=0.1,verbose=2,callbacks=[annealer])


res = model.evaluate(x_test,y_test, batch_size=bz)
print("Test accuracy:",res)
print(x_test.shape)

a = 0.

for i in range(len(x_test)):
    X_ = np.zeros((bz,28,28,1))
    for j in range(bz):
        X_[j,:,:,:] = x_test[i]
    preds = model.predict(X_,batch_size=bz)
    prob = np.max(preds)
    class_ids = np.argmax(preds,axis=-1)
    class_id1 = np.bincount(class_ids).argmax()
    if class_id1 == np.argmax(y_test[i]):
        a+=1.
print("method1:",a*1./len(x_test))

a = 0
for i in range(len(x_test)):
    X_ = np.zeros((bz,28,28,1))
    #for j in range(group_number):
    X_[:bz-2,:,:] = x_test[:bz-2,:,:]
    X_[bz-1,:,:,:] = x_test[i]
    preds = model.predict(X_,batch_size=bz)
    prob = np.max(preds)
    class_ids = np.argmax(preds,axis=-1)
    class_id1 = class_ids[-1]#np.bincount(class_ids).argmax()
    if class_id1 == np.argmax(y_test[i]):
        a+=1.
print("method2:",a*1./len(x_test))


c = 0
group_size = 10
x_test_bn = np.zeros((bz,28,28,1))
for i in range(x_test.shape[0]):

    x = x_test[:group_size,:,:,:]
    for k in range(bz):
        k1 = k%group_size
        if k1 == 0:    
            x_test_bn[k,:,:,:] =x_test[i,:,:,:]
        else:
            x_test_bn[k,:,:,:] = x[k%group_size,:,:,:]

    preds = model.predict(x_test_bn,batch_size=bz)
    class_ids = np.argmax(preds,axis=-1)
    class_ids = [class_ids[i] for i in range(0,bz,group_size)]
    class_id = np.bincount(class_ids).argmax()
    if class_id == np.argmax(y_test[i]):
        c+=1.
result = c/len(x_test)
print("method3:",result)


print("Test accuracy:",res)
print(x_test.shape)

def test_FG_noisy(n_list,inputs,isNormalized=1):
#   """
#   Multiplicative Bernoulli noise (aka binomial noise) constructs
#   a random mask m that is 1 for valid pixels and 0 for
#   zeroed/missing pixels. 
#   """
    result_ = []
    for n in n_list:
        x_test_bn = np.zeros((inputs.shape[0],28,28,1))
        for i in range(inputs.shape[0]):
            x_test_bn[i,:,:,:] = foreground_noise(inputs[i,:,:,:],n)
        if isNormalized:
            x_test_bnN = rescale(x_train,x_test_bn)
        else:
            x_test_bnN = x_test_bn
        result =model.evaluate(x_test_bnN,y_test, batch_size=bz)
        result_.append(result)
    return result_

def test_BG_noisy(n_list,inputs,isNormalized=1):
#   """
#   Multiplicative Bernoulli noise (aka binomial noise) constructs
#   a random mask m that is 1 for valid pixels and 0 for
#   zeroed/missing pixels. 
#   """
    result_ = []
    for n in n_list:
        x_test_bn = np.zeros((inputs.shape[0],28,28,1))
        for i in range(inputs.shape[0]):
            x_test_bn[i,:,:,:] = background_noise(inputs[i,:,:,:],n)
        if isNormalized:
            x_test_bnN = rescale(x_train,x_test_bn)
        else:
            x_test_bnN = x_test_bn
        result =model.evaluate(x_test_bnN,y_test, batch_size=bz)
        result_.append(result)
    return result_


noisy_result = {}
noisy_result['org_test'] = res
noisy_result['foreground'] = test_FG_noisy(np.arange(0,5.1,.1),x_test,0)
noisy_result['background'] = test_BG_noisy(np.arange(0,2.1,.1),x_test,0)



f = open(norm_type+"_mnist.pkl","wb")
pickle.dump(noisy_result,f)
f.close()
