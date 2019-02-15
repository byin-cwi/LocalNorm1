# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import os
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers


from keras.layers.normalization import BatchNormalization
from localNorm import LocalNormalization
from switchNorm import SwitchNormalization

from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import rescale,noisy_image,noisy_MBN_image,noisy_Poisson_image
import argparse



from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras.utils.np_utils import to_categorical


parser = argparse.ArgumentParser(description='Choose Norm type')

np.random.seed(451)

from keras.datasets import cifar10

# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    


parser.add_argument('--Norm_type', type=str, default='batch',
                    help='LocalNorm,BatchNorm,SwitchNorm')
parser.add_argument('--group_number', type=int, default=8,
                    help='number of group')
parser.add_argument('--max_epoch', type=int, default=200,
                    help='max training epoch')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
FLAGS = parser.parse_args()
norm_type = FLAGS.Norm_type
group_number = FLAGS.group_number
epochs = FLAGS.max_epoch
learning_rate = FLAGS.learning_rate



def norm(x,norm_type):
        if norm_type=='batch':
            _output = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        elif norm_type=='local':
            _output = LocalNormalization(groupsize=int(128/group_number),batch_size=128)(x)
        elif norm_type=='switch':
            _output = SwitchNormalization()(x)
        elif norm_type=='none':
            _output = x
        return _output
      
stack_n            = 5
layers             = 6 * stack_n + 2
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 128
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 50:
        return .1
    if epoch < 122:
        return 0.01
    return 0.003# 0.001


def residual_network(img_input,classes_num=10,stack_n=5,norm_type = 'batch',isVis=0):
    
    def residual_block(x,o_filters,increase=False):
        stride = (1,1)
        if increase:
            stride = (2,2)

#         o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        o1 = Activation('relu')(norm(x,norm_type))
        conv_1 = Conv2D(o_filters,kernel_size=(3,3),strides=stride,padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
#         o2  = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        o2 = Activation('relu')(norm(conv_1,norm_type))
        conv_2 = Conv2D(o_filters,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x,16,False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x,32,True)
    for _ in range(1,stack_n):
        x = residual_block(x,32,False)
    # mark variable to verify the function of Normalization

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x,64,True)
    for _ in range(1,stack_n):
        x = residual_block(x,64,False)
    # mark variable to verify the function of Normalization

    x = norm(x,norm_type)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



x_train, x_test = color_preprocessing(x_train, x_test)

batch_size = 128
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

x_test = x_test[:(x_test.shape[0]//batch_size)*batch_size]
x_train = x_train[:(x_train.shape[0]//batch_size)*batch_size]
x_val = x_val[:(x_val.shape[0]//batch_size)*batch_size]

y_test = y_test[:(y_test.shape[0]//batch_size)*batch_size]
y_train = y_train[:(y_train.shape[0]//batch_size)*batch_size]
y_val = y_val[:(y_val.shape[0]//batch_size)*batch_size]


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print("== DONE! ==\n== BUILD MODEL... ==")
# build network
img_input = Input(shape=(img_rows,img_cols,img_channels))
output    = residual_network(img_input,num_classes,stack_n,norm_type)
resnet    = Model(img_input, output)

# print model architecture if you need.
# print(resnet.summary())
lr_decay = 1e-6
lr_drop = 30
dl = .75

# set optimizer
sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

change_lr = LearningRateScheduler(scheduler)

datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',cval=0.)
with tf.device('/gpu:0'):
    datagen.fit(x_train)

    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                       steps_per_epoch=iterations,
                       epochs=epochs,callbacks=[change_lr],verbose=2,
                       validation_data=(x_val, y_val))

result = resnet.evaluate(x_test,y_test, batch_size=128)
print("Accuracy on test set: ",result[1]*100,"%")
vot = 0
for i in range(len(x_test)):
    img = x_test[i,:,:,:]
    img_D = np.zeros((128,32,32,3))
    for j in range(128):
        img_D[j,:,:,:] = img
    preds = resnet.predict(img_D,batch_size=128)
    prob = np.max(preds)
    class_ids = np.argmax(preds,axis=-1)
    class_id = np.bincount(class_ids).argmax()
    if class_id == np.argmax(y_test[i]):
        vot+=1.
print("Accuracy on voting: ",vot*100./len(x_test),"%")
    


# resnet.save_weights('cifar10_ResNet'+norm_type+'c.h5')


def test_MBN_noisy(n_list,inputs,negative=0,preprocessing=0):
#   """
#   Multiplicative Bernoulli noise (aka binomial noise) constructs
#   a random mask m that is 1 for valid pixels and 0 for
#   zeroed/missing pixels. 
#   """
    result_ = []
    for n in n_list:
        x_test_bn = np.zeros((inputs.shape[0],32,32,3))
        for i in range(inputs.shape[0]):
            x_test_bn[i,:,:,:] = noisy_MBN_image(inputs[i,:,:,:],n)
        x_test_bnN = rescale(x_train,x_test_bn)
        result =resnet.evaluate(x_test_bnN,y_test, batch_size=128)
        result_.append(result)
    return result_

def test_Poisson_noisy(n_list,inputs,negative=0,preprocessing=0):
#   """
#   Additive Poisson noise
#   """
    result_ = []
    for n in n_list:
        x_test_bn = np.zeros((inputs.shape[0],32,32,3))
        for i in range(inputs.shape[0]):
            x_test_bn[i,:,:,:] = noisy_Poisson_image(inputs[i,:,:,:],n)
        x_test_bnN = rescale(x_train,x_test_bn)
        result =resnet.evaluate(x_test_bnN,y_test, batch_size=128)
        result_.append(result)
    return result_

def test_noisy(n_list,inputs,negative=0,preprocessing=0):
#   """
#   Additive gaussian noise
#   """
    result_ = []
    for n in n_list:
        x_test_bn = np.zeros((inputs.shape[0],32,32,3))
        for i in range(inputs.shape[0]):
            x_test_bn[i,:,:,:] = noisy_image(inputs[i,:,:,:],n,negative)
        x_test_bnN = rescale(x_train,x_test_bn)
        result =resnet.evaluate(x_test_bnN,y_test, batch_size=128)

        result_.append(result)#1-np.sum(residuals)/len(residuals))#
    return result_


noisy_result = {}
noisy_result['org_test'] = result
noisy_result['gaussian_pos'] = test_noisy([0.,.1,.2,.5,.8,1.,1.5,2.,2.5,5],x_test,0,0)
noisy_result['gaussian_neg'] = test_noisy([0.,.1,.2,.5,.8,1.,1.5,2.,2.5,5],x_test,0,1)
noisy_result['Poission'] = test_Poission_noisy([0.,.025,.05,.1,.25,.5,.75,1.],x_test,0,0)
noisy_result['MBN'] = test_MBN_noisy(np.arange(0,1,.05),x_test,0,0)

import pickle

f = open(norm_type+"_"+str(group_number)+"groups_resnet.pkl","wb")
pickle.dump(noisy_result,f)
f.close()
# python -mpickle batch.pkl
