# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
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
    
def norm(norm_type):
        if norm_type=='batch':
            _output = BatchNormalization()
        elif norm_type=='local':
            _output = LocalNormalization(groupsize=16,batch_size=128)
        elif norm_type=='switch':
            _output = SwitchNormalization()
        return _output
      
class cifar10vgg:
    def __init__(self,train=True,norm_type = 'batch'):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.norm_type = norm_type
        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
            self.model.load_weights('cifar10vgg'+self.norm_type+'.h5')
    

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))
#         model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))
#         model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))
#         model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))
#         model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))
#         model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))
#         model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))
#         model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(norm(self.norm_type))

        model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
 

#         model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)
      
    def evaluation(self, x_test,y_test,normalize=True,batch_size=128):
        if normalize:
              x_test = self.normalize_production(x_test)
        return self.model.evaluate(x_test,y_test, batch_size=batch_size)[1]
      
    def train(self,model):
        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = .1
        lr_decay = 1e-6
        lr_drop = 30
        dl = .75
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

        x_test = x_test[:(x_test.shape[0]//batch_size)*batch_size]
        x_train = x_train[:(x_train.shape[0]//batch_size)*batch_size]
        x_val = x_val[:(x_val.shape[0]//batch_size)*batch_size]

        y_test = y_test[:(y_test.shape[0]//batch_size)*batch_size]
        y_train = y_train[:(y_train.shape[0]//batch_size)*batch_size]
        y_val = y_val[:(y_val.shape[0]//batch_size)*batch_size]

        def lr_scheduler(epoch):
            return learning_rate * (dl ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        



        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        print(model.summary())

        # training process in a for loop with learning rate drop every 25 epoches.
        with tf.device('/gpu:0'):
            datagen.fit(x_train)
            historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=batch_size),
                                steps_per_epoch=x_train.shape[0] // batch_size,
                                epochs=maxepoches,
                                validation_data=(x_val, y_val),callbacks=[reduce_lr],verbose=2)
        model.save_weights('cifar10vgg'+self.norm_type+'.h5')
        print(model.evaluate(x_test,y_test, batch_size=batch_size)[1])
        return model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

parser.add_argument('--Norm_type', type=str, default='batch',
                    help='LocalNorm,BatchNorm,SwitchNorm')
FLAGS = parser.parse_args()
norm_type = FLAGS.Norm_type

model = cifar10vgg(norm_type=norm_type)

batch_size=128
x_test = x_test[:(x_test.shape[0]//batch_size)*batch_size]
x_train = x_train[:(x_train.shape[0]//batch_size)*batch_size]


y_test = y_test[:(y_test.shape[0]//batch_size)*batch_size]
y_train = y_train[:(y_train.shape[0]//batch_size)*batch_size]
res = model.evaluation(x_test,y_test, batch_size=128)
print("Test accuracy:",res)


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
        result =model.evaluation(x_test_bnN,y_test, batch_size=128)
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
            x_test_bn[i,:,:,:] = noisy_Poission_image(inputs[i,:,:,:],n)
        x_test_bnN = rescale(x_train,x_test_bn)
        result =model.evaluation(x_test_bnN,y_test, batch_size=128)
        result_.append(result)
    return result_

def test_noisy(n_list,inputs,negative=0,preprocessing=0):
    #   """
    #   Additive Gaussian noise
    #   """
    result_ = []
    for n in n_list:
        x_test_bn = np.zeros((inputs.shape[0],32,32,3))
        for i in range(inputs.shape[0]):
            x_test_bn[i,:,:,:] = noisy_image(inputs[i,:,:,:],n,negative)
        x_test_bnN = rescale(x_train,x_test_bn)
        result =model.evaluation(x_test_bnN,y_test, batch_size=128)

        result_.append(result)
    return result_


noisy_result = {}
noisy_result['org_test'] = res
noisy_result['gaussian_pos'] = test_noisy([0.,.1,.2,.5,.8,1.,1.5,2.,2.5,5],x_test,0,0)
noisy_result['gaussian_neg'] = test_noisy([0.,.1,.2,.5,.8,1.,1.5,2.,2.5,5],x_test,0,1)
noisy_result['Poission'] = test_Poission_noisy([0.,.025,.05,.1,.25,.5,.75,1.],x_test,0,0)
#noisy_result['MBN'] = test_MBN_noisy([0.,.025,.05,.075,.1,.15,.2,.5,.7,.9],x_test,0,0)
noisy_result['MBN'] = test_MBN_noisy(np.arange(0,1,.05),x_test,0,0)
import pickle

f = open(norm_type+"_c.pkl","wb")
pickle.dump(noisy_result,f)
f.close()

# python -mpickle batch.pkl
