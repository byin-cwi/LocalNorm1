# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
import keras
import os
import math
from utils import load_model
from console_progressbar import ProgressBar
from functions import rescale,noisy_image,noisy_MBN_image,noisy_Poisson_image,noisy_Poisson_image_add
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pickle
import random
import argparse

parser = argparse.ArgumentParser(description='Choose model...')
def decode_predictions(preds, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_names[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results

def evaluat_model(img_dir, model):
    valid_data_gen = ImageDataGenerator()
    valid_generator = valid_data_gen.flow_from_directory(img_dir, (224, 224), 
                                                         batch_size=16,
                                                         class_mode='categorical')
    return model.evaluate_generator(valid_generator, steps =102 ,workers = 4)


def predict_model(img_dir, model):
    valid_data_gen = ImageDataGenerator()
    valid_generator = valid_data_gen.flow_from_directory(img_dir, (224, 224), 
                                                         batch_size=16,
                                                         class_mode='categorical')
    valid_generator.reset()
    pred = []
    test_y = []
    for i in range(102):
        X,y = valid_generator.next()
        pred.append(model.predict(X))
        test_y.append(np.argmax(y,axis=1))
    pred = np.argmax(pred,axis=-1).reshape((1632,))
    y_test = np.array(test_y).reshape((1632,))
    return pred,y_test

def predict_noisy(img_dir, model,noise_type='none',sigma=.1):
    valid_data_gen = ImageDataGenerator()
    valid_generator = valid_data_gen.flow_from_directory(img_dir, (224, 224), 
                                                         batch_size=16,
                                                         class_mode='categorical')
    valid_generator.reset()
    pred = []
    test_y = []
    for i in range(102):
        x,y = valid_generator.next()
        X = np.zeros(x.shape)

        for j in range(16):
            if noise_type=='gaussian_pos':
                X[j,:,:,:] = noisy_image(x[j,:,:,:],sigma,negative=0)
            elif noise_type == 'gaussian_neg':
                X[j,:,:,:] = noisy_image(x[j,:,:,:],sigma,negative=1)
            elif noise_type == 'MBN':
                X[j,:,:,:] = noisy_MBN_image(x[j,:,:,:],sigma)
            elif noise_type =='none':
                X[j,:,:,:] = x[j,:,:,:]
            elif noise_type =='Poission':
                X[j,:,:,:] = noisy_Poission_image_add(x[j,:,:,:],sigma)

        pred.append(model.predict(X))
        test_y.append(np.argmax(y,axis=1))
    pred = np.argmax(pred,axis=-1).reshape((1632,))
    y_test = np.array(test_y).reshape((1632,))
    return pred,y_test

def evaluate_noisy(img_dir, model,noise_type='none',sigma=.1):
    valid_data_gen = ImageDataGenerator()
    valid_generator = valid_data_gen.flow_from_directory(img_dir, (224, 224), 
                                                         batch_size=16,
                                                         class_mode='categorical')
    valid_generator.reset()
    acc_list = []
    for i in range(102):
        x,y = valid_generator.next()
        # x=np.repeat(x,16,axis=0)
        # y = np.repeat(y,16,axis=0)
        X = np.zeros(x.shape)
        
        for j in range(16):
            if noise_type=='gaussian_pos':
                X[j,:,:,:] = noisy_image(x[j,:,:,:],sigma,negative=0)
            elif noise_type == 'gaussian_neg':
                X[j,:,:,:] = noisy_image(x[j,:,:,:],sigma,negative=1)
            elif noise_type == 'MBN':
                X[j,:,:,:] = noisy_MBN_image(x[j,:,:,:],sigma)
            elif noise_type =='none':
                X[j,:,:,:] = x[j,:,:,:][0]
            elif noise_type =='Poisson':
                X[j,:,:,:] = noisy_Poission_image(x[j,:,:,:],sigma)
            #X[j,:,:,:] = noisy_Poission_image_add(x[j,:,:,:],sigma)
        score = model.evaluate(X,y,verbose=2)
        acc_list.append(score[1])
    return np.mean(acc_list)



def calc_acc(y_pred, y_test):
    num_corrects = 0
    for i in range(num_samples):
        pred = y_pred[i]
        test = y_test[i]
        if pred == test:
            num_corrects += 1
    return num_corrects / num_samples*1.


if __name__ == '__main__':
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196
    class_names = range(1, (num_classes + 1))
    num_samples = 1629

    parser.add_argument('--model', type=str, 
                    help='Location of the model....')
    FLAGS = parser.parse_args()
    model_path = FLAGS.model

    print("\nLoad the trained ResNet model....")
    # model_path = 'models/local_.29-0.88.hdf5'#'models/batch_03-0.71.hdf5'#
    norm_type = model_path[7:12]
    model = load_model(model_path,norm_type)
    acc = evaluat_model('data/valid', model)
    print(acc)


    y_pred,y_test = predict_model('data/valid', model)
    print(y_pred.shape)
    print(y_test.shape)
    print("y_pred: " + str(y_pred))
    print("y_test: " + str(y_test))
    acc = calc_acc(y_pred, y_test)
    print(np.sum(y_pred==y_test)/len(y_pred)*1.)
    print("%s: %.2f%%" % ('acc', acc * 100))


    Noise_range_dic = {'gaussian_pos':[0.,.1,.2,.5,.8,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5],
                   'gaussian_neg':[0.,.1,.2,.5,.8,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5],
                   'MBN':np.arange(0,1,.05),
                   'Poission':[0.,.025,.05,.1,.25,.5,.75,1.]}

    noisy_result = {}
    noisy_result['org_test'] = acc
    for k,v in Noise_range_dic.items():
        print(k+' start...')
        acc_list = []
        for s in v:
            prob = evaluate_noisy('data/valid', model,noise_type=k,sigma=s)
            acc_list.append(prob)
            print('\t '+str(s)+':'+str(prob))
        noisy_result[k] = acc_list

    # norm_type = 'local'
    f = open(norm_type+".pkl","wb")
    print('Result written in '+norm_type+".pkl")
    pickle.dump(noisy_result,f)
    f.close()

