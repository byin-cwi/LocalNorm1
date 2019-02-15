# -*- coding: utf-8 -*-

# import the necessary packages
import json
import os
import random

import cv2 as cv

import numpy as np
import scipy.io

if __name__ == '__main__':

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)
    
    text_file = open("classnames.txt", "w")
    for i in range(len(class_names)):
        n = class_names[i][0][0]
        text_file.writelines(str(i)+': '+n+'\n')
    text_file.close()
