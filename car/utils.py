# -*- coding: utf-8 -*-

# import cv2 as cv

from resnet_152 import resnet152_model


def load_model(path_dir,norm_type):
    model_weights_path = path_dir#'models/model.96-0.89.hdf5'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196
    model = resnet152_model(img_height, img_width, num_channels, isTrain=0,
                        num_classes=num_classes,norm_type=norm_type)
    model.load_weights(model_weights_path, by_name=True)
    return model
