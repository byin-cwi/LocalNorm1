# -*- coding: utf-8 -*-

import numpy as np
import keras
from keras import backend as K
from resnet_152 import resnet152_model
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import SGD

from functions import rescale,noisy_image,noisy_MBN_image,noisy_Poission_image
import argparse

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers


from keras.layers.normalization import BatchNormalization
from localNorm2 import LocalNormalization
from switchNorm import SwitchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras.utils.np_utils import to_categorical

img_width, img_height = 224, 224
num_channels = 3
train_data = 'data/train'
valid_data = 'data/valid'
num_classes = 196

verbose = 1
batch_size = 16
num_epochs = 150
patience = 50

num_train_samples = (6515//batch_size)*batch_size
num_valid_samples = (1629//batch_size)*batch_size
parser = argparse.ArgumentParser(description='Choose Norm type')
weight_decay       = 1e-4

if __name__ == '__main__':
    # build a classifier model

    parser.add_argument('--Norm_type', type=str, default='batch',
                    help='LocalNorm,BatchNorm,SwitchNorm')
    FLAGS = parser.parse_args()
    norm_type = FLAGS.Norm_type

    model = resnet152_model(img_height, img_width, num_channels, num_classes,norm_type=norm_type)
    # prepare data augmentation configuration
    train_data_gen = ImageDataGenerator(rotation_range=20.,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
    valid_data_gen = ImageDataGenerator()
    # callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    log_file_path = 'logs/training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_acc', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)
    trained_models_path = 'models/'+norm_type+'_g1/'
    model_names = trained_models_path + '{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr]

    # generators
    train_generator = train_data_gen.flow_from_directory(train_data, (img_width, img_height), 
                                                         batch_size=batch_size,
                                                         class_mode='categorical')
    valid_generator = valid_data_gen.flow_from_directory(valid_data, (img_width, img_height), 
                                                         batch_size=batch_size,
                                                         class_mode='categorical')

    # fine tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples / batch_size,
        validation_data=valid_generator,
        validation_steps=num_valid_samples/batch_size,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=verbose)
