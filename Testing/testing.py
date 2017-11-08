#!/usr/bin/env python3
# code to replicate the performance of the NIPS'16 MLHC workshop paper "Cam CNNS predict anatomy?"

from __future__ import absolute_import
from __future__ import print_function

import os
import csv
import six

import numpy as np
import time
import json
import warnings

from collections import deque
from collections import OrderedDict
from collections import Iterable

# os.environ["THEANO_FLAGS"] = "device=gpu0,floatX=float32" 

os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ[
#     "THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,force_device=true"#,lib.cnmem=0.7 ,nvcc.flags=-D_FORCE_INLINES"
import sys
import numpy as np
from keras import backend as K
K.set_image_data_format('channels_first')
#from keras.datasets import cifar10
#keras.callbacks.Callback()
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Convolution2D, MaxPooling2D 
from keras.layers import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import load_model
from keras.datasets import mnist
from scipy import io
from IPython.core.debugger import Tracer
import matplotlib.pyplot as plt


def load_data():
    X = np.load('/work/vsankar/Project-Luna/image_array.npy')
    Y = np.load('/work/vsankar/Project-Luna/new_labels.npy')
    for i,y in enumerate(Y):
        if y > 4.0:
            Y[i] = 4.0

    for i,y in enumerate(Y):
        if y<=2.0:
            Y[i] = 0.0
        else:
            Y[i] = 1.0

    X_train = X[:2510]
    X_test = X[2510:]
    y_train = Y[:2510]
    y_test = Y[2510:]

    return X_train,y_train,X_test,y_test

                    
def model_architecture(img_rows,img_cols,img_channels,nb_classes):
    #function defining the architecture of defined CNN
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True, input_shape=(img_channels,img_rows, img_cols)))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))

    model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))

    model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

    model.add(Convolution2D(96, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(Convolution2D(96, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(Convolution2D(96, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))

    model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    Dropout((0.5))
    model.add(Convolution2D(512, 1, 1, activation='relu', border_mode='same',init='orthogonal', bias = True))
    Dropout((0.5))
    model.add(Convolution2D(2, 1, 1, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(GlobalAveragePooling2D(dim_ordering='default'))

    #model.add(Convolution2D(10,1,1, border_mode='same',init='orthogonal', bias = True))
    #model.add(Dense(nb_classes))

    model.add(Activation('softmax'))
    model.summary()
    return model

def normalize_date(X_train,Y_train,X_test,Y_test,nb_classes):
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')

    #normalizing the data
    X_train /= 4095.0   
    X_test /= 4095.0
    
    #std
#     X_train = X_train/np.std(X_train) - np.mean(X_train)
#     X_test = X_test/np.std(X_test) - np.mean(X_test)
    
#     Tracer()()
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    
    return X_train,Y_train,X_test,Y_test

def run(batch_size,nb_classes,nb_epoch,data_augmentation,img_rows, img_cols,img_channels,model_name):
    #function to run the actual test
    # the data, shuffled and split between train and test sets
    X_test,y_test,X_train,y_train = load_data()
    print (X_train.shape)
    
#     X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1) #reshapping it according to the keras rule
#     X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
    
    X_train,Y_train,X_test,Y_test = normalize_date(X_train,y_train,X_test,y_test,nb_classes)
#     Tracer()()


    #X_small = X_train[1:100,:,:,:]
    #Y_small = Y_train[1:100,:]
    
#     Tracer()()
    
    print('Loading and formatiing of data complete...')
    
    #load the model defined in model_architecture function
#     model = model_architecture(img_rows,img_cols,img_channels,nb_classes)

    # training the model using SGD + momentum
#     adm = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=adm,
#                   metrics=['accuracy'])

  
    #filepath = "/work/vsankar/Project-Luna/Luna_weights/luna_weights_t_a_000001_c_t_s_f_z_f_b_50.hdf5"
    modelPath = '/work/vsankar/Project-Luna/Luna_weights/'+model_name+'.hdf5'
#     Tracer()()
    model = load_model(modelPath)
#     Tracer()()
#     model.predict( X_test, batch_size=None, verbose=0, steps=None)

    return model

        # serialize model to JSON
        
model_name = 'luna_weights_t_ae6_nt_b50_BN0_best2'     
batch_size = 50
nb_classes = 2
nb_epoch = 30
data_augmentation = True
# input image dimensions
img_rows, img_cols = 96,96
# the imgCLEF images are grey
img_channels = 1
model = run(batch_size,nb_classes,nb_epoch,data_augmentation,img_rows,img_cols,img_channels,model_name)

# model_architecture(img_rows,img_cols,img_channels,nb_classes)

X_test,y_test,X_train,y_train = load_data()
X_train,Y_train,X_test,Y_test = normalize_date(X_train,y_train,X_test,y_test,nb_classes)
y = model.predict( X_test, batch_size=1, verbose=0)
#print(y)
print("performance of 2")
a = model.evaluate(X_test, Y_test, batch_size=1, verbose=0)
print(a)
