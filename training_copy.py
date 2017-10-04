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



class ModelCheckpoint1(Callback):

    def __init__(self, filepath, logfilepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
#         super(ModelCheckpoint1, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        
        self.sep = ','
        self.filename = logfilepath
        #self.filename = '/work/vsankar/Project-Luna/Codes/t_a_000001_c_t_s_f_z_f_b_50.log'
        self.append = True
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        super(ModelCheckpoint1, self).__init__()
        

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)
            
    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
        
        
    def on_epoch_end_csv(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
    
    def on_epoch_end(self, epoch, logs=None):
#         Tracer()()
        logs = logs or {}
        self.epochs_since_last_save += 1
        
        self.on_epoch_end_csv(epoch, logs)
        
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                    
def model_architecture(img_rows,img_cols,img_channels,nb_classes):
    #function defining the architecture of defined CNN
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True, input_shape=(img_channels,img_rows, img_cols)))
    model.add(BN(axis=1, momentum=0.99, epsilon=0.00001))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(BN(axis=1, momentum=0.99, epsilon=0.00001))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))

    model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(BN(axis=1, momentum=0.99, epsilon=0.00001))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(BN(axis=1, momentum=0.99, epsilon=0.00001))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))

    model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

    model.add(Convolution2D(96, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(BN(axis=1, momentum=0.99, epsilon=0.00001))
    model.add(Convolution2D(96, 3, 3, activation='relu', border_mode='same',init='orthogonal', bias = True))
    model.add(BN(axis=1, momentum=0.99, epsilon=0.00001))
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

def run(batch_size,nb_classes,nb_epoch,data_augmentation,img_rows, img_cols,img_channels,weightfilepath,logfilepath   ):
    #function to run the actual test
    # the data, shuffled and split between train and test sets
    X_test,y_test,X_train,y_train = load_data()
    print (X_train.shape)
    
#     X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1) #reshapping it according to the keras rule
#     X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
    
    X_train,Y_train,X_test,Y_test = normalize_date(X_train,y_train,X_test,y_test,nb_classes)
#     Tracer()()


    X_small = X_train[1:100,:,:,:]
    Y_small = Y_train[1:100,:]
    
#     Tracer()()
    
    print('Loading and formatiing of data complete...')
    
    #load the model defined in model_architecture function
    model = model_architecture(img_rows,img_cols,img_channels,nb_classes)

    # training the model using SGD + momentum
    adm = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adm,
                  metrics=['accuracy'])

  

    model.load_weights(weightfilepath )

    save_model_per_epoch = ModelCheckpoint1(weightfilepath,logfilepath, monitor='val_loss', verbose=1, save_best_only=False)
#     Tracer()()
    if not data_augmentation:
        print('Not using data augmentation.')

        model.fit(X_test,Y_test,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  verbose=1,
                  callbacks=[save_model_per_epoch])
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            #shear_range = 0.34,  # value in radians, equivalent to 20 deg
            #zoom_range = [1/1.6, 1.6],   #same as in NIPS 2015 paper.
            #width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            #height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        #datagen.fit(X_train) #Not required as it is Only required if featurewise_center or featurewise_std_normalization or zca_whitening.

        # fit the model on the batches generated by datagen.flow() and save the loss and acc data history in the hist variable
        
#         filepath = "/work/vsankar/Project-Luna/luna_weights_1.hdf5"
#         save_model_per_epoch = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        hist = model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            verbose=1,
                            validation_data=(X_test, Y_test),
                            callbacks=[save_model_per_epoch])

        

        # serialize model to JSON
        
weightfilepath = "/work/vsankar/Project-Luna/Luna_weights/luna_weights_t_ae6_nt1_b64.hdf5"
logfilepath = '/work/vsankar/Project-Luna/Codes/t_ae6_nt1_b64.log'

batch_size = 64
nb_classes = 2
nb_epoch = 30
data_augmentation = True
# input image dimensions
img_rows, img_cols = 96,96
# the imgCLEF images are grey
img_channels = 1
run(batch_size,nb_classes,nb_epoch,data_augmentation,img_rows,img_cols,img_channels,weightfilepath,logfilepath  )
# model_architecture(img_rows,img_cols,img_channels,nb_classes)
