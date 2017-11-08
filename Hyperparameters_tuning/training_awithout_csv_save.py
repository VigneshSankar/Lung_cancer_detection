#!/usr/bin/env python3
# code to replicate the performance of the NIPS'16 MLHC workshop paper "Cam CNNS predict anatomy?"

from __future__ import print_function
import os
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
#    model.summary()
    return model

def run(batch_size,nb_classes,nb_epoch,data_augmentation,img_rows, img_cols,img_channels):
    #function to run the actual test
    # the data, shuffled and split between train and test sets
    X_test,y_test,X_train,y_train = load_data()
    print (X_train.shape)
    
#     X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1) #reshapping it according to the keras rule
#     X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    X_train /= 4095.0   #normalizing the data
    X_test /= 4095.0
#     Tracer()()
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
#     Tracer()()
    
    print('Loading and formatiing of data complete...')
    
    #load the model defined in model_architecture function
    model = model_architecture(img_rows,img_cols,img_channels,nb_classes)

    # training the model using SGD + momentum
    adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adm,
                  metrics=['accuracy'])

  
    filepath = "/work/vsankar/Project-Luna/Luna_weights/luna_weights{epoch:02d}.hdf5"

    #model.load_weights('/work/vsankar/Project-Luna/Luna_weights/luna_weights{epoch:02d}.hdf5')

    save_model_per_epoch = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False)

    class TestCallback(Callback):
    	def __init__(self, test_data):
        	self.test_data = test_data

    	def on_epoch_end(self, epoch, logs={}):
        	xt, yt, xtr, ytr = self.test_data

        	losst, acct = self.model.evaluate(xt, yt, verbose=0)
        	print('\nTesting loss: {}, acc: {}\n'.format(losst, acct))

		losstr, acctr = self.model.evaluate(xtr, ytr, verbose=0)
        	print('\nTraining loss: {}, acc: {}\n'.format(losstr, acctr))

		file = open('/work/vsankar/Project-Luna/Codes/training_out_file.txt','a') 
 		file.write('\nepoch: {}'.format(epoch)	
		file.write('Testing loss: {}, acc: {}\n'.format(losst, acct)) 
		file.write('Training loss: {}, acc: {}\n'.format(losstr, acctr))

		file.close() 


    if not data_augmentation:
        print('Not using data augmentation.')
        
	

        model.fit(X_train,Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  verbose=1,
                  callbacks=[TestCallback((X_test, Y_test,X_train,Y_train))])
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=True,  # apply ZCA whitening
            rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
            #shear_range = 0.34,  # value in radians, equivalent to 20 deg
            #zoom_range = [1/1.6, 1.6],   #same as in NIPS 2015 paper.
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        #datagen.fit(X_train) #Not required as it is Only required if featurewise_center or featurewise_std_normalization or zca_whitening.

        # fit the model on the batches generated by datagen.flow() and save the loss and acc data history in the hist variable
        
        filepath = "/work/vsankar/Project-Luna/luna_weights_1.hdf5"
        save_model_per_epoch = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        hist = model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test),
                            callbacks=[save_model_per_epoch])

        

        # serialize model to JSON
        
     
batch_size = 2
nb_classes = 2
nb_epoch = 300
data_augmentation = False
# input image dimensions
img_rows, img_cols = 96,96
# the imgCLEF images are grey
img_channels = 1
run(batch_size,nb_classes,nb_epoch,data_augmentation,img_rows,img_cols,img_channels)
# model_architecture(img_rows,img_cols,img_channels,nb_classes)