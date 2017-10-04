from __future__ import print_function
import os

#os.environ['KERAS_BACKEND']='theano'
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,force_device=true,lib.cnmem=0.9"#,nvcc.flags=-D_FORCE_INLINES"
 
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import load_model
from IPython.core.debugger import Tracer
import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

# "/work/vsankar/projects/minist/"
working_path = "/work/vsankar/projects/Luna_data/no_crop_norm/"

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
	inputs = Input((1, 512, 512))
	conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
	conv1 = Dropout(0.2)(conv1)
	conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
	conv2 = Dropout(0.2)(conv2)
	conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
	conv3 = Dropout(0.2)(conv3)
	conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
	conv4 = Dropout(0.2)(conv4)
	conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
	conv5 = Dropout(0.2)(conv5)
	conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

	up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
	conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
	conv6 = Dropout(0.2)(conv6)
	conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

	up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
	conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
	conv7 = Dropout(0.2)(conv7)
	conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

	up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
	conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
	conv8 = Dropout(0.2)(conv8)
	conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

	up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
	conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
	conv9 = Dropout(0.2)(conv9)
	conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

	conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

	model = Model(input=inputs, output=conv10)
	# model.summary()
	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

	return model

def train_and_predict(use_existing):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    
    imgs_train = np.load(working_path+"trainImages_nocrop_norm1.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path+"trainMasks_nocrop_norm1.npy").astype(np.int)

    imgs_test = np.load(working_path+"testMasks_nocrop_norm1.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path+"testImages_nocrop_norm1.npy").astype(np.int)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()

    model_checkpoint = ModelCheckpoint('/work/vsankar/Project-Luna/weights/no_crop_norm/weights_halfdata.{epoch:02d}.hdf5',verbose=1, monitor='loss',save_best_only=False)

    if use_existing:
        print('using exisitng weights')
        model.load_weights('/work/vsankar/Project-Luna/weights/no_crop_norm/weights_halfdata.21.hdf5')

#     Tracer()()
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=40, verbose=1, shuffle=True, validation_split=0.2,
              callbacks=[model_checkpoint])

if __name__ == '__main__':
    train_and_predict(False)
