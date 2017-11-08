import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

#from __future__ import print_function
import os

os.environ['KERAS_BACKEND']='theano'
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,force_device=true,lib.cnmem=0.9"#,nvcc.flags=-D_FORCE_INLINES"
 
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import load_model
from IPython.core.debugger import Tracer
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
INPUT_FOLDER = '/work/vsankar/projects/kaggle_data/stage1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    ret = np.ndarray([len(image),1,512,512],dtype=np.float32)
    for i in range (len(image)):
        ret[i,0] = image[i]
    return ret



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


model2 = get_unet()
model2.load_weights('/work/vsankar/Project-Luna/weights/no_crop_norm/weights_halfdata.21.hdf5')


patients_folder='/work/vsankar/projects/lungCancer/'
df_train = pd.read_csv(patients_folder+'/stage1_labels.csv')
numPatients = len(patients)
PatientsDict = {}
PatientsPredictedDict = {}
t=50
c=0
#patients[2] = '123' 
for ii in range(30):
    c=0
    PatientsPredictedDict = {}
    PatientsDict = {}
    for ij in range(t):
        i= (ii+6)*t + ij
        print(i)
        print('c = %d' %(c))
        label = df_train['cancer'][df_train['id']==patients[i]] 
        if label.empty == False:
            first_patient = load_scan(INPUT_FOLDER + patients[i])
            first_patient_pixels = get_pixels_hu(first_patient)
        

            imgs_test = np.ndarray([len(first_patient_pixels),1,512,512],dtype=np.float32)
            imgs_mask_test = np.ndarray([len(first_patient_pixels),1,512,512],dtype=np.float32)

            imgs_test = first_patient_pixels

            for n_imgs in range(len(first_patient_pixels)):
                img = imgs_test[n_imgs,0]
                mean = np.mean(img)
                std = np.std(img)
                img = img-mean
                img = img/std
                imgs_test[n_imgs,0] = img

            imgs_mask_test = model2.predict(imgs_test, verbose=1)

            for ni in range(len(imgs_test)):
                #imgs_mask_test[ni] = model2.predict(imgs_test[ni:ni+1], verbose=0)[0]
                imgs_mask_test[ni] = imgs_mask_test[ni]*imgs_test[ni,0]

            PatientsDict[c] = (first_patient_pixels,label)
            PatientsPredictedDict[c] = (imgs_mask_test,label)
            c=c+1
            
        
    #print('saving predict')
    np.save('/work/vsankar/projects/Kaggle_data_Predict/PatientsDict_%d.npy' % (ii+6),PatientsDict)

    np.save('/work/vsankar/projects/Kaggle_data_Predict/PatientsPredictedDict_%d.npy' % (ii+6),PatientsPredictedDict)
    
    numPat = len(PatientsPredictedDict)
    Pat_mean_full = {}
    nsize = 20
    #print(numPat)
    for ia in range(numPat):
        Pat = PatientsPredictedDict[ia][0]
        lenPat = len(Pat)
        #print ('old=')
        #print (lenPat)
        remainder = lenPat%nsize
        if remainder%2 == 0:
            red = remainder/2
            Pat = Pat[red:,:,:,:]
            Pat = Pat[:-red,:,:,:]
        else:
            red  = remainder/2
            Pat = Pat[red+1:,:,:,:]
            Pat = Pat[:-red,:,:,:]
        lenPat = len(Pat)
        #print ('new =')
        #print (lenPat)
        #print('    ')

        q = lenPat/nsize
        pat_mean = np.ndarray([nsize,1,512,512],dtype=np.float32)
        for j in range(nsize):
    #         print(j)
            pat_mean[j] = np.mean(Pat[j*q:j*(q+1),:,:,:],axis=0)

        Pat_mean_full[ia] = (pat_mean,PatientsPredictedDict[ia][1])
    #print('saving avg')
    np.save('/work/vsankar/projects/Kaggle_data_Predict/Pat_mean_full_%d.npy'% (ii+6),Pat_mean_full)