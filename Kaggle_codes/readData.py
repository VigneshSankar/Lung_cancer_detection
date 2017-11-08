import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
# from skimage.io import imread,imshow,show
from glob import glob
from os import listdir
import csv
# %matplotlib nbagg
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer
working_path = "/work/vsankar/Project-Luna/segmentedNodules1/"
# file_list=glob(working_path+"images_*.npy")

images = np.ndarray([1,1,96,96],dtype=np.float32)
mal = np.array([],dtype=np.float32)

dir = listdir(working_path)

for pn in range(900): 
    patientID = dir[pn]
    PatientID_path =  working_path + patientID + '/'
    slice_list =glob(PatientID_path+"slice*")

 
    slice_mal = list([]);
    try:
        
        with open(PatientID_path+'malignancy.csv', 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                slice_mag_values = map(int,row)
                slice_mal.append(np.mean([x for x in slice_mag_values if x]))


        for sl in range(len(slice_list)): 
            slice_path = slice_list[sl]

            try:
                sl_num = int(slice_list[sl][-2:])
            except:
                sl_num = int(slice_list[sl][-1])

            imagefiles = glob(slice_path+"/nodule_*")
            for n in range(len(imagefiles)):
                img = np.ndarray([1,1,96,96],dtype=np.float32)
                img[0,0] = plt.imread(imagefiles[n])
                images = np.append(images, img,axis=0)
                mal = np.append(mal,slice_mal[sl_num-1])
    except IOError as e:
        print e
#     print mal.shape
#     print images.shape
    print(pn)
    
np.save(working_path+"Images.npy",images)
np.save(working_path+"mal.npy",mal)