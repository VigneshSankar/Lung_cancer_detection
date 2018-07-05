import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
# from skimage.io import imread,imshow,show
from glob import glob
from os import listdir
import csv
import matplotlib.pyplot as plt
from skimage.draw import polygon_perimeter
from IPython.core.debugger import Tracer

import sys
print "This is the name of the script: ", sys.argv[0]
print "Number of arguments: ", len(sys.argv)
print "The arguments are: " , str(sys.argv)


start = int(sys.argv[1]);
stop = int(sys.argv[2]);



working_path = "/work/vsankar/Project-Luna/segmentedNodules2/"
images_path = "/work/vsankar/Project-Luna/Output_toolbox1/images/"
saving_path = "/work/vsankar/Project-Luna/data/array/"



def returnSquareMask(msk,img):
    idxr,idxc = np.nonzero(msk)
    rmax = max(idxr)
    cmax = max(idxc)
    rmin = min(idxr)
    cmin = min(idxc)
    
    img1 = img.copy()
    img1[rmin:rmax,cmin] = np.amax(img)+1000;
    img1[rmin:rmax,cmax] = np.amax(img)+1000;
    img1[rmin,cmin:cmax] = np.amax(img)+1000;
    img1[rmax,cmin:cmax] = np.amax(img)+1000;
    
    
    smsk = np.zeros_like(msk);
    smsk[rmin:rmax,cmin:cmax] = 1
    
        # ploting
#     plt.subplot(1,2,1)
#     plt.imshow(msk)
#     plt.subplot(1,2,2)
#     plt.imshow(smsk)
#     plt.show()
#                 plt.subplot(1,3,3)
#     plt.imshow(img1)
#     plt.show()
#     Tracer()()
    
    
#     Tracer()()
    return smsk,img1


# file_list=glob(working_path+"images_*.npy")

images = np.ndarray([1,1,96,96],dtype=np.float32)
original_mask = np.ndarray([1,1,512,512],dtype=np.float32)
square_mask = np.ndarray([1,1,512,512],dtype=np.float32)

mal = np.array([],dtype=np.float32)

slice_images = np.ndarray([1,1,512,512],dtype=np.float32)
slice_images_border = np.ndarray([1,1,512,512],dtype=np.float32)

dir = listdir(working_path)
# Tracer()()
for pn in range(start,stop):
    print(pn)
    patientID = dir[pn]
    PatientID_path =  working_path + patientID + '/'
    slice_list =glob(PatientID_path+"slice*")

#     Tracer()() 
    slice_mal = list([]);
    try:
#         Tracer()()
        
        if len(slice_list)==0:
#             Tracer()()
            continue
        with open(PatientID_path+'malignancy.csv', 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                slice_mag_values = map(int,row)
                 
                slice_mal.append(np.mean([x for x in slice_mag_values if x]))
#             Tracer()() 

        for sl in range(len(slice_list)): 
#             Tracer()() 
            slice_path = slice_list[sl]

            try:
                sl_num = int(slice_list[sl][-2:])
            except:
                sl_num = int(slice_list[sl][-1])

            imagefiles = glob(slice_path+"/nodule_*")
            for n1 in range(len(imagefiles)):
                
#                 Tracer()() 
                rad_num = imagefiles[n1][-6]
                #nodule image
                img = np.ndarray([1,1,96,96],dtype=np.float32)
                image_name = slice_path + "/nodule_image"+str(rad_num)+".tiff"
                img[0,0] = plt.imread(image_name)
                images = np.append(images, img, axis=0)
                
#                 Tracer()() 
                #slice image
                slice_img = np.ndarray([1,1,512,512],dtype=np.float32)
                slice_image_name = images_path + patientID + '/slice'+ str(sl_num) + '.tif'
                slice_img[0,0] = plt.imread(slice_image_name)
                slice_images = np.append(slice_images, slice_img, axis=0)
                
#                 Tracer()() 
                #mask original
                msk = np.ndarray([1,1,512,512],dtype=np.float32)
                msk_name = slice_path + "/nodulemask"+str(rad_num)+".tiff"
                msk[0,0] = plt.imread(msk_name)
                original_mask = np.append(original_mask, msk, axis=0)
                
#                 Tracer()() 
                #mask square
                smsk = np.ndarray([1,1,512,512],dtype=np.float32)
                slice_image_b = np.ndarray([1,1,512,512],dtype=np.float32)
                smsk[0,0],slice_image_b[0,0] = returnSquareMask(msk[0,0],slice_img[0,0])
                
                square_mask = np.append(square_mask, msk, axis=0)
                slice_images_border = np.append(slice_images_border, slice_image_b, axis=0)
                
                #malignancy
                mal = np.append(mal,slice_mal[sl_num-1])
                

                
                
    except IOError as e:
        print e
#         Tracer()()
#     print mal.shape
#     print images.shape
    

    
    
    
images= np.delete(images, 0, 0)
slice_images_border= np.delete(slice_images_border, 0, 0)

print("size of images =      ")
print(images.shape)
    
np.save(saving_path+"Images_mask_"+str(start)+"_"+ str(stop)+"_.npy",images)
np.save(saving_path+"square_mask_"+str(start)+"_"+ str(stop)+"_.npy",square_mask)
np.save(saving_path+"originak_mask_"+str(start)+"_"+ str(stop)+"_.npy",original_mask)
# np.save(saving_path+"slice_images.npy",slice_images)
np.save(saving_path+"slice_images_border_"+str(start)+"_"+ str(stop)+"_.npy",slice_images_border)
np.save(saving_path+"mal_mask_"+str(start)+"_"+ str(stop)+"_.npy",mal)