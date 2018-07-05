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
import scipy.misc

import sys
print "This is the name of the script: ", sys.argv[0]
print "Number of arguments: ", len(sys.argv)
print "The arguments are: " , str(sys.argv)


start = int(sys.argv[1]);
stop = int(sys.argv[2]);

working_path = "/work/vsankar/Project-Luna/segmentedNodules2/"
images_path = "/work/vsankar/Project-Luna/Output_toolbox1/images/"
saving_path = "/work/vsankar/Project-Luna/data/array_new/"

def Y_threshold(Y):
#     Tracer()()
    Yr = np.round(Y)
    
    if Y<3:
        Yr = 0.0            
    elif Y==3:
        Yr = 1.0
    elif Y > 3:
        Yr = 2.0

#     Tracer()()
    return Yr

def returnWindow(x,y,step):

    if x - step > 0 and x + step < 512:
        xl = x-step ; 
        xh = x+step-1 ; 
    elif x - step <= 0:
        xl = 1;
        xh = 96;
    elif  x + step >= 512:   
        xl = 512-step*2 ;
        xh = 512 - 1;
    
    if y - step > 0 and y + step < 512:
        yl = y-step ; 
        yh = y+step-1 ; 
    elif y - step <= 0:
        yl = 1;
        yh = 96;
    elif  y + step >= 512:   
        yl = 512-step*2 ;
        yh = 512 - 1;    
    return xl,xh,yl,yh


def returnSquareMask(msk,img):
#     try:
#     Tracer()()
    idxr,idxc = np.nonzero(msk)
    rmax = max(idxr)
    cmax = max(idxc)
    rmin = min(idxr)
    cmin = min(idxc)

    img1 = img.copy()

    img1_RGB = np.asarray(scipy.misc.toimage(img1,mode='L').convert("RGB"))
    img1_RGB.setflags(write=1)

    img1_RGB[rmin:rmax,cmin,0] = 255;
    img1_RGB[rmin:rmax,cmax,0] = 255;
    img1_RGB[rmin,cmin:cmax,0] = 255;
    img1_RGB[rmax,cmin:cmax,0] = 255;

    step=48;
    x = int(np.mean(idxr))
    y = int(np.mean(idxc))

    [xl,xh,yl,yh] = returnWindow(x,y,step);
    cropped_img = img1_RGB[xl:xh+1,yl:yh+1];

    smsk = np.zeros_like(msk);
    smsk[rmin:rmax,cmin:cmax] = 1
#     Tracer()()
#     if len(cropped_img) != 96:
#         Tracer()()
    
    return smsk,cropped_img


# file_list=glob(working_path+"images_*.npy")

images = np.ndarray([1,1,96,96],dtype=np.float32)
original_mask = np.ndarray([1,1,512,512],dtype=np.float32)
square_mask = np.ndarray([1,1,512,512],dtype=np.float32)

c0=0;c1=0;c2=0;
c=0;
mal = np.array([],dtype=np.float32)
label = np.array([],dtype=np.float32)

slice_images = np.ndarray([1,1,512,512],dtype=np.float32)
nodule_images_border = np.ndarray([1,1,96,96,3],dtype=np.float32)

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
                 
                slice_mal.append(np.mean([x for x in slice_mag_values if x>0]))
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
                nodule_image_b = np.ndarray([1,1,96,96,3],dtype=np.float32)
                smsk[0,0],nodule_image_b[0,0] = returnSquareMask(msk[0,0],slice_img[0,0])
                
                square_mask = np.append(square_mask, msk, axis=0)
                nodule_images_border = np.append(nodule_images_border, nodule_image_b, axis=0)

                scipy.misc.imsave(saving_path+'images/nodule_'+ str(pn) +'_'+ str(sl) +'_'+ str(n1) +'_'+'_.png', img[0,0])
                scipy.misc.imsave(saving_path+'images/nodule_images_border_'+ str(pn) +'_'+ str(sl) +'_'+ str(n1) +'_'+'_.png', nodule_image_b[0,0])
                c=c+1
                #malignancy
#                 Tracer()()
                
                
                yr = Y_threshold(slice_mal[sl_num-1])
                
                if yr == 1:
                    c1 = c1+1;
                elif yr == 0:
                    c0 = c0+1;
                elif yr == 2:
                    c2 = c2+1
                    
                mal = np.append(mal,slice_mal[sl_num-1])
                label = np.append(label,yr)
                
    except IOError as e:
        print e
#         Tracer()()
#     print mal.shape
#     print images.shape
print('#0 = ' + str(c0))    
print('#1 = ' + str(c1))
print('#2 = ' + str(c2))    
    
print('size of images = '+ str(len(images)))    
images = np.delete(images, 0, 0)
nodule_images_border = np.delete(nodule_images_border, 0, 0)

print(saving_path+"Images_mask1_"+str(start)+"_"+ str(stop)+"_.npy")
np.save(saving_path+"Images_mask1_"+str(start)+"_"+ str(stop)+"_.npy",images)

# np.save(saving_path+"square_mask_"+str(start)+"_"+ str(stop)+"_.npy",square_mask)
# np.save(saving_path+"originak_mask_"+str(start)+"_"+ str(stop)+"_.npy",original_mask)
# np.save(saving_path+"slice_images.npy",slice_images)
np.save(saving_path+"nodule_images_border1_"+str(start)+"_"+ str(stop)+"_.npy",nodule_images_border)
np.save(saving_path+"mal_mask1_"+str(start)+"_"+ str(stop)+"_.npy",mal)
np.save(saving_path+"label1_"+str(start)+"_"+ str(stop)+"_.npy",label )