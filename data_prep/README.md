Dataset Preparation:

The LIDC dataset is processed to give
  -Nodules images
  -Lables: Malignant or benign
  
1) First we make use of LIDC Toolbox master to structure the dataset according to partients.
The toolbox will create three separate folders. All the images in the folders are divided by patients ID.
  -gts
  -images
  -masks
https://github.com/TesterTi/LIDCToolbox

2) Then, we ran the matlab program "main.m" to segment the nodules for each patient and acquire the nodule size for the corresponding patients
The Masks from the previous step is used to find the nodule location and 96x96 windows is segmented for each nodule. The nodule size marked by the 
radiologist is read from the XML data of each patients to form the labels.

3) THe images and labels form the dataset for training. First the labels are converting to binary values to indicate malignant or benign.
labels(labels > upper_therhold ) = 1 (malginant)
labels(labels < lower_therhold ) = 0 (benign)
Initial experiments are done using the following threshold.
upper_therhold = 4
lower_therhold = 2

4)Since the dataset is imbalanced, as the number of malignant nodules is larger then the benign nodules. We use keras.datageneration to increase 
the number of benign cases. Here the images  are rotated and translated to create more data.

5) Then the dataset is divided into traning, testing and validation set. Approx 10% of the data is used for testing and another 10% is used for testing.

