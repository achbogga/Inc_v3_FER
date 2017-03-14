from keras.utils import np_utils
import keras.backend as K
import cv2
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from sklearn.cross_validation import train_test_split
#import matplotlib.pyplot as plt
#import matplotlib
from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

def copy_to_n_channels(input_arr, n):
    temp = np.zeros(input_arr.shape+(n,), dtype='float32')
    for i in range(n):
        temp[:, :, :, i] = input_arr
    return temp

def convert_to_grey_scale(source, dest, img_rows, img_cols):
	#source = 'CKplus/dataset/images'
	#dest = 'CKplus/dataset/processed_inc_v3'
	listing = os.listdir(source)
	num_samples=len(listing)
	for file in listing:
	    im = Image.open(source+'/'+file)
	    img = im.resize((img_rows, img_cols))
	    gray = img.convert('L')
	    gray.save(dest+'/'+file, "PNG")
	print "\n saved gray scale images to destination."

def prepare_for_inc_v3(img_rows=299, img_cols=299, img_chs=3, out_put_classes=7, img_src='CKplus/dataset/processed_inc_v3', label_src='CKplus/dataset/labels/emotion_labels.txt', asGray=False, face_detection_xml ="opencv2_data/haarcascades/haarcascade_frontalface_default.xml"):
	imlist = os.listdir(img_src)
	DatasetPath = []
	for i in imlist:
	    DatasetPath.append(os.path.join(img_src, i))
	imageData = []
	num_samples = len(imlist)
	imageLabels = []
	for i in DatasetPath:
	    imgRead = io.imread(i,as_grey=asGray)
	    imageData.append(imgRead)

	label = np.ones((num_samples,),dtype = int)
	label_dict = {}
	with open(label_src) as fp:
	    for line in fp:
		    temp = line.split(' ')
		    label_dict[temp[0]] = int(float(temp[1]))
	for i in range(len(imlist)):
	    label[i]=(label_dict[imlist[i]]-1)
	imageLabels = label

	faceDetectClassifier = cv2.CascadeClassifier(face_detection_xml)

	imageDataFin = []
	for i in imageData:
	    facePoints = faceDetectClassifier.detectMultiScale(i)
	    x,y,w,h = facePoints[0]
	    cropped = i[y:y+h, x:x+w]
	    face = resize(cropped, [img_rows,img_cols])
	    #face_3d = face[:, :, None] * np.ones(3, dtype=int)[None, None, :]
	    imageDataFin.append(face)


	# In[7]:

	c = np.array(imageDataFin)

	X_train, X_test, y_train, y_test = train_test_split(np.array(imageDataFin),np.array(imageLabels), train_size=0.9, random_state = 20)
	X_train = np.array(X_train)
	X_test = np.array(X_test)

	nb_classes = out_put_classes
	y_train = np.array(y_train)
	y_test = np.array(y_test)


	# In[14]:

	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	X_train = copy_to_n_channels(X_train,img_chs)
	X_test = copy_to_n_channels(X_test,img_chs)
	# transpose according to dimension ordering
	if K.image_dim_ordering()=='th':
	    print "\nfound the dimension ordering as th"
	    X_train = X_train.transpose(0,3,1,2)
	    X_test = X_test.transpose(0,3,1,2)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= 255
	X_test /= 255

	name_gray = False
	if (asGray):
		name_gray = True
	info = str(name_gray) + "_" + str(img_rows) + "_" + str(img_cols)+ "_" + str(img_chs)
	name_X_train = ("X_train_"+info)
	name_X_test = ("X_test_"+info)
	name_Y_train = ("Y_train_"+info)
	name_Y_test = ("Y_test_"+info)
	# In[18]:

	print("Training matrix shape", X_train.shape)
	print("Testing matrix shape", X_test.shape)

	np.save(name_X_train, X_train)
	np.save(name_X_test, X_test)
	np.save(name_Y_train, Y_train)
	np.save(name_Y_test, Y_test)
	print("\nSaving the processed and loaded data as .npy files")
