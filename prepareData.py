import numpy as np
import cv2
import glob
import itertools
import random
from keras.utils import Sequence

img_height = 384
img_width = 512

def img_channels(img, width, height, n):
    #img = cv2.resize(img, (width , height),interpolation = cv2.INTER_NEAREST)
    if n == 14:
	# R,G,B Channels
        IB = img[:, :, 0]/255.0 
        IG = img[:, :, 1]/255.0 
        IR = img[:, :, 2]/255.0
	# Excess Green
        IExG = 2*IG - IR - IB
        IExG = IExG.astype(np.uint8)
	# Excess Red
        IExR = 1.4*IR - IG
	# Color Index of Vegetation Extraction
        ICIVE = 0.881*IG - 0.441*IR - 0.385*IB - 18.78745
	# Normalized Difference Index
        INDI = (IG - IR)/(IG + IR)
	# HSV Color Space
        image_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        IHUE,ISAT,IVAL = cv2.split(image_hsv)
    # Laplacian on IExG
        IExG_lap = cv2.Laplacian(IExG,cv2.CV_32F)
    # Sobel in x & y directions on IExG
        IExG_sobelx = cv2.Sobel(IExG,cv2.CV_32F,1,0)
        IExG_sobely = cv2.Sobel(IExG,cv2.CV_32F,0,1)
    # Canny Edge Detector on IExG 
        IExG_canny = cv2.Canny(IExG,100,100)
        img = np.dstack((IB,IG,IR,IExG,IExR,ICIVE,INDI,IHUE,ISAT,IVAL,IExG_sobelx,IExG_sobely,IExG_lap,IExG_canny))
    else:
        img = img/255.0
    return img
        

# background (black), # crop (green), # weed (red) 
color_mapper = {(0, 0, 0) : 0 , (0, 255, 0) : 1 , (0, 0, 255) : 2}

def segmented_to_labelImg(seg_img):
    result = np.zeros((seg_img.shape[0], seg_img.shape[1], 3), dtype=np.uint8)
    for i in range(0, seg_img.shape[0]):
        for j in range(0, seg_img.shape[1]):
            key = (seg_img[i, j, 0], seg_img[i, j, 1], seg_img[i, j, 2]) # Checking pixel by pixel
            result[i, j] = color_mapper.get(key, 0) # default value if key was not found is 0
    return result

def getImageArray(path, width=512, height=384, nChannels = 14, imgNorm="divide"):
	try:
		img = cv2.imread(path)
		if imgNorm == "sub_and_divide":
			img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
		elif imgNorm == "sub_mean":
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32)
			img[:,:,0] -= 103.939
			img[:,:,1] -= 116.779
			img[:,:,2] -= 123.68
		elif imgNorm == "divide":
			img = img_channels(img, width, height, nChannels)
		img = np.rollaxis(img, 2, 0)
		return img
	except Exception as e:
		print(path , e)
		img = np.zeros((height, width, 3))
		img = np.rollaxis(img, 2, 0)
		return img

def getSegmentationArray(path, nClasses=3, width=512, height=384):
	seg_labels = np.zeros((height, width, nClasses))
	try:
		img = cv2.imread(path)
		#img = cv2.resize(img, (width , height),interpolation = cv2.INTER_NEAREST)
		img = img[:, : , 0]

		for c in range(nClasses):
			seg_labels[:, :, c] = (img == c).astype(int)

	except Exception as e:
		print(e)
		
	seg_labels = np.reshape(seg_labels, (width*height, nClasses))
	return seg_labels

def displayImage(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

class Data_Generator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([getImageArray(file_name) for file_name in batch_x]), np.array([getSegmentationArray(file_name) for file_name in batch_y])


#For pixel - class labelling
# =============================================================================
# filenames = glob.glob("groundtruth/*.png")
# filenames.sort()
# 
# start = time.time()   
# 
# for i,filename in enumerate(filenames):
#     img = cv2.imread(filename)
#     img = cv2.resize(img,(img_width,img_height),interpolation = cv2.INTER_NEAREST)
#     j = segmented_to_labelImg(img)
#     path = "annotation\\"+filename.split('\\')[-1]
#     cv2.imwrite(path,j)       
# 
# end = time.time()      
# print("Time Taken : ",end-start)   
# =============================================================================
# =============================================================================
# filenames = glob.glob("rgb_resized/*.png")
# filenames.sort()  # make sure that the filenames have a fixed order before shuffling
# random.seed(230)
# random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)
# 
# split_1 = int(0.8 * len(filenames))
# split_2 = int(0.9 * len(filenames))
# train_filenames = filenames[:split_1]
# validation_filenames = filenames[split_1:split_2]
# test_filenames = filenames[split_2:]    
# gt_training = ["groundtruth_resized\\"+filename.split('\\')[-1] for filename in train_filenames]
# gt_validation = ["groundtruth_resized\\"+filename.split('\\')[-1] for filename in validation_filenames]
# gt_testing = ["groundtruth_resized\\"+filename.split('\\')[-1] for filename in test_filenames]
#   
# 
#     
# def writeImage(img1,img2,num):
#     path1 = "augment_test\\"+str(num).zfill(3)+"_image.png"
#     path2 = "augment_test_annotations\\"+str(num).zfill(3)+"_image.png"
#     cv2.imwrite(path1,img1)
#     cv2.imwrite(path2,img2)
# 
# 
# filenames = glob.glob("groundtruth/*.png")
# filenames.sort()
# for filename in filenames:
#     img = cv2.imread(filename)
#     img = cv2.resize(img,(img_width,img_height),interpolation = cv2.INTER_NEAREST)
#     path = "groundtruth_resized\\"+filename.split('\\')[-1]
#     cv2.imwrite(path,img)
# =============================================================================
# =============================================================================
# weights = [0,0,0]
# for i in gt_training:
#     y = getSegmentationArray(i)
#     sumarr = y.sum(axis=0)
#     weights[0] += sumarr[0]
#     weights[1] += sumarr[1]
#     weights[2] += sumarr[2]    
# =============================================================================
    
    
