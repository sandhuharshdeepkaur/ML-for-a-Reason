import os
os.chdir("D:\\trainings\\computer_vision")

#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import the necessary packages specific to Computer vision
import cv2

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())

#%% Basic image Arithmetic using OpenCv

# Thumb rule: Use OpenCv function and avoid Numpy operations as both has different meaning
#Example: use cv2.add(im1, im2) instead of im1 + im2

my_image_color_1 = cv2.imread('./image_data/forest.jpg')
my_image_color_2 = cv2.imread('./image_data/greenscreen.jpg')

# get the minimum parameters on which resize can happen
my_image_color_1.shape, my_image_color_2.shape # h = 375, w = 500
new_h = min(my_image_color_1.shape[0], my_image_color_2.shape[0])
new_w = min(my_image_color_1.shape[1], my_image_color_2.shape[1])

#resize on same scale
my_image_color_1 = cv2.resize(my_image_color_1, (new_w, new_h), cv2.INTER_AREA)
my_image_color_2 = cv2.resize(my_image_color_2, (new_w, new_h), cv2.INTER_AREA)

# Simple addition
image_added = cv2.add(my_image_color_1,my_image_color_2)
cv2.imshow('image_added',image_added); cv2.waitKey(0); cv2.destroyAllWindows()
# Color changed

# Weighted addition
image_weighted = cv2.addWeighted(my_image_color_1,0.4,my_image_color_2,0.6,0)
cv2.imshow('image_weighted',image_weighted); cv2.waitKey(0); cv2.destroyAllWindows()

#%% Put one image on another background
# Load two images
my_image_color_1 = cv2.imread('./image_data/forest.jpg')
my_image_color_2 = cv2.imread('./image_data/123.PNG')

# get the minimum parameters on which resize can happen
my_image_color_1.shape, my_image_color_2.shape # h = 375, w = 500
new_h = min(my_image_color_1.shape[0], my_image_color_2.shape[0])
new_w = min(my_image_color_1.shape[1], my_image_color_2.shape[1])

#resize on same scale
my_image_color_1 = cv2.resize(my_image_color_1, (new_w, new_h), cv2.INTER_AREA)
my_image_color_2 = cv2.resize(my_image_color_2, (new_w, new_h), cv2.INTER_AREA)

# Now create a mask of second image and create its inverse mask
my_image_color_2gray = cv2.cvtColor(my_image_color_2,cv2.COLOR_BGR2GRAY)
cv2.imshow('image_weighted',my_image_color_2gray); cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imwrite('./image_output/1.my_image_color_2gray.png',my_image_color_2gray)

# add a threshold
ret, mask = cv2.threshold(my_image_color_2gray, 20, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('image_weighted',mask); cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imwrite('./image_output/2.mask.png',mask)

mask_inv = cv2.bitwise_not(mask)
cv2.imshow('image_weighted',mask_inv); cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imwrite('./image_output/3.mask_inv.png',mask_inv)

# Now black-out the area of second image in my_image_color_1
my_image_color_1_bg = cv2.bitwise_and(my_image_color_1,my_image_color_1,mask = mask_inv)
cv2.imshow('image_weighted',my_image_color_1_bg); cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imwrite('./image_output/4.my_image_color_1_bg.png',my_image_color_1_bg)

# Take only region of second image
my_image_color_2_fg = cv2.bitwise_and(my_image_color_2,my_image_color_2,mask = mask)
cv2.imshow('image_weighted',my_image_color_2_fg); cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imwrite('./image_output/5.my_image_color_2_fg.png',my_image_color_2_fg)

dst = cv2.add(my_image_color_1_bg,my_image_color_2_fg)
cv2.imshow('image_weighted',dst); cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imwrite('./image_output/6.dst.png',dst)

my_image_color_1[:, :] = dst
cv2.imshow('image_weighted',my_image_color_1); cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imwrite('./image_output/7.my_image_color_1_final.png',my_image_color_1)
#%% Put green image on forest background: Usage of changing color by HSV
# Load two images
my_image_color_1 = cv2.imread('./image_data/forest.jpg')
my_image_color_2 = cv2.imread('./image_data/greenscreen.jpg')

# get the minimum parameters on which resize can happen
my_image_color_1.shape, my_image_color_2.shape # h = 375, w = 500
new_h = min(my_image_color_1.shape[0], my_image_color_2.shape[0])
new_w = min(my_image_color_1.shape[1], my_image_color_2.shape[1])

#resize on same scale
my_image_color_1 = cv2.resize(my_image_color_1, (new_w, new_h), cv2.INTER_AREA)
my_image_color_2 = cv2.resize(my_image_color_2, (new_w, new_h), cv2.INTER_AREA)

# First let us see various color space conversions in cv2
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
len(flags)
#flags2 = [i for i in flags if 'RGB' in i and 'BGR' in i]
flags2 = [i for i in flags if 'BGR' in i]
len(flags2)
flags3 = [i for i in flags2 if 'HSV' in i]
len(flags3)
flags3

# Change green to

# Convert BGR to HSV
my_image_color_2hsv = cv2.cvtColor(my_image_color_2,cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
#green = np.uint8([[[0,255,0 ]]])
#hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
#hsv_green # [[[ 60 255 255]]] Now use 60-30 & 60+30
# Note: for red color see https://stackoverflow.com/questions/30331944/finding-red-color-using-python-opencv

lower_range = np.array([30,0,0]); upper_range = np.array([90,255,255])

# Threshold the HSV image to get only required colors
mask = cv2.inRange(my_image_color_2hsv, lower_range, upper_range)
my_image_color_2hsv[mask > 0] = 0
my_image_color_2_withoutgreen = cv2.cvtColor(my_image_color_2hsv,cv2.COLOR_HSV2BGR)
cv2.imshow('image_weighted',my_image_color_2_withoutgreen); cv2.waitKey(0); cv2.destroyAllWindows()

# Now create a mask of second image and create its inverse mask
my_image_color_2gray = cv2.cvtColor(my_image_color_2_withoutgreen,cv2.COLOR_BGR2GRAY)
cv2.imshow('image_weighted',my_image_color_2gray); cv2.waitKey(0); cv2.destroyAllWindows()

# Threshold the image to convert boolean image. 20 has been taken as per hist (shown in
#contour section)
ret, im_th = cv2.threshold(my_image_color_2gray, 20, 255, cv2.THRESH_BINARY)
im_th = np.uint8(im_th)
cv2.imshow('image_weighted',im_th); cv2.waitKey(0); cv2.destroyAllWindows()

# Define kernel. May choose 3 or 5 or 7 too
kernel = np.ones((5,5),np.uint8)

# add a threshold
#ret, mask = cv2.threshold(my_image_color_2gray, 120, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow('image_weighted',mask); cv2.waitKey(0); cv2.destroyAllWindows()
# Did not work

# Run erosion and dilation (Morphological operations discussed in filtering section)
my_image_erosion = cv2.erode(im_th,kernel,iterations = 1)
cv2.imshow('image_weighted',my_image_erosion); cv2.waitKey(0); cv2.destroyAllWindows()

my_image_dilation = cv2.dilate(im_th,kernel,iterations = 3)
cv2.imshow('image_weighted',my_image_dilation); cv2.waitKey(0); cv2.destroyAllWindows()

mask = my_image_dilation
cv2.imshow('image_weighted',mask); cv2.waitKey(0); cv2.destroyAllWindows()

mask_inv = cv2.bitwise_not(mask)
cv2.imshow('image_weighted',mask_inv); cv2.waitKey(0); cv2.destroyAllWindows()

# Now black-out the area of second image in my_image_color_1
my_image_color_1_bg = cv2.bitwise_and(my_image_color_1,my_image_color_1,mask = mask_inv)
cv2.imshow('image_weighted',my_image_color_1_bg); cv2.waitKey(0); cv2.destroyAllWindows()

# Take only region of second image
my_image_color_2_fg = cv2.bitwise_and(my_image_color_2,my_image_color_2,mask = mask)
cv2.imshow('image_weighted',my_image_color_2_fg); cv2.waitKey(0); cv2.destroyAllWindows()

dst = cv2.add(my_image_color_1_bg,my_image_color_2_fg)
cv2.imshow('image_weighted',dst); cv2.waitKey(0); cv2.destroyAllWindows()

my_image_color_1[:, :] = dst
cv2.imshow('image_weighted',my_image_color_1); cv2.waitKey(0); cv2.destroyAllWindows()

# CW: You can see dark line. Correct that and post solution in discussion forum
#%% GrabCut Foreground
#http://en.wikipedia.org/wiki/GrabCut
#http://dl.acm.org/citation.cfm?id=1015720

# Why GrabCut and why not simple rectangle?
# Let us start with rectangle then answer above question
img = cv2.imread('./image_data/elephant.png')

# Got this data from mouse click events
rect = (59, 70,264-59, 279-70)#x,y,w,h

# Clrate mask similar to image
mask = np.zeros(img.shape[:2],np.uint8)

# Make rectangle area to 1
mask[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]] = 1

# Do 'AND ' operation so that all part is black except rectangle part
img = cv2.bitwise_and(img,img,mask = mask) # img = img*mask[:,:,np.newaxis]
cv2.imshow('image',img); cv2.waitKey(0); cv2.destroyAllWindows()
# If you see. it is now what we are looking for

# Let us apply GrabCut
img = cv2.imread('./image_data/elephant.png')

# Create mask, background and foreground models
mask = np.zeros(img.shape[:2],np.uint8)

#These are arrays used by the algorithm internally.
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# Run Grabcut
mask, bgdModel, fgdModel = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
np.unique(mask) # [0, 2, 3]

#the mask is changed so that all 0 and 2 pixels are converted to the background, where the
#1 and 3 pixels are now the foreground
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = cv2.bitwise_and(img,img,mask = mask2)
cv2.imshow('image',img); cv2.waitKey(0); cv2.destroyAllWindows()
# Did you see the difference

del(img, mask, mask2, bgdModel, fgdModel,rect )

#CW: Study Watershed algorithm

#%% Identify color
# Let us start with rectangle then answer above question
img = cv2.imread('./image_data/balloon.jpg')

# Got this data from mouse click events
rect = (100, 100,50, 50)#x,y,w,h

#get ROI
roi = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
cv2.imshow('image', roi); cv2.waitKey(0); cv2.destroyAllWindows()

# get mean values
mean_blue = int(np.mean(roi[:,:,0]))
mean_green = int(np.mean(roi[:,:,1]))
mean_red = int(np.mean(roi[:,:,2]))
print("R: %s, G: %s B: %s" %(mean_red, mean_green, mean_blue))

# CW: Use HSV way to get the color
