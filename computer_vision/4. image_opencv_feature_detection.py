import os
os.chdir("D:\\trainings\\computer_vision")

#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import the necessary packages specific to Computer vision
import cv2 # cv2.__version__

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())
#%% Theory on ppt

#%%Harris Corner Detector
# http://www.bmva.org/bmvc/1988/avc-88-023.pdf

my_image_color = cv2.imread('./image_data/arrows.png') #  elephant.png
my_image_gray = np.float32(cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY))

# cornerHarris arguments are
#img - Input image, it should be grayscale and float32 type.
#blockSize - It is the size of neighbourhood considered for corner detection
#ksize - Aperture parameter of Sobel derivative used.
#k - Harris detector free parameter in the equation.
my_image_corner_objects = cv2.cornerHarris(my_image_gray,2,3,0.04)

#result is dilated(make or become wider, larger, or more open) for marking the corners
my_image_corner_objects = cv2.dilate(my_image_corner_objects,None)

# Threshold for an optimal value, it may vary depending on the image.
my_image_color[my_image_corner_objects>0.01*my_image_corner_objects.max()] = [0,0,255] # Color BGR
my_image_corner_objects.min(), my_image_corner_objects.max()

cv2.imshow('my_image',my_image_color); cv2.waitKey(0); cv2.destroyAllWindows()

del(my_image_color, my_image_gray, my_image_corner_objects)
#%% Harris Corner with SubPixel Accuracy
#To find the corners with maximum accuracy use cv2.cornerSubPix() uses centroids (of bunch of pixels at a
#corner) to refine them. Harris corners are marked in red pixels and refined corners are marked in green pixels

my_image_color = cv2.imread('./image_data/arrows.png') #  elephant.png
my_image_gray = np.float32(cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY))

# Overall approach -> Find Harris corners and then centroids and then compare with subpixel corners

# find Harris corners
my_image_corner_objects = cv2.cornerHarris(my_image_gray,2,3,0.04)

#result is dilated(make or become wider, larger, or more open) for marking the corners
my_image_corner_objects = cv2.dilate(my_image_corner_objects,None)

# Converting to boolean image as connectedComponentsWithStats needs
ret, my_image_corner_objects = cv2.threshold(my_image_corner_objects,
        0.01*my_image_corner_objects.max(), # threshold value which is used to classify the pixel values
        255, #maxVal which represents the value to be given if pixel value is more than the threshold value
        cv2.THRESH_BINARY)
my_image_corner_objects = np.uint8(my_image_corner_objects)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(my_image_corner_objects)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(my_image_gray ,
                           np.float32(centroids), # corners
                           (5,5),#winSize. 5∗2+1×5∗2+1=11×11 earch window
                           (-1,-1), # zeroZone. -1 means auto select
                           criteria)

# Now draw them
res = np.int0(np.hstack((centroids,corners))) # np.int0 is int64 on 64 bit OS
my_image_color [res[:,1],res[:,0]]=[0,0,255]
my_image_color [res[:,3],res[:,2]] = [0,255,0]

# See the saved image after enlarging 400% to see the red and green dots
cv2.imwrite('./image_output/arrows_harris.png',my_image_color )

del(my_image_color, my_image_gray, ret, labels, stats, centroids, criteria, corners, res)

#%% Shi-Tomasi Corner Detector & Good Features to Track
# http://www.ai.mit.edu/courses/6.891/handouts/shi94good.pdf
#It is based on the Harris corner detector. The main variation is in "selection criteria" ->
#The Harris corner detector has a corner selection criteria. A score is calculated for each
#pixel, and if the score is above a certain value, the pixel is marked as a corner.

my_image_color = cv2.imread('./image_data/arrows.png')
my_image_gray = np.float32(cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY))

# call to find N strongest corners
corners = np.int0(cv2.goodFeaturesToTrack(my_image_gray,
                                  50, # number of corners you want to find
                                  0.01, # minimum quality of corner below which everyone is rejected
                                  10)) # minimum euclidean distance between corners
                                    # useHarrisDetector can also be used

# Iterate over each corner and draw circle
for i in corners:
    x,y = i.ravel()
    cv2.circle(my_image_color,(x,y),3,255,-1)

# See the saved image after enlarging to see the blue dots
cv2.imwrite('./image_output/arrows_shi_tomasi.png',my_image_color )

del(my_image_color, my_image_gray, corners)
#%% Just FYI
# Scale-invariant feature transform (SIFT): SIFT is one of the most popular algorithms in
#the entire field of computer vision. Paper is at http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf.
#SIFT is patented and it's not freely available for commercial use.
# Drawback is that it is computationally intensive

# Speeded-up robust features (SURF): It is faster than SIFT and paper availble at http://www.vision.ee.ethz.ch/~surf/eccv06.pdf
#SURF is also patented and it is not freely available for commercial use
# Not fast enough for building a real-time application on a mobile device.

# Use Features from accelerated segment test(FAST) below
#%% Features from accelerated segment test (FAST)
# It is super fast because - instead of doing expensive calculations, it does high-speed test
#to quickly determine if the current point is a potential keypoint
#Note: it is just for keypoint detection. Use other technique to compute the descriptors like
# Binary robust independent elementary features (BRIEF)

# read image
my_image_color = cv2.imread('./image_data/arrows.png')
my_image_gray = cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY)

# create fast object
fast = cv2.FastFeatureDetector_create()

#Default value
fast.getNonmaxSuppression() # Non-max suppression is a way to eliminate points that do not
                            # lie in important edges.

# Detect keypoints
keypoints = fast.detect(my_image_gray, None)
print("Number of keypoints with non max suppression:", len(keypoints))

# Draw keypoints on top of the input image
img_keypoints_with_nonmax=my_image_color.copy()
cv2.drawKeypoints(my_image_color, keypoints, img_keypoints_with_nonmax, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('FAST keypoints - with non max suppression', img_keypoints_with_nonmax)
cv2.waitKey(0); cv2.destroyAllWindows()

# Disable nonmaxSuppression
fast.setNonmaxSuppression(False)

# Detect keypoints again
keypoints = fast.detect(my_image_gray, None)

print("Total Keypoints without nonmaxSuppression:", len(keypoints))

# Draw keypoints on top of the input image
img_keypoints_without_nonmax=my_image_color.copy()
cv2.drawKeypoints(my_image_color, keypoints, img_keypoints_without_nonmax, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('FAST keypoints - without non max suppression', img_keypoints_without_nonmax)
cv2.waitKey(0); cv2.destroyAllWindows()

del(keypoints, my_image_color, my_image_gray)
#%% ORB (Oriented FAST and Rotated BRIEF) corner detectors
# Theory on ppt

# Define in one function as we will call two times for different images
def orb_corner_detectors(orb, file_image):
    # read and convert to RGB
    my_image_color = cv2.imread(file_image) #  elephant.png
    my_image_color = cv2.cvtColor(my_image_color,cv2.COLOR_BGR2RGB)

    #detect
    key_points, description = orb.detectAndCompute(my_image_color, None)

    return my_image_color, key_points, description

# Create object
orb = cv2.ORB_create()

# Call above function
my_image_color, key_points, description = orb_corner_detectors(orb, './image_data/arrows.png')

# Draw basic circles
my_image_color_basic = my_image_color
img_building_keypoints = cv2.drawKeypoints(my_image_color, key_points, my_image_color_basic)
cv2.imwrite('./image_output/arrows_orb_basic.png',my_image_color_basic)

# Draw rich circles
my_image_color_rich = my_image_color
img_building_keypoints = cv2.drawKeypoints(my_image_color, key_points, my_image_color_rich,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw circles
cv2.imwrite('./image_output/arrows_orb_rich.png',my_image_color_rich)

# Do detection for partial image
my_image_color_partial, key_points_partial, description_partial = orb_corner_detectors(orb, './image_data/arrows_partial.png')

# Brute-force matcher.
bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, # It is advisisable for orb
                          crossCheck=True) # return pairs (i,j) such that for i-th query descriptor the
                            #j-th descriptor in the matcher's collection is the nearest and vice versa

#get class after matching keypoint descriptors
matches = bf.match(description, description_partial)
matches = sorted(matches, key = lambda x: x.distance) # Sort matches by distance.  Best come first.
#for m in matches:
#    print(m.distance)

my_image_matches = cv2.drawMatches(my_image_color, key_points, my_image_color_partial, key_points_partial, matches[:10], my_image_color_partial, flags=2) # Show top 10 matches
cv2.imwrite('./image_output/arrows_orb_matches.png',my_image_matches)

#plt.figure(figsize=(16, 16))
#plt.title(type(orb))
#plt.imshow(my_image_matches); plt.show()

#Need improvement

# CW: Use basic circles image and show matches

del(matches, my_image_matches, my_image_color_partial, key_points_partial, description_partial, my_image_color, my_image_color_basic, my_image_color_rich, img_building_keypoints, orb, key_points, description)
#%% Understanding and Building an Object Detection Model from Scratch in Python
#Using retinanet -> https://www.analyticsvidhya.com/blog/2018/06/understanding-building-object-detection-model-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
# Using retinanet https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
#and  ImageAI pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.1/imageai-2.0.1-py3-none-any.whl
