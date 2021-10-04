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
#%% Contours
#Contours is a curve joining all the continuous points (along the boundary), having same color
#or intensity.
#A useful tool for shape analysis and object detection and recognition.

#Prerequisite:
#Need Binary image
#findContours function modifies the source image so give copy
#Need white object from black background

# Read and process
my_image_color = cv2.imread('./image_data/arrows.png')
my_image_gray = cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY)
cv2_show_fullscr('grey image', my_image_gray)
# If you see image is in black color although contours need it in white color

#Let us see the histogram to get cutoff value
hist = cv2.calcHist([my_image_gray],[0], # channels
                    None,# mask - NOne means full image
                    [256], # bin count - for full scale [256]
                    [0,256]) # ranges - [0,256]
plt.plot(hist,color = 'b')
plt.show()
# Almost binary. Let us choose cutoff 200

# Threshold the image to convert boolean image
ret, im_th = cv2.threshold(my_image_gray, 200, 255, cv2.THRESH_BINARY_INV)
im_th = np.uint8(im_th)
cv2_show_fullscr('thres image', im_th)

# Find contours
im2, contours, hierarchy = cv2.findContours(im_th,cv2.RETR_CCOMP, #RETR_TREE related to hierarchy list. Not covering right now
                           cv2.CHAIN_APPROX_SIMPLE) #cv2.CHAIN_APPROX_NONE, all the boundary points
#are stored. cv2.CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour, thereby
#saving memory.

# Let us draw the contours
cv2.drawContours(my_image_color, contours, -1, # all else provide index of contours
                 (0,255,0), # color - green
                 1) # width
cv2_show_fullscr('contours image', my_image_color)

# Want to show you duplicate and hence making copy of original
my_image_color_org = my_image_color.copy()
index = 0 # Image index just for readability
#draw Straight Bounding Rectangle
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(my_image_color,(x,y),(x+w,y+h),(0,0, 255),1)
    cv2.putText(my_image_color,str(index),(int(x+w/2),int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1,cv2.LINE_AA)
    index = index + 1
# end of for contour in contours

cv2_show_fullscr('contours with Straight Bounding Rectangle image', my_image_color)

my_image_color = my_image_color_org.copy()

#remove duplicates
contours = cv2_get_unique_contours(my_image_color, contours)

index = 0
#Again draw Straight Bounding Rectangle
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(my_image_color,(x,y),(x+w,y+h),(0,0, 255),1)
    cv2.putText(my_image_color,str(index), (int(x+w/2),int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1,cv2.LINE_AA)
    index = index + 1
    # str(index) + ', T' if cv2.isContourConvex(contour) else ', F'
# end of for contour in contours

cv2_show_fullscr('contours with Straight Bounding Rectangle image', my_image_color)

#draw Rotated Rectangle
for contour in contours:
    rect = cv2.minAreaRect(contour)
    box = np.int0(cv2.boxPoints(rect)) # np.int0 is int64 on 64 bit OS
    cv2.drawContours(my_image_color,[box],0,(255,0,0),1)
# end of for contour in contours

cv2_show_fullscr('contours with Rotated Rectangle image', my_image_color)
#Note: Enclosing by Circle, Ellipse or line can be done in the similar way

## Image matching
#Hu-Moments are seven moments invariant to translation, rotation and scale.
#Seventh one is skew-invariant. Those values can be found using cv.HuMoments() function.
#https://en.wikipedia.org/wiki/Image_moment
# Following usages Hu-Moments

my_image_color = my_image_color_org.copy()
index = 0; index_test = 7 # upper arrow
#Again draw Straight Bounding Rectangle and score of matching
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(my_image_color,(x,y),(x+w,y+h),(0,0, 255),1)
    cv2.putText(my_image_color,str(index) + ', '+str(round(cv2.matchShapes(contours[index_test],contour,1,0.0),2)), (int(x+w/2),int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1,cv2.LINE_AA)
    # The smaller score is better
    index = index + 1
# end of for contour in contours
cv2_show_fullscr('contours with Rotated Rectangle image', my_image_color)
#Notice the match score, many are identified near to each other

# Let us try the count of vertices
my_image_color = my_image_color_org.copy()
index = 0; index_test = 7 # upper arrow
#Again draw Straight Bounding Rectangle and count of vertices
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(my_image_color,(x,y),(x+w,y+h),(0,0, 255),1)

    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

    cv2.putText(my_image_color,str(index) + ', '+ str(len(approx)), (int(x+w/2),int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1,cv2.LINE_AA)
    index = index + 1
# end of for contour in contours

cv2_show_fullscr('contours with count of vertices', my_image_color)
# Almost there with one exception.
# CW: Try combination of Match score and count of vertices

#Till now, shape was known and now let us try to cluster all shapes in 2 cluster
#using KMeans on data 'Solidity factor':the area of shape/the area of convex hull (definition on ppt)

my_image_color = my_image_color_org.copy()

# Calculate 'Solidity factor' and list of all contours having non zero hull areas
solidity_values = []; contours_hull_areas = []
# Compute solidity factors of all the contours
for contour in contours:
    area_contour = cv2.contourArea(contour)
    area_hull = cv2.contourArea(cv2.convexHull(contour))
    if area_hull > 0.0:
        solidity_values.append(float(area_contour)/area_hull)
        contours_hull_areas.append(contour)

    del(area_contour, area_hull)
# end of for contour in contours:

# Change shape as required by KMeans
solidity_values = np.array(solidity_values).reshape((len(solidity_values),1)).astype('float32')
solidity_values.shape # (11, 1)

# Clustering using KMeans

#algorithm termination criteria, that is, the maximum number of iterations
#and/or the desired accuracy.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

compactness, labels, centers = cv2.kmeans(solidity_values, 2, # Number fo cluster
                                          None, # cluster indices for every sample
                                          criteria, # Explianed above
                                          10, # number of times the algorithm is executed using different initial labellings
                                          cv2.KMEANS_RANDOM_CENTERS)

#Now
for contour, label in zip(contours_hull_areas, labels):
    x,y,w,h = cv2.boundingRect(contour)
    if label[0] == 0:
        cv2.rectangle(my_image_color,(x,y),(x+w,y+h),(255,0, 0),1)
    elif label[0] == 1:
        cv2.rectangle(my_image_color,(x,y),(x+w,y+h),(0, 255,0),1)

cv2_show_fullscr('contours with count of vertices', my_image_color)

# CW: Make cluster count to 3 and draw

# cleaning
del(contours_hull_areas, compactness, labels, centers, criteria, solidity_values)

#Convexity defect: Any deviation of the object from convex hull
my_image_color = my_image_color_org.copy()
for contour in contours:
    convex_hull = cv2.convexHull(contour,returnPoints = False) # False while finding convex hull, in order to find convexity defects
    convexity_defects = cv2.convexityDefects(contour,convex_hull)
    if convexity_defects is not None:
        for i in range(convexity_defects.shape[0]):
            s,e,f,d = convexity_defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            cv2.line(my_image_color,start,end,[255,0,0],1)
            cv2.circle(my_image_color,far,5,[0,0,255],-1)
        #end of for i in range(defects.shape[0]):
    # end of if convexity_defects is not None:

cv2_show_fullscr('contours with convexity defects', my_image_color)

#%% Contours: Few Miscellaneous functions

#Point Polygon Test: It finds the shortest distance between a point in the image
#and a contour. It returns the distance which is negative when point is outside
#the contour, positive when point is inside and zero if point is on the contour.
#cv2.pointPolygonTest(contour,(50,50),True)

# cv2.contourArea(contour)

#cv2.arcLength

# Detail list is at https://docs.opencv.org/3.3.0/d3/dc0/group__imgproc__shape.html

