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
#%% Blurring: a low pass filter
# Revisist the filter ppt

#Source: Kernels are taken from book "OpenCV 3.x with Python By Example by Gabriel Garrido and Prateek Joshi"

#low pass filter: It allows low frequencies, and blocks higher frequencies. Here frequency
#mean is the rate of change of pixel values - plain areas would be low-frequency and the sharp
#edges would be high-frequency because the pixel values change rapidly.
#Usage: It smooth the edges.

from skimage.io import imread

# Read image
my_image_gray = np.array(imread("./image_data/wint_sky.gif", as_grey=True))
cv2_show_fullscr('Original', my_image_gray)

# Define kernel
kernel_3 = np.ones((3,3), np.float32) / 9.0 # Note: Normalization

# Apply convolution
image_kernel_3 = cv2.filter2D(my_image_gray, -1, # Output is same size as input
                      kernel_3)
list_my_image = np.hstack([my_image_gray, image_kernel_3])
cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()

#CW: Make kernel of 5x5 and see the impact

#Inbuilt blur functions
#cv2.blur: Same as above
#cv2.GaussianBlur: Uses gaussian kernel cv2.getGaussianKernel. The Gaussian filter just looks
#at the immediate neighborhood and averages the pixel values using a Gaussian kernel,
#cv2.medianBlur:  Used in removing salt-and-pepper noise

#resize just to view result properly
my_image_gray = cv2.resize(my_image_gray, (300, 300), cv2.INTER_AREA)

# Apply inbuilt blur functions
image_blur = cv2.blur(my_image_gray,(3,3))
image_gaussblur = cv2.GaussianBlur(my_image_gray,(3,3),0)
image_medianblur = cv2.medianBlur(my_image_gray,3)

list_my_image = np.hstack([my_image_gray, image_blur, image_gaussblur, image_medianblur])
cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()


# Directional blurring
my_image_color = cv2.imread('./image_data/elephant.png')
my_image_gray = cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY)

# Horizontal
kernel_hor = np.array([[0,0,0], [1,1,1],[0,0,0]], np.float32) / 3.0

# Apply convolution
image_kernel_hor = cv2.filter2D(my_image_color, -1, kernel_hor)
list_my_image = np.hstack([my_image_color, image_kernel_hor])
cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()
# i. For larger effect, use larger kernel
# ii. Similary, direction can be changed
# CW: Try with kernel_ver = np.array([[0,1,-1,0], [1,3,-3,-1],[1,3,-3,-1],[0,1,-1,0]], np.float32)

# Sharpening
kernel_sharp_1 = np.array([[-1,-1,-1], [-1,9,-1],[-1,-1,-1]], np.float32)
kernel_sharp_2 = np.array([[1,1,1], [1,-7,1],[1,1,1]], np.float32)

# Apply convolution
image_kernel_sharp_1 = cv2.filter2D(my_image_color, -1, kernel_sharp_1)
image_kernel_sharp_2 = cv2.filter2D(my_image_color, -1, kernel_sharp_2)
list_my_image = np.hstack([my_image_color, image_kernel_sharp_1, image_kernel_sharp_2])
cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()
# Looks Unnaturally enhanced - hence may not be practically viable

#Embossing: Replace each pixel with a shadow or a highlight depending upon direction
# let us take 3 different kernel for embossing
kernel_emboss_1 = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
kernel_emboss_2 = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
kernel_emboss_3 = np.array([[1,0,0],[0,0,0],[0,0,-1]])

# Apply convolution
image_kernel_emboss_1 = cv2.filter2D(my_image_gray, -1, kernel_emboss_1) + 128 # the offset to produce the shadow
image_kernel_emboss_2 = cv2.filter2D(my_image_gray, -1, kernel_emboss_2) + 128
image_kernel_emboss_3 = cv2.filter2D(my_image_gray, -1, kernel_emboss_3) + 128
list_my_image = np.hstack([my_image_gray, image_kernel_emboss_1, image_kernel_emboss_2, image_kernel_emboss_3])
cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()


#High pass filtering: It allows high-frequency content to pass through and blocks
#the low-frequency content.

#Edge detection: It is high pass filter for identifying high-frequency content.
#We can use kernel or inbuilt function cv2.Sobel as below

my_image_color = cv2.imread('./image_data/balloon.jpg')
my_image_gray = cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY)

#resize just to view result properly
my_image_gray = cv2.resize(my_image_gray, (400, 400), cv2.INTER_AREA)
cv2.imshow("All", my_image_gray); cv2.waitKey(0); cv2.destroyAllWindows()

# Do convolution using inbuilt function
image_sobel_hor = cv2.Sobel(my_image_gray, cv2.CV_64F, # depth of cv2.CV_64F
                            1,  # Horizontal
                            0,  # NO vertical
                            ksize=5) # Kernel size can be: 1,3,5 or 7.

image_sobel_ver = cv2.Sobel(my_image_gray, cv2.CV_64F, 0, 1, ksize=5)
image_sobel_both = cv2.Sobel(my_image_gray, cv2.CV_64F, 1, 1, ksize=5)

list_my_image = np.hstack([image_sobel_hor, image_sobel_ver,image_sobel_both])
cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()
#Practically, both directional edge detection is required and Sobel is not helping and hence

#try Laplacian as follows
image_laplacian_both = cv2.Laplacian(my_image_gray, cv2.CV_64F)

list_my_image = np.hstack([image_sobel_both, image_laplacian_both])
cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()
# It a bit better and can be imroved by following
#CW: Try with 'arrows.png'

image_canny_both = cv2.Canny(my_image_gray, 60, # low threshold
                  220) # high threshold: beyond it marked as a strong edge

list_my_image = np.hstack([image_sobel_both, image_laplacian_both, image_canny_both])
cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()
# A bit better :-)

# CW: Practice with different value of thresholds

#Contrast enhancement by Histogram equalization (theory on ppt)
my_image_color = cv2.imread('./image_data/photo_1.jpg')
my_image_gray = cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY)

#resize just to view result properly
my_image_gray = cv2.resize(my_image_gray, (400, 400), cv2.INTER_AREA)

# equalize the histogram of the input image
image_histogram_equalization = cv2.equalizeHist(my_image_gray)

list_my_image = np.hstack([my_image_gray, image_histogram_equalization])
cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()

# CW: To see the outcome of HE, Blur image first and then apply HE

# Note: For color images, convert to YUV color space {Luminance (Y) and two
#Chrominance (UV) components}. Now equalize the Y-channel and combine it with the
#other two channels for result

#%% Morphological operations: A set of operations that process images according to their shapes

#Erosion (think like soil erosion): The main aim is to grow dark regions by considering 1 only if
#all the pixels under the kernel is 1, otherwise it is eroded (made to zero)
#It is useful for removing small white noises and detach two connected objects etc.

#Dilation (just opposite of erosion) : If atleast one pixel under the kernel is '1' then keep 1.
#So it increases the white region (foreground). Normally, in cases like noise removal, erosion is
#followed by dilation. Because, erosion removes white noises, but it also shrinks our object.
#So we dilate it. Since noise is gone, they won't come back, but our object area increases.
#It is also useful in joining broken parts of an object.

# Read image, make gray and BW
my_image_color = cv2.imread('./image_data/123.png')
my_image_gray = cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY)

#resize just to view result properly
my_image_gray = cv2.resize(my_image_gray, (200, 200), cv2.INTER_AREA)
cv2_show_fullscr('gray image', my_image_gray)

# Threshold the image to convert boolean image. 200 has been taken as per hist (shown in contour section)
ret, im_th = cv2.threshold(my_image_gray, 200, 255, cv2.THRESH_BINARY_INV)
im_th = np.uint8(im_th)
cv2_show_fullscr('thres image', im_th)

# Define kernel. May choose 3 or 5 or 7 too
kernel = np.ones((5,5),np.uint8)

# Run erosion and dilation
my_image_erosion = cv2.erode(im_th,kernel,iterations = 1)
my_image_dilation = cv2.dilate(im_th,kernel,iterations = 1)

list_my_image = np.hstack([im_th, my_image_erosion, my_image_dilation])
cv2.imshow("All", list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()

#CW1: Add few white dots and run above to see noise cleaning process
#CW2: Dilate the erroded image. It should give 'im_th' to prove that both are opposite to each other
#CW3: DO for 3 iterations and see impact
