import os
os.chdir("D:\\trainings\\computer_vision")
import numpy as np
import scipy.ndimage
from scipy.misc.pilutil import Image
import scipy.misc as mi
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())

#%% Filter: Basic definition on ppt

#Filter for removing noise

# Mean Filter
# opening the image and converting it to grayscale
a = Image.open('./image_data/elephant_marked.png').convert('L')
plt.imshow(np.array(a)); plt.show()

# initializing the filter of size 5 by 5. the filter is divided by 25 for normalization
k = np.ones((5,5))/25
np.sum(k)

# performing convolution
b = scipy.ndimage.filters.convolve(a, k)
plt.imshow(b)
plt.show()

# The mean filter effectively removed the noise but in the process blurred the image.
#Advantages of the mean lter
# Removes noise.
# Enhances the overall quality of the image, i.e. mean filter brightens an image.
#Disadvantages of the meanlter
# In the process of smoothing, the edges get blurred.
# Reduces the spatial resolution of the image.

# b is converted from an ndarray to an image
b = Image.fromarray(b)
#b.save('./image_data/elephant_marked_mean_filter.png')

# Median Filter
#The median lter is most commonly used in removing salt-and-pepper (black and white spots) noise and impulse

# performing the median filter
b = scipy.ndimage.filters.median_filter(a,size=5, footprint=None,output=None,mode='reflect',cval=0.0,origin=0)
plt.imshow(b)
plt.show()

# The median filter eciently removed the salt-and-pepper noise.

# Max Filter: This filter enhances the bright points in an image. More suitable for BW images
# performing maximum filter
b = scipy.ndimage.filters.maximum_filter(a,size=5, footprint=None,output=None,mode='reflect',cval=0.0,origin=0)
plt.imshow(b)
plt.show()

# After application of the max filter, the white pixels have grown.

# Min Filter: This filter is used to enhance the darkest points in an image
b = scipy.ndimage.filters.minimum_filter(a,size=5, footprint=None,output=None,mode='reflect',cval=0.0,origin=0)
plt.imshow(b)
plt.show()

del(a, b, k)

#Note: We will learn more techniques in this course
#%% Edge Detection :Theory on ppt

from skimage import filters, feature
#Sobel and Prewitt filter

# opening the image and converting it to grayscale
a = Image.open('./image_data/elephant.png').convert('L')

# performing Sobel filter
b = filters.sobel(a)
plt.imshow(b)
plt.show()
#Use if one required -  vsobel, hsobel

# performing prewitt filter
b1 = filters.prewitt(a)
plt.imshow(b1)
plt.show()
#Use if one required -  vprewitt and hprewitt

# performing Canny filter
b2 = np.asarray(a).astype(float)
np.min(b2), np.max(b2) #Shold be 0-1 else normalize
b2 = b2/float(np.max(b2))

b2 = feature.canny(b2, sigma=1, low_threshold = 0.1, high_threshold = 0.2)
plt.imshow(b2)
plt.show()

# Class work: Play with sigma 1-3 and see the difference

# performing laplace filter
b3 = scipy.ndimage.filters.laplace(a)
plt.imshow(b3)
plt.show()



# b is converted from an ndarray to an image
b = Image.fromarray(b)
#b.save('./image_data/elephant_sobel_filter.png')

