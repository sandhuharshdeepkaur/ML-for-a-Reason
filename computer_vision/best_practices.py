import os
os.chdir("D:\\ShivPersonal\\Analytics\\Practice\\Python\\Packages\\tensorflow\\computer_vision")

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())

#%% Best practices
#Notes on array order: Although the labeling of the axes seems arbitrary, it can have a
#significant effect on speed of operations. This is because modern processors never retrieve
#just one item from memory, but rather a whole chunk of adjacent items. (This is called
#prefetching.) Therefore, processing elements that are next to each other in memory is faster
#than processing them in a different order, even if the number of operations is the same:

from skimage.io import imread

def in_order_multiply(arr, scalar):
    for plane in list(range(arr.shape[0])):
        arr[plane, :, :] *= scalar

def out_of_order_multiply(arr, scalar):
    for plane in list(range(arr.shape[2])):
        arr[:, :, plane] *= scalar


import time
im3d = np.random.rand(100, 1024, 1024)
t0 = time.time(); x = in_order_multiply(im3d, 5); t1 = time.time()
print("%.2f seconds" % (t1 - t0))  #0.06 seconds
im3d_t = np.transpose(im3d).copy() # place "planes" dimension at end
im3d_t.shape #(1024, 1024, 100)

s0 = time.time(); x = out_of_order_multiply(im3d, 5); s1 = time.time()
print("%.2f seconds" % (s1 - s0))  # 1.13 seconds
print("Speedup: %.1fx" % ((s1 - s0) / (t1 - t0)))  # Speedup: 18 x

#When the dimension you are iterating over is even larger, the speedup is even more dramatic. It is
#worth thinking about this data locality when writing algorithms. In particular, know that
#scikit-image uses C-contiguous arrays unless otherwise specified, so one should iterate
#along the last/rightmost dimension in the innermost loop of the computation.

#A note on time: Although scikit-image does not currently (0.11) provide functions to work specifically
#with time-varying 3D data, our compatibility with numpy arrays allows us to work quite naturally with
# a 5D array of the shape (t, pln, row, col, ch):
"""
for timepoint in image5d:
    # each timepoint is a 3D multichannel image
    do_something_with(timepoint)
"""
