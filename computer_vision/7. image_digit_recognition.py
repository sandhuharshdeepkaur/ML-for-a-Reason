import os
os.chdir("D:\\trainings\\computer_vision")

#%matplotlib inline
import os
import sys
import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

# import the necessary packages specific to Computer vision
import cv2
from sklearn.externals import joblib
from skimage.feature import hog

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())

#%% Source reference: http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html

################### Get data to Build the model #####################
# Import the modules
from mnist import MNIST

# Load the dataset
mnist_data = MNIST('./image_data/mnist_yann_lecun'); mnist_data.gz = True
images, labels = mnist_data.load_training()

# Extract the features and labels
ar_features = np.array(images, 'int16')
ar_labels = np.array(labels, 'int')
ar_features.shape, ar_labels.shape # (60000, 784), (60000,)

#View images
# View one image of particular index
index = 1
x_image = np.reshape(ar_features[index], [28, 28])
plt.imshow(x_image); plt.show()

# see the label
ar_labels[index]

#View the first 10 images and the class name
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    x_image = np.reshape(ar_features[i], [28, 28])
    plt.imshow(x_image, cmap=plt.cm.binary)
    plt.xlabel(ar_labels[i])
    # end of 'for'

# clean up
del(mnist_data, images, labels, index, x_image)

# Extract the hog features
list_hog_fd = []
for feature in ar_features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64'); del(list_hog_fd)
hog_features.shape

print("Count of digits in dataset")
unique, counts = np.unique(ar_labels, return_counts=True)
dict(zip(unique, counts))
# You can see the data is balanced

#%% LinearSVC
from sklearn.svm import LinearSVC

# Create an linear SVM object
clf = LinearSVC(max_iter=10000)

# Perform the training
clf.fit(hog_features, ar_labels)

# self predict
predictions = clf.predict(hog_features)

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(ar_labels.tolist(), predictions.tolist())
confusion_matrix

#Statistics are also available as follows
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
# SVC: Overall Accuracy is  0.87 , Kappa is  0.85

# Save the classifier
joblib.dump(clf, "./model/digit_recog_linearsvms.pkl")

#%% Random forest
from sklearn.ensemble import RandomForestClassifier

# Default training param for RF
param = {'n_estimators': 1000, # The number of trees in the forest.
         'min_samples_leaf': 5, # The minimum number of samples required to be at a leaf node (external node)
         'min_samples_split': 10, # The minimum number of samples required to split an internal node (will have further splits)
         'max_depth': None, 'bootstrap': True, 'max_features': "auto", # The number of features to consider when looking for the best split
          'verbose': True, 'n_jobs': os.cpu_count()} # , 'warm_start' : True

#Build model on training data
classifier = RandomForestClassifier(**param)
classifier = classifier.fit(hog_features, ar_labels)

# Self predict
predictions = classifier.predict(hog_features)

# Compute confusion matrix
confusion_matrix = ConfusionMatrix(ar_labels.tolist(), predictions.tolist())
confusion_matrix

#Statistics are also available as follows
cms = confusion_matrix.stats()
print("Overall Accuracy is ", round(cms['overall']['Accuracy'], 2),", Kappa is ", round(cms['overall']['Kappa'], 2))
#RF: Overall Accuracy is  0.98 , Kappa is  0.98

# Save the classifier
joblib.dump(classifier, "./model/digit_recog_rf.pkl")

########### Predict the images written in black ink on white paper #######################
# Import the modules

# Load the classifier
classifier = joblib.load("./model/digit_recog_linearsvms.pkl")

# Read the input image
im = cv2.imread("./image_data/photo_1.jpg", cv2.IMREAD_UNCHANGED) #123.png typed_123.png photo_1.jpg horse_text_digit.png
np.any(im) == None
cv2.imshow("original", im); cv2.waitKey(); cv2.destroyAllWindows()

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
cv2.imshow("gray gauss", im_gray); cv2.waitKey(); cv2.destroyAllWindows()

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
#im_th = cv2.adaptiveThreshold(im_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
cv2.imshow("gray binary", im_th); cv2.waitKey(); cv2.destroyAllWindows()

# Find contours in the image
_, ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
#rects[0], rects[1] # 2 & 5 rect

# For each rectangular region, calculate HOG features and predict using RF
for rect in rects: # rect = rects[2]
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    print("w: ", rect[2], ", h: ", rect[3])
    # Now, get the digit image
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    if roi.size > 0:
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = classifier.predict(np.array([roi_hog_fd], 'float64'))
        print('Predicted number is ', nbr)
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    # end of if
# end of for rect in rects

cv2.imshow("Resulting Image is ", im); cv2.waitKey(); cv2.destroyAllWindows()

#CW: Test with your own handwriting, typed_123.png and 123.png

########### Predict the images from video written in black ink #######################
# Now, how to do it on video
# Start with one image. Test with 'horse_text_digit.png" and "horse_text_digit_2.png" - Did not do
#well - Discuss how to go about before moving to next section

# Load the classifier
classifier = joblib.load("./model/digit_recog_linearsvms.pkl")

# Read the input image
im = cv2.imread("./image_data/horse_text_digit.png", cv2.IMREAD_UNCHANGED) # horse_text_digit.png horse_text_digit_2.png
np.any(im) == None

# These constants have been taken from source video
w_offset = 100; h_offset = 100; W_orig = 490; H_orig = 360; W_text = 32; H_text = 28

# WxH is:  (490, 360) and lw, lh is 32 28
#im = cv2.resize(im, (W_orig, H_orig), interpolation = cv2.INTER_AREA)
w_factor = im.shape[1]/ W_orig; h_factor = im.shape[0]/ H_orig;
w_offset = int(100*w_factor); h_offset = int(100*h_factor); W_text = int(32*w_factor); H_text = int(28*h_factor)

# Store original to show later
im_orig = im.copy()

# Image is in HxW format although cv2 is in WxH format
h = int(im.shape[0]-h_offset); w = int(im.shape[1]-w_offset) # 632, 345
#cv2.rectangle(im, (w - W_text, h-int(H_text*1.6)), (w+W_text*4, h+int(H_text*.6)), (0, 255, 0), 3)
im = im[h-int(H_text*1.6):h+int(H_text*.6), w - W_text:]
h = h-int(H_text*1.6); w = w - W_text
cv2.imshow("Original truncated", im) # [-150,w:w+80]
cv2.waitKey(); cv2.destroyAllWindows()

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
cv2.imshow("gray gauss", im_gray); cv2.waitKey(); cv2.destroyAllWindows()

# Uncommnet follwoing to see - not good quality
_, im_th = cv2.threshold(im_gray, 10, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("gray binary", im_th); cv2.waitKey(); cv2.destroyAllWindows()

# Find contours in the image
_, ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
#rects[0], rects[1] # 2 & 5 rect

# For each rectangular region, calculate HOG features and predict using RF
for rect in rects: # rect = rects[0]
    # Draw the rectangles
    cv2.rectangle(im_orig, (w + rect[0] , h + rect[1] ), (w+ rect[0] + rect[2], h + rect[1] + rect[3]), (0, 255, 0), 3)
    #print("w: ", rect[2], ", h: ", rect[3])
    # Now, get the digit image
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))
    pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    #print(pt1, ':', pt1+leng, ' - ', pt2, ':', pt2+leng)
    if roi.size > 0:
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = classifier.predict(np.array([roi_hog_fd], 'float64'))
        print('Predicted number is ', nbr)
        cv2.putText(im_orig, str(int(nbr[0])), (w + rect[0], h + rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        del(roi_hog_fd)
    # end of if
    del(leng, pt1, pt2, roi)
# end of for rect in rects

cv2.imshow("Resulting Image is ", im_orig); cv2.waitKey(); cv2.destroyAllWindows()

del(h, w, im, im_orig, im_gray, im_th, ctrs, rects, classifier)

########### Predict from video written in black ink #######################
# Load the classifier
classifier = joblib.load("./model/digit_recog_linearsvms.pkl")

# Define one function to predict digits per images
def get_image_with_predicted_digit(classifier, im):
    # Basic sanity test
    if np.any(im) == None:
        return im

    w_offset = 100; h_offset = 100; W_orig = 490; H_orig = 360; W_text = 32; H_text = 28
    # WxH is:  (490, 360) and lw, lh is 32 28
    #im = cv2.resize(im, (W_orig, H_orig), interpolation = cv2.INTER_AREA)
    w_factor = im.shape[1]/ W_orig; h_factor = im.shape[0]/ H_orig;
    w_offset = int(100*w_factor); h_offset = int(100*h_factor); W_text = int(32*w_factor); H_text = int(28*h_factor)

    # Store original to show later
    im_orig = im.copy()

    # Image is in HxW format although cv2 is in WxH format
    h = int(im.shape[0]-h_offset); w = int(im.shape[1]-w_offset)
    im = im[h-int(H_text*1.6):h+int(H_text*.6), w - W_text:]
    h = h-int(H_text*1.6); w = w - W_text

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Make BW image
    _, im_th = cv2.threshold(im_gray, 10, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    _, ctrs, _ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, calculate HOG features and predict using RF
    for rect in rects: # rect = rects[0]
        # Draw the rectangles
        cv2.rectangle(im_orig, (w + rect[0] , h + rect[1] ), (w+ rect[0] + rect[2], h + rect[1] + rect[3]), (0, 255, 0), 3)

        # Now, get the digit image
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))
        pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        #print(pt1, ':', pt1+leng, ' - ', pt2, ':', pt2+leng)
        if roi.size > 0:
            # Resize the image
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = classifier.predict(np.array([roi_hog_fd], 'float64'))
            print('Predicted number is ', nbr)
            cv2.putText(im_orig, str(int(nbr[0])), (w + rect[0], h + rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
            del(roi_hog_fd)
        # end of if
        del(leng, pt1, pt2, roi)
    # end of for rect in rects

    del(h, w, im, im_gray, im_th, ctrs, rects)

    return(im_orig)
#get_image_with_predicted_digit

# Now read video
capture_video = cv2.VideoCapture('./image_output/horse_text.mp4')
# Change wait time for seeing digit clearly

# for full screen
window_name = 'digit recignition'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Detection
while(capture_video.isOpened()): # read the video continuously if file is open
    # Read frame from camera
    ret, image_np = capture_video.read()
    if ret == True:
#        start = time.time()
        image_np = get_image_with_predicted_digit(classifier, image_np)

        # Display output
        cv2.imshow(window_name, image_np)
        k = cv2.waitKey(20) & 0xFF #Change wait time for seeing digit clearly
        if k == 113 or k == 27: # ord('q') = 113
            break

#        print("Time taken per frame:", str(round((time.time() - start), 2)), ' (sec)')
    else:
        break
    # end of if ret == True:

capture_video.release(); cv2.destroyAllWindows()

# CW Discussion: How to build solution to detect digits with better accuracy
# CW Discussion: How to build solution to detect digits anywhere in the screen