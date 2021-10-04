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

#%% Not Deep Learning prediction/categorization/classification
# Partial discussion will happen in '7.image_digit_recognition.py'

#%% Compare images
# Let us read one image
my_image_color = cv2.imread('./image_data/elephant.png')
my_image_gray = cv2.cvtColor(my_image_color,cv2.COLOR_BGR2GRAY)

# To compare/match, we need to to have few test images. Let us create two images

# Define kernel and Run erosion and dilation
kernel = np.ones((5,5),np.uint8)
my_image_erosion = cv2.erode(my_image_gray,kernel,iterations = 1)
my_image_dilation = cv2.dilate(my_image_gray,kernel,iterations = 1)

# View
list_image_faces = [my_image_gray, my_image_erosion, my_image_dilation]
cv2.imshow('image_',np.hstack(list_image_faces)); cv2.waitKey(0); cv2.destroyAllWindows()

#%% Structural Similarity Measure (SSIM)
#Paper: http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf
#1. (x, y) location of the N x N window in each image
#2. the mean of the pixel intensities in the x and y direction
#3. the variance of intensities in the x and y direction, along with the covariance.
#Result: Can vary between -1 and 1, where 1 indicates perfect similarity.

from skimage.measure import compare_ssim as ssim # compare_mse

# Call SSIM
list_ssims = []; title = ''
for i in range(len(list_image_faces)):
    for j in range(i+1,len(list_image_faces)):
        ssim_temp = round(ssim(list_image_faces[i], list_image_faces[j]),2)
        list_ssims.append(ssim_temp)
        title = title + str(i) + str(j) + ': ' + str(ssim_temp) + ', '
        del(ssim_temp)

print(title)# (0.51, 0.51, 0.38)
cv2.imshow(title,np.hstack(list_image_faces)); cv2.waitKey(0); cv2.destroyAllWindows()

# Note: See help for comparing color images and full images.
del(list_ssims, title, list_image_faces, kernel, my_image_gray, my_image_erosion, my_image_dilation, my_image_color)

#CW: Practice with image elephant.png and elephant_marked.png and share the result
#CW: Practice with few other images mentioned in 'Data Augmentation ' Slide
#%% Image (human) comparision: We will use Face (extracted using Haar Cascades) as images

# Write one function for face detection so that we can use it for face in videos too
def detect_and_show_face_in_image(image_for_face_detection, face_cascade_classifier, sz):
    # Convert to gray
    gray = cv2.cvtColor(image_for_face_detection, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = face_cascade_classifier.detectMultiScale(gray, scaleFactor = 1.3, # jump in the scaling factor, as in, if we don't find an image in the current scale, the next size to check will be, in our case, 1.3 times bigger than the current size.
                                                     minNeighbors = 5, minSize=(30, 30))
    str_msg_on_image = 'faces: ' + str(len(faces)); rect_faces = []
    for (x,y,w,h) in faces:
        #Sanity test
        if h <= 0 or w <= 0: pass

        # get face rectangle
        image_for_face_detection = cv2.rectangle(image_for_face_detection,(x,y),(x+w,y+h),(255,0,0),2)
        rect_faces.append([x,y,w,h])
    # end of for (x,y,w,h) in faces:

    #Update text and put the text on image
    cv2.putText(image_for_face_detection, str_msg_on_image, (sz[0]-200, sz[1]-25), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)

    return(image_for_face_detection, rect_faces)
# end of detect_and_show_face_in_image

# get classifiers
face_cascade_classifier = cv2.CascadeClassifier('./model/haarcascades/haarcascade_frontalface_default.xml')
if face_cascade_classifier.empty():
    print('Missing face classifier xml file')

# get images
image_for_face_detection = cv2.imread('./image_data/face_3.png') # face_4.jpg greenscreen.jpg
sz = (image_for_face_detection.shape[1], image_for_face_detection.shape[0])

#do face detection
image_for_face_detection, rect_faces = detect_and_show_face_in_image(image_for_face_detection, face_cascade_classifier, sz)
if(len(rect_faces)) == 0:
    print('No faces detected. Revist the images')

# show the image with both face and eyw rectangle
cv2.imshow('image_for_face_detection',image_for_face_detection); cv2.waitKey(0); cv2.destroyAllWindows()

#Get all images and get the minimum parameters on which resize can happen
list_image_faces = []; new_w = new_h = max(rect_faces[0]) # Just dummy high value
for rect in rect_faces:
    image_face = image_for_face_detection[rect[1]:rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
    list_image_faces.append(image_face)
    new_h = min(new_h, image_face.shape[0])
    new_w = min(new_w, image_face.shape[1]);
    del(image_face)
# end of for loop

#resize on same scale and cnvert to gray. Also keeping color faces as need in future algorithums
list_color_image_faces = []
for i in range(len(list_image_faces)):
    color_image_face = cv2.resize(list_image_faces[i], (new_w, new_h), cv2.INTER_AREA)
    list_color_image_faces.append(color_image_face)
    list_image_faces[i] = cv2.cvtColor(color_image_face, cv2.COLOR_BGR2GRAY)
    del(color_image_face)
# end of for i in range(len(list_image_faces)):

list_my_image = np.hstack(list_image_faces)
cv2.imshow('image_for_faces',list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()

#%% Structural Similarity Measure (SSIM)
#Result: Can vary between -1 and 1, where 1 indicates perfect similarity.

from skimage.measure import compare_ssim as ssim # compare_mse

# Call SSIM
list_ssims = []; title = ''
for i in range(len(list_image_faces)):
    for j in range(i+1,len(list_image_faces)):
        ssim_temp = round(ssim(list_image_faces[i], list_image_faces[j]),2)
        list_ssims.append(ssim_temp)
        title = title + str(i) + str(j) + ': ' + str(ssim_temp) + ', '
        del(ssim_temp)

print(title)# (0.31, 0.23, 0.42)

cv2.imshow(title,list_my_image); cv2.waitKey(0); cv2.destroyAllWindows()

del(list_ssims, title, image_for_face_detection, sz, rect_faces, list_image_faces, new_w, new_h, list_color_image_faces, list_my_image)


#Pause and think: Did we do well. How to improve

#In general, Think of same platform and some data correction.
# Data Augmentation (See the example in ppt) will be taken in DL prediction.

#Another image comparison technique: locality sensitive hashing & Mean Squared Error (compare_mse)

