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

#%% Face Detection using Haar Cascades
#2001, Paul Viola and Michael Jones came up with a very effective object detection method in their seminal paper
#http://www.cs.ubc.ca/~lowe/425/slides/13-ViolaJones.pdf
# Haar features are simple summations and differences of patches across the image

# Few trained classifiers are here
#https://github.com/opencv/opencv/tree/master/data/haarcascades

# Write one function for face detection so that we can use it for face in videos too
def detect_and_show_face_in_image(image_for_face_detection, face_cascade_classifier, eye_cascade_classifier, sz):
    # Convert to gray
    gray = cv2.cvtColor(image_for_face_detection, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = face_cascade_classifier.detectMultiScale(gray, scaleFactor = 1.3, # jump in the scaling factor, as in, if we don't find an image in the current scale, the next size to check will be, in our case, 1.3 times bigger than the current size.
                                                     minNeighbors = 5, minSize=(30, 30))
    str_msg_on_image = 'faces: ' + str(len(faces)); eyes_count = 0
    for (x,y,w,h) in faces:
        #Sanity test
        if h <= 0 or w <= 0: pass

        # get face rectangle
        image_for_face_detection = cv2.rectangle(image_for_face_detection,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

        # detect eye
        eyes = eye_cascade_classifier.detectMultiScale(roi_gray)


        roi_color = image_for_face_detection[y:y+h, x:x+w]
        # get eye rectangle
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyes_count = eyes_count + 1
    # end of for (x,y,w,h) in faces:

    #Update text and put the text on image
    str_msg_on_image = str_msg_on_image + ', eyes: ' + str(eyes_count)
    cv2.putText(image_for_face_detection, str_msg_on_image, (sz[0]-200, sz[1]-25), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)

    return(image_for_face_detection)
# end of detect_and_show_face_in_image

# get classifiers
face_cascade_classifier = cv2.CascadeClassifier('./model/haarcascades/haarcascade_frontalface_default.xml')
if face_cascade_classifier.empty():
    print('Missing face classifier xml file')

eye_cascade_classifier = cv2.CascadeClassifier('./model/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
if eye_cascade_classifier.empty():
    print('Missing eye classifier xml file')

# get images
image_for_face_detection = cv2.imread('./image_data/face_3.png') # face_4.jpg greenscreen.jpg
sz = (image_for_face_detection.shape[1], image_for_face_detection.shape[0])

#do face detection
image_for_face_detection = detect_and_show_face_in_image(image_for_face_detection, face_cascade_classifier, eye_cascade_classifier, sz)

# show the image with both face and eyw rectangle
cv2.imshow('image_for_face_detection',image_for_face_detection)
cv2.waitKey(0); cv2.destroyAllWindows()

# CW: Do face detecion in video (captured from webcam)

# Capture from device index or the name of a video file
capture_video = cv2.VideoCapture(0)
sz = (int(capture_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Define the codec and create VideoWriter object. http://www.fourcc.org/codecs.php
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./image_output/faces_video.avi',fourcc, 20, (640,480))

# for full screen
window_name = 'frame'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while(capture_video.isOpened()):
    ret, frame = capture_video.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret==True:
        #do face detection
        frame = detect_and_show_face_in_image(frame, face_cascade_classifier, eye_cascade_classifier, sz)
        # write the frame
        out.write(frame)

        cv2.imshow(window_name,frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
    else:
        break

# Release everything if job is finished
capture_video.release(); out.release(); cv2.destroyAllWindows()

#%% Custom object detection
# OpenCV allows custom cascades, but the process isn’t well documented.
#Example:Build cascade to detect a banana
#https://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html

#%% Image Denoising
# Many methods - one of them is follwoing
#Take a pixel, take small window around it, search for similar windows in the image, average all the
#windows and replace the pixel with the result we got. This method is Non-Local Means Denoising. It
#takes more time compared to blurring techniques we saw earlier, but its result is very good.

#OpenCV provides four variations of this technique.
#cv2.fastNlMeansDenoising() - works with a single grayscale images
#cv2.fastNlMeansDenoisingColored() - works with a color image.
#cv2.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time (grayscale images)
#cv2.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.
#Common arguments are:
#h : parameter deciding filter strength. Higher h value removes noise better, but removes details of image also. (10 is ok)
#hForColorComponents : same as h, but for color images only. (normally same as h)
#templateWindowSize : should be odd. (recommended 7)
#searchWindowSize : should be odd. (recommended 21)

image_noisy = cv2.imread('./image_data/elephant_marked.png', cv2.IMREAD_COLOR)
image_noisy_denoised = cv2.fastNlMeansDenoisingColored(image_noisy,None,10,10,7*5,21*5)
# step is big and hence will take few minutes

show_image([cv2.cvtColor(image_noisy, cv2.COLOR_BGR2RGB), cv2.cvtColor(image_noisy_denoised, cv2.COLOR_BGR2RGB)], row_plot = 1)

# Now , let us add more artificial noise
noise = np.random.randn(*image_noisy.shape)*10
image_noisy2 = np.uint8(np.clip(image_noisy + noise,0,255))
show_image([cv2.cvtColor(image_noisy, cv2.COLOR_BGR2RGB), cv2.cvtColor(image_noisy2, cv2.COLOR_BGR2RGB)], row_plot = 1)

image_noisy_denoised = cv2.fastNlMeansDenoisingColored(image_noisy2,None,10,10,7*5,21*5)
show_image([cv2.cvtColor(image_noisy, cv2.COLOR_BGR2RGB), cv2.cvtColor(image_noisy2, cv2.COLOR_BGR2RGB), cv2.cvtColor(image_noisy_denoised, cv2.COLOR_BGR2RGB)], row_plot = 1)

# For videos: Let us make noisy video

# Capture from device index or the name of a video file
capture_video = cv2.VideoCapture('./image_data/horse.mp4')
sz = (int(capture_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Define the codec and create VideoWriter object. http://www.fourcc.org/codecs.php
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for avi - DIVX, mp4 - mp4v
out = cv2.VideoWriter('./image_output/horse_noisy.mp4',fourcc,
                      int(capture_video.get(cv2.CAP_PROP_FPS)), # frame per sec
                      sz, True) # size of video
# The above property are availble in detail at https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
#Example: frame width and height by capture_video.get(3) and capture_video.get(4). It gives me 640x480 by default.
#To modify it to 320x240. Just use ret = capture_video.set(3,320) and ret = capture_video.set(4,240)
count = 0
while(capture_video.isOpened()):
    ret, frame = capture_video.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret==True:
        # write the frame
        noise = np.random.randn(*frame.shape)*10
        frame = np.uint8(np.clip(frame + noise,0,255))
        out.write(frame)

#        cv2.imshow('frame',frame)
#        #cv2.imwrite('./image_output/image' + str(count) + '.jpg',frame)
#        #print(count);
#        count = count + 1

#        k = cv2.waitKey(1) & 0xFF
#        if k == ord('q') or k == 27:
#            break
    else:
        break

# Release everything if job is finished
capture_video.release()
out.release()
cv2.destroyAllWindows()

#################### Load noisy video and denoise ######################
#Note: 1 FRAME TAKES 1 min and HENCE DO IT OFFLINE
#capture_video = cv2.VideoCapture('./image_output/horse_noisy.mp4')
#sz = (int(capture_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#
## Define the codec
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for avi - DIVX, mp4 - mp4v
#out = cv2.VideoWriter('./image_output/horse_noisy_denoised.mp4',fourcc,
#                      20.0, # frame per sec
#                      sz, True) # size of video
#
## Few initialisations
#frame_sets = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])]
#count = 0
#while(capture_video.isOpened()):
#    ret, frame = capture_video.read()
#    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#    if ret==True:
#        # Shift the image one backward and Copy the current frame to the last
#        frame_sets[0] = frame_sets[1]; frame_sets[1] = frame_sets[2]; frame_sets[2] = frame_sets[3]; frame_sets[3] = frame_sets[4]
#        frame_sets[4] = frame
#        print(count)
#        if count > 3:
#
#            # Denoise 3rd frame considering all the 5 frames
#            frame_denoised = cv2.fastNlMeansDenoisingMulti(frame_sets, 2, 5, None, 4, 7, 35)
#
#            # write the frame
#            out.write(frame_denoised); del(frame_denoised)
#        # end if count > 3
#        count = count + 1
#    else:
#        break
#    # end if ret==True:
## end while(capture_video.isOpened()):
#
## Release everything if job is finished
#capture_video.release()
#out.release()
#cv2.destroyAllWindows()
#
#del(frame_sets)

