import os
os.chdir("D:\\trainings\\computer_vision")

#%matplotlib inline
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())

#%%  TensorFlow Object detection
# Source: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html
# Source: https://github.com/tensorflow/models/tree/master/research/object_detection
# Download steps
# 1. Download models from  https://github.com/tensorflow/models
# 2. Download protoc-*-win32.zip from https://github.com/protocolbuffers/protobuf/releases
# 3. copy "protoc.exe" to research folder or set the enviorment path to folder 'protoc-3.6.1-win32\bin"
# 4. Unzip 'models-master.zip' and move to folder models-master\models-master\research
# 5. open command prompt and run "protoc --python_out=. research\object_detection\protos\*.proto"
# If above throws error then as per discussion at link https://github.com/tensorflow/models/issues/2930 run following
# for /f %i in ('dir /b object_detection\protos\*.proto') do protoc --python_out=. object_detection\protos\%i
# Restart the Spyder (or IDE)

import six.moves.urllib as urllib
import tarfile
import zipfile
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from PIL import Image

# Since working folder is different then current working directory and hence adding the path
#sys.path.append("model\\models-master\\models-master\\research\\")
sys.path.append("model\\models-master\\research\\")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
# It may throw some warning. Ignore warning
from object_detection.utils import visualization_utils as vis_util

import time
# import the necessary packages specific to Computer vision
import cv2

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
MODEL_FILEPATH = './model/' + MODEL_FILE
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/' + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('model\\models-master\\models-master\\research\\object_detection\\data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('model\\models-master\\research\\object_detection\\data', 'mscoco_label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90 # max number of classes for given model

# Download Model: Run only once per model file
if not os.path.exists(MODEL_FILEPATH):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILEPATH)
    tar_file = tarfile.open(MODEL_FILEPATH)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, './model/')
#end of if not os.path.exists(MODEL_FILEPATH):

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map: indices to category names
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
categories # see the objects that will be predicted
category_index = label_map_util.create_category_index(categories)

##################### Object detection on image #############################
# Read image
#image_file_name = ['./model/models-master/models-master/research/object_detection/test_images/image2.jpg', './image_data/elephant.png']
image_file_name = ['./model/models-master/research/object_detection/test_images/image2.jpg', './image_data/elephant.png']
image_np = cv2.imread(image_file_name[0], cv2.IMREAD_COLOR)
image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# for full screen
window_name = 'object detection'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Extract image tensor
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Extract detection boxes
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Extract detection scores
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # Extract detection classes
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Extract number of detectionsd
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
    #end of with detection_graph.as_default():
#end of with tf.Session(graph=detection_graph) as sess:

# Display output
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
cv2.imshow(window_name, image_np); cv2.waitKey(0); cv2.destroyAllWindows()

##################### Object detection on video/webcam #############################
#capture_video = cv2.VideoCapture(0)
capture_video = cv2.VideoCapture('./image_data/horse.mp4')

# for full screen
#cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while(capture_video.isOpened()): # read the video continuously if file is open
            # Read frame from camera
            ret, image_np = capture_video.read()
            if ret == True:
                start = time.time()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Extract image tensor
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Extract detection classes
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                # Extract number of detectionsd
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                # Display output
                cv2.imshow(window_name, image_np)
                k = cv2.waitKey(1) & 0xFF
                if k == 113 or k == 27: # ord('q') = 113
                    break

                print("Time taken per frame:", str(round((time.time() - start), 2)), ' (sec)')
            else:
                break
            # end of if ret == True:

capture_video.release()
cv2.destroyAllWindows()