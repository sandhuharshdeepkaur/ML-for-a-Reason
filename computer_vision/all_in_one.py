import os, sys
import numpy as np
os.chdir("D:\\trainings\\computer_vision")

try:
    exec(open(os.path.abspath('1.1 image_sklearn_basics.py'), encoding="utf8").read())
    print('1.1 image_sklearn_basics.py complete')
    exec(open(os.path.abspath('1.2 image_sklearn_processing.py'), encoding="utf8").read())
    print('1.2 image_sklearn_processing.py complete')
    exec(open(os.path.abspath('2.1 image_opencv_basics.py'), encoding="utf8").read())
    print('2.1 image_opencv_basics.py complete')
    exec(open(os.path.abspath('2.2 image_opencv_arithmetic.py'), encoding="utf8").read())
    print('2.2 image_opencv_filtering.py complete')
    exec(open(os.path.abspath('2.3 image_opencv_contours.py'), encoding="utf8").read())
    print('2.3 image_opencv_contours.py complete')
    exec(open(os.path.abspath('2.4 image_opencv_filtering.py'), encoding="utf8").read())
    print('2.4 image_opencv_arithmetic.py complete')
    exec(open(os.path.abspath('3. image_opencv_face_detection.py'), encoding="utf8").read())
    print('3. image_opencv_face_detection.py')
    exec(open(os.path.abspath('4. image_opencv_feature_detection.py'), encoding="utf8").read())
    print('4. image_opencv_feature_detection.py complete')
    exec(open(os.path.abspath('5. image_opencv_image_matching.py'), encoding="utf8").read())
    print('5. image_opencv_image_matching.py complete')
    exec(open(os.path.abspath('6. image_tensorflow_object_detection.py'), encoding="utf8").read())
    print('6. image_tensorflow_object_detection.py complete')
    exec(open(os.path.abspath('7. image_digit_recognition.py'), encoding="utf8").read())
    print('7. image_digit_recognition.py complete')

    print("All in One complete!")
except:
    print("Unexpected error: ", sys.exc_info()[0])
