######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import imutils
import pytesseract

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'test3.mpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join('labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
# frame_width = int(video.get(3))
# frame_height = int(video.get(4))
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    _, frame = video.read()
    frame = imutils.resize(frame, width=720)
    # frame.set(cv2.cv.CV_CAP_PROP_FPS, 20)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    # print("test",test)
    # print("Boxes",boxes[0][0])

    boxes=boxes[0][0]
    width, height, channal= frame.shape
    # print(width)
    # print(height)
    y=boxes[0]
    x=boxes[1]
    h=boxes[2]
    w=boxes[3]
    # # crop_img = frame[int(y)*255:255, int(x)*255:100]
    img = frame[int(y*width)+5:int(h*width)-5, int(x*height)+5:int(w*height)-5]

    # print(crop_img)

    # All the results have been drawn on the frame, so it's time to display it.


    # img = cv2.imread("test4.png") 
    # cv2.imwrite('messigray.png',img)
    retval, img = cv2.threshold(img,130,255,cv2.THRESH_BINARY)
    img = cv2.resize(img,(0,0),fx=3,fy=3)
    img = cv2.GaussianBlur(img,(11,11),5)
    img = cv2.medianBlur(img,1)
    # thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    cv2.imwrite('messigray1.png',img)
    custom_oem_psm_config = r'--oem 3 --psm 8'
    txt = pytesseract.image_to_string(img, config=custom_oem_psm_config)
    print('recognition:', txt)
    txt = pytesseract.image_to_string(img,lang='eng',config='--psm 6')
    print('recognition:', txt)
    txt = pytesseract.image_to_string(img,lang='eng',config='--psm 7')
    print('recognition:', txt)
    txt = pytesseract.image_to_string(img,lang='eng',config='--psm 8')
    print('recognition:', txt)
    txt = pytesseract.image_to_string(img,lang='eng',config='--psm 9')
    print('recognition:', txt)
    txt = pytesseract.image_to_string(img,lang='eng',config='--psm 10')
    print('recognition:', txt)
    txt = pytesseract.image_to_string(img,lang='eng',config='--psm 11')
    print('recognition:', txt)
    print("")
    cv2.imshow('Object detector', frame)


    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
# out.release()
cv2.destroyAllWindows()
