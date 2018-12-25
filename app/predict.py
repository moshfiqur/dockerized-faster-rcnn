'''
This prediction script uses faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28 model to detect objects.
'''

import warnings
warnings.simplefilter('ignore')

import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import cv2

ROOT_DIR = '/usr/src/app'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# List of the strings that is used to add correct label for each box.
LABELS_PATH = os.path.join(MODELS_DIR, 'oid_bbox_trainable_label_map.pbtxt')
NUM_CLASSES = 200000

MODEL_PATH = os.path.join(MODELS_DIR, 'frozen_inference_graph.pb')

# We need several utilities from tensorflow/model repo. The repo can be found here:
# https://github.com/tensorflow/models
# Download the repo and add the path of research and research/object_detection 
# to the python path
TF_MODELS_DIR = os.path.join(ROOT_DIR, 'tf-models')
sys.path.append(os.path.join(TF_MODELS_DIR, 'research'))
sys.path.append(os.path.join(TF_MODELS_DIR, 'research', 'object_detection'))

# from helper import run_inference_for_single_image, load_image_into_numpy_array
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

# initialize the model variables 
detection_graph = None
label_map = None
categories = None
category_index = None

image_tensor = None
boxes = None
scores = None
classes = None
num_detections = None

def load_model():
    global detection_graph, label_map, categories, category_index
    global image_tensor, boxes, scores, classes, num_detections
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(LABELS_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def predict(image):
    detection_result = {}
    detections = []

    try:
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
                
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                
                # Actual detection.
                (result_boxes, result_scores, result_classes, result_num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                result_boxes = np.squeeze(result_boxes)
                result_classes = np.squeeze(result_classes)
                result_scores = np.squeeze(result_scores)
                
                for index, score in enumerate(result_scores):
                    if score < 0.1:
                        continue
                    
                    label = category_index[result_classes[index]]['name']
                    ymin, xmin, ymax, xmax = result_boxes[index]
                    
                    pred_dict = {
                        'label': label,
                        'score': float(score),
                        'ymin': float(ymin),
                        'xmin': float(xmin),
                        'ymax': float(ymax),
                        'xmax': float(xmax)
                    }

                    detections.append(pred_dict)
    except Exception:
        detection_result['status'] = 'error'
        detection_result['detections'] = detections
        return detection_result
    

    detection_result['detections'] = detections

    return detection_result