import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import pandas as pd
import os.path as path
import mrcnn.visualize
import cv2 as cv
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import pathlib
import glob


from object_detection.utils import visualization_utils as viz_utils
filename = []
filesc = []
fileclass = []
filex = []
filey = []
filew = []
fileh = []
img_height = 180
img_width = 180
batch_size = 32
j = 0
submit2=pd.DataFrame()
y = 0




PATH_TO_MODEL_DIR = os.path.join('.././data/models/centernet_hg104_1024x1024_coco17_tpu-32')


import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"

print('Loading model... ', end='')
start_time = time.time()



configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

# @tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done!')


def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return label_dir

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = download_labels(LABEL_FILENAME)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


i=0
for file in os.listdir("../ivslab_test_private_qualification/Public_Private_Testing_Dataset_Only_for _detection/1/"):
    try:
        # 讀取一幀(frame) from camera or mp4
        image_np = cv.imread("../ivslab_test_private_qualification/Public_Private_Testing_Dataset_Only_for _detection/1/"+file)
    # 加一維，變為 (筆數, 寬, 高, 顏色)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # 轉為 TensorFlow tensor 資料型態
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    # detections：物件資訊 內含 (候選框, 類別, 機率)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))

        if i==0:
            print(f'物件個數：{num_detections}')
        detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(int)

    # 第一個 label 編號
        label_id_offset = 1
    
        image_np_with_detections = image_np
        print(detections)
        viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'] + label_id_offset,
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.31,
          agnostic_mode=False)
        j = 0
        imgg = cv.resize(image_np_with_detections, # Resizing the mapped image to the original image size.
                                   (1920,
                                    1080), interpolation=cv.INTER_NEAREST)
        cv.imwrite("../files/"+file+file+'.png',imgg)
        for detection_boxes, detection_classes, detection_scores in \
        zip(detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']):
    
    # 顯示偵測結果
            if(detections['detection_scores'][j] >= 0.3):
                filex.append(int(detections['detection_boxes'][j][1]*1920))
                filey.append(int(detections['detection_boxes'][j][0]*1080))  
                fileh.append(int(detections['detection_boxes'][j][2]*1080)-int(detections['detection_boxes'][j][0]*1080))
                filew.append(int(detections['detection_boxes'][j][3]*1920)-int(detections['detection_boxes'][j][1]*1920))
                print(detections['detection_classes'][j])
                filename.append(file)

                filesc.append(detections['detection_scores'][j])
                if(detections['detection_classes'][j] == 0):
                    fileclass.append(2)
                elif(detections['detection_classes'][j] == 1):
                    fileclass.append(4)
                elif(detections['detection_classes'][j] == 2):
                    fileclass.append(1)
                elif(detections['detection_classes'][j] == 3):
                    fileclass.append(3)
                elif(detections['detection_classes'][j] == 5):
                    fileclass.append(1)
                elif(detections['detection_classes'][j] == 7):
                    fileclass.append(1)

                else:
                    fileclass.append('x')
            j = j+1
        y = y+1
    except:
        y = y+1
        print("error")
submit2.insert(0,column="confidence",value=filesc)
submit2.insert(0,column="h",value=fileh)
submit2.insert(0,column="w",value=filew)
submit2.insert(0,column="y",value=filey)
submit2.insert(0,column="x",value=filex)
submit2.insert(0,column="label_id",value=fileclass)
submit2.insert(0,column="image_filename",value=filename)

submit2.to_csv("submission.csv",index=False)
filename = []
filesc = []
fileclass = []
filex = []
filey = []
filew = []
fileh = []
img_height = 180
img_width = 180
batch_size = 32
j = 0
submit2=pd.DataFrame()
y = 0

i=0
for file in os.listdir("../ivslab_test_private_qualification/Public_Private_Testing_Dataset_Only_for _detection/2/"):
    try:
        # 讀取一幀(frame) from camera or mp4
        image_np = cv.imread("../ivslab_test_private_qualification/Public_Private_Testing_Dataset_Only_for _detection/2/"+file)
    # 加一維，變為 (筆數, 寬, 高, 顏色)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # 轉為 TensorFlow tensor 資料型態
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    # detections：物件資訊 內含 (候選框, 類別, 機率)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))

        if i==0:
            print(f'物件個數：{num_detections}')
        detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(int)

    # 第一個 label 編號
        label_id_offset = 1
    
        image_np_with_detections = image_np

        viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'] + label_id_offset,
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.31,
          agnostic_mode=False)
        j = 0
        imgg = cv.resize(image_np_with_detections, # Resizing the mapped image to the original image size.
                                   (1920,
                                    1080), interpolation=cv.INTER_NEAREST)
        cv.imwrite("../files/"+file+file+'.png',imgg)
        for detection_boxes, detection_classes, detection_scores in \
        zip(detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']):
    
    # 顯示偵測結果
            if(detections['detection_scores'][j] >= 0.3):
                filex.append(int(detections['detection_boxes'][j][1]*1920))
                filey.append(int(detections['detection_boxes'][j][0]*1080))  
                fileh.append(int(detections['detection_boxes'][j][2]*1080)-int(detections['detection_boxes'][j][0]*1080))
                filew.append(int(detections['detection_boxes'][j][3]*1920)-int(detections['detection_boxes'][j][1]*1920))
                print(detections['detection_classes'][j])
                filename.append(file)

                filesc.append(detections['detection_scores'][j])
                if(detections['detection_classes'][j] == 0):
                    fileclass.append(2)
                elif(detections['detection_classes'][j] == 1):
                    fileclass.append(4)
                elif(detections['detection_classes'][j] == 2):
                    fileclass.append(1)
                elif(detections['detection_classes'][j] == 3):
                    fileclass.append(3)
                elif(detections['detection_classes'][j] == 5):
                    fileclass.append(1)
                elif(detections['detection_classes'][j] == 7):
                    fileclass.append(1)

                else:
                    fileclass.append('x')
            j = j+1
        y = y+1
    except:
        y = y+1
        print("error")
submit2.insert(0,column="confidence",value=filesc)
submit2.insert(0,column="h",value=fileh)
submit2.insert(0,column="w",value=filew)
submit2.insert(0,column="y",value=filey)
submit2.insert(0,column="x",value=filex)
submit2.insert(0,column="label_id",value=fileclass)
submit2.insert(0,column="image_filename",value=filename)

submit2.to_csv("submission2.csv",index=False)


cv.destroyAllWindows()
