import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import PIL.Image as Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import glob
import cv2 as cv
import numpy as np
import d2l

cap = cv.VideoCapture(0)
cap.set(2,240)
class ShapeAnalysis:
        def analysis(self, frame):
            a = 0
            h, w, ch = frame.shape
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.bilateralFilter(gray,7,75,75)
            result = np.zeros((h, w, ch), dtype=np.uint8)
            h, w = gray.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)

            ret, binary = cv.threshold(gray, 68,255,cv.THRESH_BINARY_INV)
            kernel = np.ones((3,3), np.uint8)
            gray = cv.dilate(binary, kernel, iterations = 30)
            gray = cv.erode(gray, kernel, iterations = 30)
            gray = cv.dilate(binary, kernel, iterations = 10)
            
            
            
            gray = cv.bitwise_not(gray)
            dilation = cv.Canny(gray, 20, 86)
            contours, hierarchy = cv.findContours(dilation,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            try:
                            print(len(contours))
                            for cnt in contours:

                                    (x, y, w, h) = cv.boundingRect(cnt)
                                    cv.imshow('img', src)
                                    shape_type=""
                                    
                                    crr_image = src[y-20:y+h+20 , x-20:x+w+20]
                                    cv.imwrite("test/test.jpg", crr_image)


                    
                            batch_size = 50
                            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                            model = keras.models.load_model("model.h5")
                            data_dir = pathlib.Path('../train')

                            batch_size = 32
                            img_height = 180
                            img_width = 180


                            train_ds = tf.keras.utils.image_dataset_from_directory(
                              data_dir,
                              image_size=(img_height, img_width),
                              batch_size=batch_size)
                            class_names = train_ds.class_names
                            

                            
                             
                            files = os.listdir('test')
                            img = Image.open('test/test.jpg')
                            img = img.resize((img_height, img_width))
                            img_array = tf.keras.utils.img_to_array(img)
                            img_array = tf.expand_dims(img_array, 0) # Create a batch

                            predictions = model.predict(img_array)
                            score = tf.nn.softmax(predictions[0])
                            if(class_names[np.argmax(score)] == "無資料"):
                                        print("無資料")
            except:
                        print("錯誤")


while(True):
    # 擷取影像
    ret, frame = cap.read()
    src = frame
    cv.imshow("Analysis Result",src)
    ret, frame = cap.read()
    src = frame
    ld = ShapeAnalysis()
    ld.analysis(src)






