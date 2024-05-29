import matplotlib.pyplot as plt
import numpy as np
import os
import PIL



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import glob
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


keras.backend.clear_session()
i = 0

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#data_dir = pathlib.Path('/Volumes/SSD2/aidea/')
data_dir = pathlib.Path("../train/")


batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=3,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.8,
  subset="validation",
  seed=3,
  image_size=(img_height, img_width),
  batch_size=batch_size)




class_names = train_ds.class_names
print(class_names)


AUTOTUNE = tf.data.AUTOTUNE


normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)




num_classes =  len(class_names)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)



model = tf.keras.Sequential([
  data_augmentation,
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(1,img_height, img_width, 3)),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(16, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(8, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  layers.Dropout(0.2),
  tf.keras.layers.Flatten(
      ),
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dense(img_height*img_width*3)
])

log_dir = os.path.join('/Users/max/tb_logs/fit', 'model2')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
# ModelCheckpoint回調函數幫忙儲存網路模型，可以設定只儲存最好的模型，「monitor」表示被監測的數據，「mode」min則代表監測數據越小越好。
model_mckp = keras.callbacks.ModelCheckpoint('model_out.h5',monitor='val_mean_absolute_error',save_best_only=True,mode='min')


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#model = tf.keras.models.load_model('../h5/model_hit.h5')
#history = model.fit(
#train_ds,
#  validation_data=val_ds,
#  epochs=100,
#  callbacks=[model_cbk, model_mckp]
#)



model.save('../h5/model_game.h5')
#reload_model = tf.keras.models.load_model('../h5/model.h5')
reload_model.summary()


img = Image.open('42.jpg')
img = img.resize((img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
print(img_array)
img_array = tf.expand_dims(img_array, 0)
predictions = reload_model.predict(img_array)
predictions = predictions.reshape(1,img_width,img_height,3)
predictions = tf.nn.softmax(predictions[0])
print(predictions)
predictions = np.array(predictions)
predictions = predictions*100
predictions = cv.resize(predictions, (1920, 1080), interpolation=cv.INTER_AREA)
cv.imwrite("b.jpg",predictions)
