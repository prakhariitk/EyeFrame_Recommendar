# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 21:30:20 2021

@author: LENOVO
"""
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
df = pd.read_csv('eyeframe_dataset.csv')
print(df.head())

df = df[['product_id','parent_category','Image_Front','frame_shape']]


multi_label = []
for i in range(len(df)):
  k = [df['parent_category'][i],df['frame_shape'][i]]
  multi_label.append(k)

df['multilabel'] = multi_label
print(multi_label[0])

from sklearn import preprocessing
lb = preprocessing.MultiLabelBinarizer(classes=("eyeframe","sunglasses","Non-Power Reading",'Rectangle','Aviator','Wayfarer','Oval'))
Y = lb.fit_transform(multi_label)
Y_test = lb.fit_transform(multi_label)

print(Y_test[0])

tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)

imported_img = []
for f in range(len(df)):
  filename = 'imgfiles/' + 'image' + str(f) + '.jpg' 
  original = load_img(filename, target_size=(224, 224))
  numpy_image = img_to_array(original)
  image_batch = np.expand_dims(numpy_image, axis=0)
  imported_img.append(image_batch)
    
images = np.vstack(imported_img)


print(images.shape)
print(images[0])


import pickle

from os import listdir
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import vgg16
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.applications import vgg19 #For Transfer Learning
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout

#defining training and test sets
x_train = np.array(images,dtype=np.float32)/255.0
Y_test = np.array(Y_test)

x_train, x_test, y_train, y_test = train_test_split(x_train, Y_test, test_size=0.2, random_state = 69)

train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)


test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip= True, zoom_range=.1)

#Fitting the augmentation defined above to the data
train_generator.fit(x_train)
test_generator.fit(x_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)


from tensorflow import keras
from keras.applications.vgg16 import VGG16
base_model = tf.keras.applications.ResNet50(include_top = False, weights = 'imagenet', input_shape = (224,224,3), classes = y_train.shape[1])
#Adding the final layers to the above base models where the actual classification is done in the dense layers
model= Sequential()
model.add(base_model) 
model.add(Flatten()) 
#Model summary
model.summary()

#Adding the Dense layers along with activation and batch normalization
model.add(Dense(1024,activation=('relu'),input_dim=512))
model.add(Dense(512,activation=('relu'))) 
model.add(Dense(256,activation=('relu'))) 
model.add(Dense(128,activation=('relu')))
model.add(Dropout(.2))
model.add(Dense(7,activation=('softmax'))) 

#Checking the final model summary
model.summary()

from tensorflow import keras
batch_size= 5
epochs=1
learn_rate=.001
sgd=keras.optimizers.SGD(lr=learn_rate,momentum=.9,nesterov=False)
adam=keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
model.fit_generator(train_generator.flow(x_train, y_train, batch_size= batch_size),epochs = 1, steps_per_epoch = x_train.shape[0]//batch_size,callbacks=[checkpoint], verbose = 1)




#Plotting the training and validation loss and accuracy
f,ax=plt.subplots(2,1) 

#Loss
ax[0].plot(model.history.history['loss'],color='b',label='Training Loss')


#Accuracy
ax[1].plot(model.history.history['accuracy'],color='b',label='Training  Accuracy')

