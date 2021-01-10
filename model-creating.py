#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense,Flatten,BatchNormalization,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings


# In[ ]:


#NOTE:
#you have to create a folder called 'dataset'
#in 'dataset' folder you have to create folders called 'training_set','validation_set','test_set'
#in each folder you have to create 2 folders called 'cat' and 'dog'  <---- simple letters only
#you have to copy the images of cats and dogs into the folders 'cat' and 'dog' respectively 

train_path='dataset/training_set'
valid_path='dataset/validation_set'
test_path='dataset/test_set'


# In[ ]:


train_batches=ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['cat','dog'],batch_size=32)
valid_batches=ImageDataGenerator().flow_from_directory(valid_path,target_size=(224,224),classes=['cat','dog'],batch_size=32)


# In[ ]:


#for the first time vgg16 model will be downloaded if its not in your hard disk

vgg16_model=tf.keras.applications.vgg16.VGG16()


# In[ ]:


model=Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)


# In[ ]:


#we dont have to train the layers because vgg16 model has done it previously

for layer in model.layers:
    layer.trainable=False


# In[ ]:


#we add last layer with 2 units

model.add(Dense(units=2,activation='softmax'))


# In[ ]:


model.compile(Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


model.fit(x=train_batches,validation_data=valid_batches,epochs=5,verbose=1)


# In[ ]:


#to save the model

model.save('cat_vs_dog_with_VGG16.h5')

