#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import tensorflow as tf


# In[ ]:


model=tf.keras.models.load_model('cat_vs_dog_with_VGG16.h5')
img=cv2.imread('ssss.jpg')
img=cv2.resize(img,(224,224))
img=np.reshape(img,(1,224,224,3))
pred=model.predict(img)
idx=np.argmax(pred)
if(idx==0):
    print('its a cat')
else:
    print('its a dog')

