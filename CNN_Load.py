#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 23:25:12 2018

@author: sand_boa
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()


classifier.add(Conv2D(32, (3, 3), input_shape = (80, 80, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'sigmoid'))
classifier.load_weights("Driver_Distraction.h5")

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
result = 0
import numpy as np
from keras.preprocessing import image
img_path = 'T/img_19.jpg'
test_img = image.load_img(img_path, target_size = (80, 80))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis = 0)
result = classifier.predict(test_img)
res = np.squeeze(result)
classes = ['Driving', 'Using Mobile in right hand', 'talking in the phone in right hand', 'Using Mobile in left hand', 'talking in the phone in left hand',
           'Doing Something to car', 'Drinking', 'Looking Behind', 'Doing Something With his body', 'Looking Outside']
tupl = zip(res,classes)
for i in tupl:
    if i[0] == 1:
        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw 

        img = Image.open(img_path)
        print(img)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/pagul/Pagul.ttf", 20)
        draw.text((10,10), i[1], fill=(255,0,0),font = font)
        img.save('pil_text4.jpg')
        break


