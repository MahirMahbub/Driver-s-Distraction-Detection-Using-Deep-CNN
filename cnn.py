
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


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=10,
                                   zoom_range = 0.2,
                                   horizontal_flip = False,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rotation_range=10,
                                  rescale = 1./255,   zoom_range = 0.2,
                                   horizontal_flip = False,
                                   fill_mode='nearest')

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (80, 80),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


classifier.fit_generator(training_set)
classifier.save_weights("Driver_Distraction.h5")


import numpy as np
from keras.preprocessing import image
test_img = image.load_img('T/img_14.jpg', target_size = (80, 80))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis = 0)
result = classifier.predict(test_img)

