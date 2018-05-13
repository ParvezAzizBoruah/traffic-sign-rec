#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#Initialising the CNN
classifier = Sequential()
#1st Convolution with 32 filters
classifier.add(Conv2D(32, (3, 3), input_shape=(32,32, 3), activation = 'relu'))
#Pooling layer,of size 2x2 matrix
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Adding a 2nd convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Flattening, 2D matrix into 1D
classifier.add(Flatten())
# Dense layer with nodes 128 and 1 respectively
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(r'C:\Users\Parvez\Desktop\krs',target_size = (32,32),batch_size = 32,class_mode = 'binary')
test_set = test_datagen.flow_from_directory(r'C:\Users\Parvez\Desktop\hh',target_size = (32,32),batch_size = 32,class_mode = 'binary')
classifier.fit_generator(training_set,epochs = 2,validation_data = test_set,validation_steps = 2000)
# Part 3 - Making new predictions
import numpy as np
import cv2
img = cv2.imread('no.jpg')
img = cv2.resize(img,(32,32))
img = np.reshape(img,[1,32,32,3])

classes = classifier.predict_classes(img)
print(classes)