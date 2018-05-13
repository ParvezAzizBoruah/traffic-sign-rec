from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers.core import Activation
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import cv2
import os
def feature_vector(image):
    image = cv2.resize(image, (28, 28))
    im = img_to_array(image)
    return im
print("Describing images...")
imagePaths = list(paths.list_images(r'C:\Users\Parvez\Desktop\train\\'))
#initialize the data matrix and labels list
data = []
labels = []
# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	features = feature_vector(image)
	data.append(features)
	labels.append(label)
    # show an update every 10 images
	if i > 0 and i % 1000 == 0:
		print("Processed "+str(i)+"/"+str(len(imagePaths)))
# converting labels from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# the labels into vectors in the range [0, num_classes] -- this
# generates a vector for each label where the index of the label
data = np.array(data,dtype='float') / 255.0
labels = np.array(labels)
# partition the data into training and testing 
print("Constructing training/testing...")
(tData, testData, tLabels, testLabels) = train_test_split(data, labels, test_size=0.10)
# convert the labels from integers to vectors
tLabels = to_categorical(tLabels, num_classes=2)
testLabels = to_categorical(testLabels, num_classes=2)
# define the architecture of the network
#Initialising the CNN
classifier = Sequential()
#1st Convolution with 32 filters
classifier.add(Conv2D(20 (5,5), input_shape=(32,32,3)))
classifier.add(Activation('relu'))
#Pooling layer,of size 2x2 matrix
classifier.add(MaxPooling2D(pool_size = (2, 2),strides=(2,2)))
#Adding a 2nd convolutional layer
classifier.add(Conv2D(50, (5,5), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2),strides=(2,2)))
# Flattening, 2D matrix into 1D
classifier.add(Flatten())
# Dense layer with nodes 128 and 1 respectively
classifier.add(Dense(500))
classifier.add(Activation('relu'))
classifier.add(Dense(2))
classifier.add(Activation('relu'))
#Compiling the CNN
#classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#train the model using SGD
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
print("Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
classifier.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
classifier.fit(tData, tLabels, epochs=20)
#show the accuracy on the testing set
print("Evaluating on testing set...")
(loss, acc)=classifier.evaluate(testData, testLabels)
print("Loss="+str(loss)+" accuracy="+str(acc* 100))
from keras.preprocessing import image
img= cv2.resize(image,(32,32))
img = np.reshape(img,[1,32,32,3])
result = classifier.predict_classes(img)
print(result)
classifier.save('krs.h5')  # creates a HDF5 file 'my_model.h5'


# returns a compiled model
# identical to the previous one