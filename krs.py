from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers.core import Activation
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import cv2
import os
def feature_vector(image):
    image = cv2.resize(image, (28, 28))#resize image to 28,28
    #equalize the histogram
    img_to_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    hist=cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    im = img_to_array(hist)
    return im
print("Describing images...")
imagePaths = list(paths.list_images(r'train\\'))
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
    # show an update every 1000 images
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
tLabels = to_categorical(tLabels, num_classes=44)
testLabels = to_categorical(testLabels, num_classes=44)
# define the architecture of the network
#Initialising the CNN
classifier = Sequential()
#1st Convolution with 20 filters the activation func with relu
classifier.add(Conv2D(25,(5,5),input_shape=(28,28,3),padding="same"))
classifier.add(Activation('relu'))
#Pooling layer,of size 2x2 matrix
classifier.add(MaxPooling2D(pool_size = (2, 2),strides=(2,2)))
#Adding a 2nd convolutional layer
classifier.add(Conv2D(50, (5,5),input_shape=(28,28,3),padding="same"))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2),strides=(2,2)))
#Adding a 3rd convolutional layer
classifier.add(Conv2D(80, (5,5),input_shape=(28,28,3),padding="same"))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2),strides=(2,2)))
#Adding a 4th convolutional layer
classifier.add(Conv2D(100, (5,5),input_shape=(28,28,3),padding="same"))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2),strides=(2,2)))
# Flattening, 2D matrix into 1D
classifier.add(Flatten())
# Dense layer with nodes 500 and 12 respectively
classifier.add(Dense(600))
classifier.add(Activation('relu'))
classifier.add(Dense(44))
classifier.add(Activation('softmax')) #softmax function for probability
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
#train the model using Adam
EPOCHS = 25
INIT_LR = 1e-3
BS = 32 #batch size
print("Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
classifier.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])#compile using Adam optimizer
# train the network
print("Training network...")
cl=classifier.fit_generator(aug.flow(tData, tLabels, batch_size=BS),validation_data=(testData, testLabels), steps_per_epoch=len(tData) // BS,epochs=EPOCHS, verbose=1)
print("Evaluating on testing set...")
(loss, acc)=classifier.evaluate(testData, testLabels)
print("Loss="+str(loss)+" accuracy="+str(acc* 100))
classifier.save('krs.h5')  # creates a HDF5 file 'krs.h5' to save the model
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), cl.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), cl.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), cl.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), cl.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("hhhjk.jpg")