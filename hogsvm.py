# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import cv2
import os
# extract a 3D color histogram from the HSV color
def extracthistog(image):
    image1 = cv2.resize(image, (28, 28))  #resize the image
    hog = cv2.HOGDescriptor()
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(image1,winStride)
    return hist
print("Describing images")
imagePaths = list(paths.list_images(r'C:\Users\Parvez\Desktop\nhh\\'))
# initialize the data matrix and labels list
data = []
labels = []
 # loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)  # load the image
	label = imagePath.split(os.path.sep)[-1].split(".")[0]  # extract the class label
	hist = extracthistog(image) #extract a histogram
	data.append(hist) #inserting the histogram in the data list
	labels.append(label) #inserting the class label in the data list
    # show an update every 10 images
	if i > 0 and i % 10 == 0:
		print("Processed images:"+str(i)+"/"+str(len(imagePaths)))
print(data[0],data[1],data[2])
le = LabelEncoder() 
labels = le.fit_transform(labels) # encode the labels, converting them from strings to integers
# partition the data into training and testing splits
"""print("Spliting training/testing sets")
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.10)
# train the linear regression clasifier
print("Training Linear SVM Classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)
# evaluate the classifier
print("Evaluating Classifier")
predictions=model.predict(testData)
print(classification_report(testLabels, predictions,target_names=None))
print(predictions)
"""