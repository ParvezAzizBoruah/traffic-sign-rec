from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import cv2
import os
# extract a 3D color histogram from the HSV color
def extracthist(image, bins=(8, 8, 8)):
    image1 = cv2.resize(image, (50, 50))  #resize the image
    hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV) #convert to hsv
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
    return hist.flatten() # return the flattened histogram as the feature vector
print("Describing images")
imagePaths = list(paths.list_images(r'C:\Users\Parvez\Desktop\nhh'))
# initialize the data matrix and labels list
data = []
labels = []
 # loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)  # load the image
    la = imagePath.split(os.path.sep)[-1].split(".")[0]  # extract the class label
    hist = extracthist(image) #extract a histogram
    data.append(hist) #inserting the histogram in the data list
    labels.append(la) #inserting the class label in the data list
    # show an update every 50 images
    if i > 0 and i % 50 == 0:
        print("Processed images:"+str(i)+"/"+str(len(imagePaths)))
le = LabelEncoder() 
labels = le.fit_transform(labels) # encode the labels, converting them from strings to integers
# partition the data into training and testing splits
print("Spliting training/testing sets")
(tData, testData, tLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.1)
# train the linear regression clasifier
print(tData[1])
print("Training Linear SVM Classifier...")
model = LinearSVC()
model.fit(tData, tLabels)
# evaluate the classifier
print("Evaluating Classifier")
pred=model.predict(testData)
print(classification_report(testLabels, pred,target_names=None))
print(pred)
import pickle
# now you can save it to a file
with open('file1.pkl', 'wb') as f:
    pickle.dump(model, f)



        
