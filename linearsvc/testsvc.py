import pickle
import cv2
with open('file1.pkl', 'rb') as f:
    clf = pickle.load(f)
def extracthist(image, bins=(8, 8, 8)):
    image1 = cv2.resize(image, (50, 50))  #resize the image
    hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV) #convert to hsv
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
    return hist.flatten() # return the flattened histogram as the feature vector
image = cv2.imread(r"hh\107.jpg")  # load the image
image2 = cv2.imread(r"hh\stop.jpg")  # load the image
image3 = cv2.imread("pacr.jpg")  # load the image
hist = extracthist(image)
hist2 = extracthist(image2)
hist3 = extracthist(image3)
print(clf.predict([hist]))
print(clf.predict([hist2]))
print(clf.predict([hist3]))
