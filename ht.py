import cv2
from matplotlib import pyplot as plt
img = cv2.imread('ff.jpg',0)#read image
plt.hist(img.ravel(),256,[0,256])#plot histogram
plt.show()
