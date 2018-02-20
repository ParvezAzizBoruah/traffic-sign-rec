import numpy as np
import cv2
i=0
img = cv2.imread('ff.jpg', 0) # Read in your image
et,thresh = cv2.threshold(img,127,255,0)
im2, contours, hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # Your call to find the contours
l=int(len(contours))
for i in range(l):
    idx = i # The index of the contour that surrounds your object
    mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
    out = np.zeros_like(img) # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]
    cv2.imwrite('img/'+str(idx) + '.jpg', out)
    