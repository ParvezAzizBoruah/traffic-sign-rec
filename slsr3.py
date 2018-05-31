from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
# load the image
# load the trained convolutional neural network
print("Loading network...")
model = load_model(r"cnn5/krs.h5")
j=0
# speed-up using multithreads
cv2.setUseOptimized(True);
cv2.setNumThreads(4); 
# read image
im = cv2.imread('115.jpg')
 # resize image
im=cv2.resize(im,(256,186))
 
    # create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
ss.setBaseImage(im)
#ss.switchToSingleStrategy()
    # Switch to fast but low recall Selective Search method
    
ss.switchToSelectiveSearchFast()
 
    # Switch to high recall but slow Selective Search method
    

   # run selective search segmentation on input image
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))
     

increment = 50
 
while True:
        # create a copy of original image
        imOut = im.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if(i<200):
                x, y, w, h = rect
                
                cr=im[y:y+h, x:x+w]
                crgray = cv2.cvtColor(cr,cv2.COLOR_BGR2GRAY)
                if(crgray.size>5000 and crgray.size<15000):
                
                
                    image=cv2.resize(cr,(25,25))
                    image=cv2.resize(cr,(28,28))
                    img_to_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
                    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
                    image = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
                    image = image.astype("float") / 255.0
                    image = img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    # classify the input image
                    c= model.predict(image)[0]
                    c2=model.predict_classes(image)[0]
                    p=max(c)
                    if(c2!=38 and p>0.7):
            
                        cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.imwrite(r'crp2\win'+str(i)+'.jpg',cr)
                        j=j+1
                        if(j>2):
                            break
                    else:
                        cv2.imwrite(r'crp\win'+str(i)+'.jpg',cr)
                
        # show output
       
        cv2.imshow("Output", imOut)
 
        # record key press
        k = cv2.waitKey(0) & 0xFF
 
        if k == 109:
            break
    # close image show window
cv2.destroyAllWindows()