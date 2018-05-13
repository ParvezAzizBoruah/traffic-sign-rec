from keras.models import load_model
import cv2
import numpy as np
# returns a compiled model
# identical to the previous one
image=cv2.imread('hisp.png')
img= cv2.resize(image,(32,32))
img = np.reshape(img,[1,32,32,3])

model = load_model('krs.h5')
result = model.predict_classes(img)
print(result)