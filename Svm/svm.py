import numpy as np
from sklearn import svm
X = np.array([[1,2],   #array of x,y coordinates
             [5,8],    #for training set
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])
y = [0,1,0,1,0,1] #target classes 0 or 1
clf = svm.SVC(kernel='linear', C = 1.0) #define classifier
clf.fit(X,y)
print(clf.decision_function(X))
print(clf.predict([[1.,15.]] ))    #test values
print(clf.predict([[10.58,10.76]]))