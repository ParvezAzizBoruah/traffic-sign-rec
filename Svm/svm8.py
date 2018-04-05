import numpy as np
from sklearn import svm
X = np.array([[1,2],   #array of x,y coordinates
             [1,3],    #for training set
             [1,1],
             [3,2],[2,4],[3,3],
             [7,8],[8,8],[9,10]])
y = [0,0,0,1,1,1,2,2,2] #target classes 0 or 1
clf = svm.SVC() #define classifier
clf.fit(X,y)
print(clf.decision_function(X))
print(clf.predict([[1,1]] ))    #test values
print(clf.predict([[3,1]]))
print(clf.predict([[10.58,10.76]]))