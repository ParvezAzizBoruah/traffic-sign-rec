#Range[0,1]:0 is perfect
import cv2
i=0
test=cv2.imread('ff.jpg',0)  #read the test image
histt = cv2.calcHist([test], [0], None, [256], [0, 256])  #calculate the histogram of test image
print('With similiar images:')
for i in range(6):
    im=cv2.imread(r't1\Similiar\\'+str(i)+'.jpg',0) #read images  
    hist2 = cv2.calcHist([im], [0], None, [256], [0, 256])  #calculate the histogram of all the images one-by-one
    sc=cv2.compareHist(histt, hist2, cv2.HISTCMP_BHATTACHARYYA)  #compare the histogram using BHATTACHARYYA
    print(sc)
    del im
    del hist2
i=0
print('\nWith unsimiliar images:')
for i in range(4):
    im=cv2.imread(r't1\Unsimiliar\\'+str(i)+'.jpg') #read images  
    hist2 = cv2.calcHist([im], [0], None, [256], [0, 256])  #calculate the histogram of all the images one-by-one
    sc=cv2.compareHist(histt, hist2, cv2.HISTCMP_BHATTACHARYYA)  #compare the histogram using BHATTACHARYYA
    print(sc)
    del im
    del hist2
