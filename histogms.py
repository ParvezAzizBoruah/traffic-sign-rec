import cv2
import csv
img = cv2.imread('n1/1.jpg',0)
img2 = cv2.imread('n1/2.jpg',0)
img3 = cv2.imread('n1/3.jpg',0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])    #calculate histograms
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])
l=len(hist)  #length of array
print(hist)
i=0
with open('his.csv', "w") as f:
    writer = csv.writer(f, dialect='excel')  #writer to write oin csv file
    writer.writerow(['Pic 1','Pic 2','Pic 3'])
    for i in range(l):
        writer.writerow([hist[i],hist2[i],hist3[i]])
