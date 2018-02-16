# traffic-sign-rec
# A Repository
import cv2
from PIL import Image
import csv
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-framey
    ret, f = cap.read()
    # Our operations on the frame come here
    # Display the resulting frame
    cv2.imshow('frame',f)
    if cv2.waitKey(1) & 0xFF == ord('q'):#loop and press q to close window
        cv2.imwrite('pa.jpg',f)
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()#close the window
c=1;#counter for pixel no.
im=Image.open('pa.jpg')#open the image
pix=im.load()#load the image
w=im.size[0]#size of coloumn
h=im.size[1]#size of row
with open("par.csv",'w') as f:#open a csv file
     writer = csv.writer(f, dialect='excel')#writer to write oin csv file
     writer.writerow(['Pixelno.','X','Y','RGB colour'])
     for i in range(w):#loop for coloumn
         for j in range(h):#loop for row
              p=pix[i,j]#store pixel value
              writer.writerow([c,i,j,p])#write pixno.,xvalu,yvalue
              c=c+1
