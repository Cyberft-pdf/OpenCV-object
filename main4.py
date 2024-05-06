import cv2
face_cascade=cv2.CascadeClassifier("/Users/Adela/Desktop/data/classifier/cascade.xml")###path of cascade file
## following is an test image u can take any image from the p folder in the temp folder and paste address of it on below line 
img= cv2.imread("/Users/Adela/Desktop/data/n/1.jpg")###path of image file which we want to detect
resized = cv2.resize(img,(400,200))
gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,6.5,17)#try to tune this 6.5 and 17 parameter to get good result 
##if not getting good result try to train new cascade.xml file again deleting other file expect p and n in temp folder
for(x,y,w,h) in faces:
    resized=cv2.rectangle(resized,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('img',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()