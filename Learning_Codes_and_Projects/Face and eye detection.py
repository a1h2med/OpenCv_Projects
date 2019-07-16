import cv2 as cv
import numpy as np


Part28 = "Face and eye detection"
# all the idea here that we are going to apply haar
# haar is basically a method which applies different things on the picture to detect something

# for eye detection, you won't get a wrong answer if you didn't crop the face, it will gave you the same result

#image = cv.imread('C:/Users/LENOVO/Desktop/Pictures and videos/Me.jpg')

image = cv.imread('C:/Users/LENOVO/Desktop/Me.jpg')
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

# actually I didn't see any difference when I used haarcascade_frontalface_alt2, it gave me the same result

face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_classifier = cv.CascadeClassifier('haarcascade_eye.xml')
eye = eye_classifier.detectMultiScale(gray,1.3,5)

# faces is a list holds four lines which borders the face

faces = face_classifier.detectMultiScale(gray,1.3,5)

for x,y,w,h in faces:
    cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    cv.imshow("window",image)

for x,y,w,h in eye:
    cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    cv.imshow("window",image)

cv.waitKey()