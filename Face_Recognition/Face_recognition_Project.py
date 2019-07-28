import cv2 as cv
import numpy as np
import dlib as dl


Part29 = "Face Recognition based on Pics that already exists in data_base"

#image = cv.imread('C:/Users/LENOVO/Desktop/Pictures and videos/Me.jpg')

Live = cv.VideoCapture(0)


# I made that to just store an image, its okay if you have one, but I weren't.
#_,dbPic = Live.read()
#cv.imshow("dbPic",dbPic)

# the template which I want to compare with (detect)
template = cv.imread('C:/Users/LENOVO/Desktop/TRY1.JPG',cv.IMREAD_GRAYSCALE)
cv.imshow("window ",template)
while True:
    _,frame = Live.read()

    # we need to apply one of the haars algorithms to detect any face and draw a rectangle on it
    # after detecting the face we will print Yes if it's matching the one we have, and nothing if it's not

    grey_Frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face_detected = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_detected.detectMultiScale(grey_Frame,1.3,5)

    for x,y,w,h in faces:

        # here I'm drawing a rectangle on the face which has been detected
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        # here I'll match the Frame with the template
        result = cv.matchTemplate(grey_Frame,template,cv.TM_CCOEFF_NORMED)

        # getting the highest intensity points detected
        loc = np.where(result > .8)

        for c in zip(*loc[::-1]):  # I put [::-1] in here because the width and hight are inverted in here and if i didn't put it it will draw a wrong rectangle although it detected the shape correctly
            cv.putText(frame,"Yes",(int((x+w)/2),int((y+h/2))),cv.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,255,0),3)

        cv.imshow("window",frame)
    key = cv.waitKey(1)
    if key == 13:
        break
Live.release()
cv.destroyAllWindows()