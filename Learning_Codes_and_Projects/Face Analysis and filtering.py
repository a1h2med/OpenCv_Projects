import cv2 as cv
import numpy as np
import dlib as dl


Part29 = "Face Analysis and filtering"

#image = cv.imread('C:/Users/LENOVO/Desktop/Pictures and videos/Me.jpg')

def TooManyfaces(exe):
    pass

def nofaces(exe):
    pass

PREDICTOR_PATH = "C:/Users/LENOVO/AppData/Local/Programs/Python/Python37/shape_predictor_68_face_landmarks.dat"
predictor = dl.shape_predictor(PREDICTOR_PATH)
detector = dl.get_frontal_face_detector()

image = cv.imread('C:/Users/LENOVO/Desktop/Me.jpg')

array = detector(image,1)
print(array)
print(array[0])
if len(array) > 1:
    TooManyfaces
if len(array) == 0:
    nofaces

landmark = np.matrix([[p.x,p.y] for p in predictor(image,array[0]).parts()])

print(landmark)
for idx,point in enumerate(landmark):
    pos = (point[0,0],point[0,1])
    #cv.putText(image,str(idx),pos , fontFace=cv.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=.4,color= (0,255,0))
    cv.circle(image,pos,3,(0,255,255))

cv.imshow("win",image)
cv.waitKey()