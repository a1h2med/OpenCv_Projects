import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part23 = "identifying shapes"
###### it's trying to find the closest shape of a found contour
###### I have a shape and another image and trying to search for that shape in my image, so i did such a thing using contour Matching

image = cv.imread('C:/Users/LENOVO/Desktop/input_images.jpg')
cv.imshow("window1",image)

grey = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
canny = cv.Canny(grey,20,200)
contours , cont = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

for c in contours:
    approx = cv.approxPolyDP(c,.01*cv.arcLength(c,True),True)
    cx = approx.ravel()[0]
    cy = approx.ravel()[1]
    if len(approx) == 3:
        shape_name = "Triangle"
        cv.drawContours(image,[approx],0,(255,255,255),-1)
        cv.putText(image,shape_name,(cx,cy),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)

    elif len(approx) == 4:
        shape_name = "Rectangle"
        print(cx,cy)
        cv.drawContours(image,[approx],0,(255,255,255),-1)
        cv.putText(image,shape_name,(cx,cy),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,125),1)

    elif 6 < len(approx) < 15:
        shape_name = "star"
        cv.drawContours(image,[approx],0,(255,255,255),-1)
        cv.putText(image,shape_name,(cx,cy),1,cv.FONT_HERSHEY_SIMPLEX,(0,100,255),1)

    elif len(approx) > 12:
        cv.drawContours(image,[approx],0,(100,100,100),-1)
        print(1)
        cv.putText(image,"circle",(520,32),1,cv.FONT_HERSHEY_SIMPLEX,(255,255,255),1)

cv.imshow("win",image)
cv.waitKey()