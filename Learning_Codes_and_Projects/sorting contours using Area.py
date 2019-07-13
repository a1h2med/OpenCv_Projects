import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part22 = "Sorting Contours"

##### to get contour area just type cv.contour

image = cv.imread('C:/Users/LENOVO/Desktop/shapes.jpg')

grey = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
canny = cv.Canny(grey,20,200)
contours , cont = cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

###### here you can add (reverse = True) if wanted to make it sort tsa3ody or tnazoly

sorted_contours = sorted(contours,key=cv.contourArea)

for c in sorted_contours:
    cv.drawContours(image,[c],-1,(0,255,0),3)
    cv.imshow("contour",image)
    cv.waitKey(0)
