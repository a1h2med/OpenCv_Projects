import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

###### here I calculated the centroid of each shape the draw a circle in it
###### After that i sorted the contours ccentroid

Part22 = "Sorting Contours using char from left to right"

image = cv.imread('C:/Users/LENOVO/Desktop/shapes.jpg')
grey = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
canny = cv.Canny(grey,20,200)
contours , cont = cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

def centroid_calculation(image,contour):
    M = cv.moments(contour)

    # centroid calculation of x point
    cx = int(M['m20'] / M['m00'])       # i tried to use m20/m10 but it gave me a little bit far point, if you tried m20/m00 it will give you a very very far oint which is not located in the shape
    cy = int(M['m02'] / M['m00'])
    cv.circle(image,(cx,cy),10,(0,0,255),2)

def x_function(contours):
    M = cv.moments(contours)
    return int(M['m10']/M['m00'])

for i in contours:
    centroid_calculation(image,i)
    cv.imshow("circle", image)
    cv.waitKey(0)

contours_sorted = sorted(contours,key= x_function,reverse=False)    # key must be a function which return all the values

for i,c in enumerate(contours_sorted):
    cv.drawContours(image,c,-1,(0,255,0),3)
    M = cv.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
