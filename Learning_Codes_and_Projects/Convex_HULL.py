import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part23 = "Convex Hull"
###### it's something wich follow the shape from the outside (moving from a point to another) which I can fit around the shape

image = cv.imread('C:/Users/LENOVO/Desktop/Hand.jpg')
grey = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
canny = cv.Canny(grey,20,200)
contours , cont = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

n=len(contours)-1       # he said that I have to use it as if I didn't he will detect the window border as a contour (which didn't happen to me)
contourss = sorted(contours,key=cv.contourArea,reverse=False)[:n]
hull = []
cv.imshow("window1",image)
for c in contourss:
    hull = cv.convexHull(c)

cv.drawContours(image,[hull],0,(0,255,0),3)         # I draw it in here because it didn't show me the last result of it -_-

cv.imshow("window", image)
cv.waitKey()