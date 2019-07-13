import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part23 = "Line Detection"

image = cv.imread('C:/Users/LENOVO/Desktop/lines.png')

# it's pretty cool to implement blurring
grey = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
canny = cv.Canny(grey,75,150)

# threshold is the number of points that we get, the higher it's the less lines you are going to find

lines = cv.HoughLinesP(canny,1,np.pi/180,20,maxLineGap=250 )
print(lines)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(image,(x1,y1),(x2,y2),(0,255,0),3)
cv.imshow("window",image)
cv.waitKey()