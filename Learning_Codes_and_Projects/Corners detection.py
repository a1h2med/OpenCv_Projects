import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part25 = "Corners detection"

image = cv.imread('C:/Users/LENOVO/Desktop/squares.jpg')
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

# the higher the quality corner value the higher the edges will be detected
corners = cv.goodFeaturesToTrack(gray,60,.9,50)

corners = np.int0(corners)

# here we have to use ravel function as if we expanded corners we will see it's list of lists
# if we tried to get the values using something like x = corner[0], it won't give us what we want as it will give us the whole first list
for corner in corners:
    x, y = corner.ravel()
    cv.circle(image,(x,y),3,(0,0,255),-1)

cv.imshow("original image",image)
cv.waitKey()