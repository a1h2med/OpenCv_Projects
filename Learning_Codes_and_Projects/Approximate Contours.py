import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part23 = "Approximate contours"

###### simply it tries to follow the shape you entered
###### actually it will be very useful if you are trying to see an image and it's very noisy but you are trying to identify the shape
###### when you decrease the accuracy it's getting more precise, when you increase it, the more value is calculated, the more it's trying to make a real shape
image = cv.imread('C:/Users/LENOVO/Desktop/House.jpg')
grey = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
canny = cv.Canny(grey,20,200)
contours , cont = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)


for c in contours:
    accuracy = .01 * cv.arcLength(c,True)       # .03 is a low accuracy, if you use it you will get a pretty cool shape, if you increased it lets say for .1 it will give you a weared shape, if you decrased it to .01 it will try to follow the shape, you have to try to reach the wanted accuracy
    approx = cv.approxPolyDP(c,accuracy,True)   # this function take three parameters (contour , accuracy , closed or open) it's preferred to be closed
    cv.drawContours(image,[approx],0,(0,255,0),3)
    cv.imshow("window", image)
    cv.waitKey(0)
