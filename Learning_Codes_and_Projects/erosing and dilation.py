import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part18 = "erosion and dilation"
###### Erosion Removes pixels at the boundaries of objects in an image
###### dilation adds pixels to the boundaries of objects in an image
###### open means to do erosion then dilation(used to remove noise) , closing is vice versa
kernel = np.ones((5,5),np.uint8)

erosion = cv.erode(image,kernel)
dilation = cv.dilate(image,kernel)
openning = cv.morphologyEx(image,cv.MORPH_OPEN,kernel)
closing = cv.morphologyEx(image,cv.MORPH_CLOSE,kernel)
#cv.imshow("erosion",erosion)

