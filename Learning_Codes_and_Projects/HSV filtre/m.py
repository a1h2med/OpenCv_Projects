import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

###############    HSV ### Ranges => H: 0 - 180       ,    S: 0 - 255        ,  V: 0 - 255

Part3 = "we will implement HSV filtration on it"

hsv_img = cv.cvtColor(image,cv.COLOR_RGB2HSV)
cv.imshow("ordinary",hsv_img)
cv.imshow("Heu",hsv_img[:,:,0])
cv.imshow("value",hsv_img[:,:,1])
cv.imshow("saturation",hsv_img[:,:,2])
cv.waitKey()
