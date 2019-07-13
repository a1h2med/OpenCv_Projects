import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part2 = "we will convert it into gray scale"


image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image


gray_img = cv.cvtColor(image,cv.COLOR_RGB2GRAY)                 # used to convert the image into grey scale
#print(gray_img.shape)
#cv.imshow("Gray",gray_img)
#cv.waitKey()

