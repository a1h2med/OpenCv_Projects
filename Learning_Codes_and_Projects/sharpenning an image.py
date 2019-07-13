import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part16 = "sharpenning an image"
##### sharpenning is the opposite of blurring which sharpen your image
##### for the kernel as blurring it must equal to 1 or it will be quite brighter or darker

Kernel = np.array([[-1,-1,-1,-1],[-1,-1,-1,15],[-1,-1,-1,-1],[-1,-1,-1,-1]])

sharpened  = cv.filter2D(image,-1,Kernel)
#cv.imshow("sharp",image)
#cv.imshow("sharpened",sharpened)

