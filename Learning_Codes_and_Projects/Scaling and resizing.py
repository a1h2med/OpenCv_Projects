import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part10 = "scaling and resizing"
######## in here we will make a resizing methods using different resizing algorithms
########     1- CUBIC interpolatoon , 2- Linear interpolation , 3- Nearest interpolation , 4- AREA Interpolation
########     CUBIC is better , Linear is good for zooming and sampling
########     Nearest is the fastest , Area is good for shrinking or down sampling
########     there's also lANCZOS4 which is the best
image_scaled = cv.resize(image,None,fx=.75,fy=.75)
image_scaled2= cv.resize(image,None,fx=.75,fy=.75,interpolation= cv.INTER_LANCZOS4)
image_scaled3 = cv.resize(image,None,fx=.75,fy=.75,interpolation= cv.INTER_LINEAR)
image_scaled4 = cv.resize(image,None,fx=.75,fy=.75,interpolation= cv.INTER_NEAREST)
image_scaled5= cv.resize(image,(500,500),fx=.75,fy=.75,interpolation= cv.INTER_CUBIC)
#image_scaled6 = cv.resize(image,None,fx=.75,fy=.75,interpolation= cv.INTER_LINEAR)
#image_scaled7 = cv.resize(image,None,fx=.75,fy=.75,interpolation= cv.INTER_NEAREST)
#cv.imshow("scaled",image_scaled)
#cv.waitKey()
