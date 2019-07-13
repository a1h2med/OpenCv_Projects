import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part19 = "edge detection"
###### here we will try to detect edges through different methods
###### 1- sobel in which you can choose whether you want it vertically or horizontally
###### 2- Laplacian          , 3- Canny which is the coolest one
###### 1- convolving the image with vertical and horizontal kernel and it's commonly used
###### 2- you just convert the image into laplace domain and apply kernel to it
###### 3- it's the most effective and commonly used way, it applies gaussian filter, then finds the intensity gradient of the image, then removes pixels that are not images and finally applies hysterises threshold(if pixel is within the upper and lower threshold, it's considered an edge)

grey_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

cany = cv.Canny(image,20,170)
Laplacian = cv.Laplacian(image,cv.CV_64F)

sobel_x = cv.Sobel(image,cv.CV_64F,0,1,ksize=5)
sobel_y = cv.Sobel(image,cv.CV_64F,1,0,ksize=5)
sobel = cv.bitwise_or(sobel_x,sobel_y)

##### i saw that there's no difference between bitwise or & bitwise and, no difference when i used CV_64F & CV_32F
new_sobel_x = cv.Sobel(grey_image,cv.CV_64F,0,1,ksize=5)
new_sobel_y = cv.Sobel(grey_image,cv.CV_64F,1,0,ksize=5)
new_sobel = cv.bitwise_and(new_sobel_x,new_sobel_y)

#cv.imshow("sobel_x",sobel_x)
#cv.imshow("sobel_y",sobel_y)
#cv.imshow("sobel",sobel)
#cv.imshow("Laplacian",Laplacian)
#cv.imshow("cany",cany)
#cv.imshow("new_sobel_x",new_sobel_x)
#cv.imshow("new_sobel_y",new_sobel_y)
#cv.imshow("new_sobel",new_sobel)

