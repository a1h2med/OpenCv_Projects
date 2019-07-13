import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part17 = "thresholding which means converting images to its binary form"
######## to do such a thing you have to convert it to gray scale
####### we have four ways to do such a thing 1- Threshold Binary , 2- Threshold Binary inverse
####### 3- Threshold Trunic , 4- Threshold To zero , 5- Threshold to zero inverse
####### 1- when you set the Threshold value it sets all the above to zero and all the coming to one , 2- do the same as (1) but inverting it
####### 3- it comes to the Threshold value and leaves all the previous as it's , but making all the coming values as the Threshold value without change
####### 4- it sets all the above values of the Threshold value to zero and leaves all the coming , 5- do the same as (4) but opposite of it

gradient_image = cv.imread('C:/Users/LENOVO/Desktop/gradient.jpg')

ret,threshold1 = cv.threshold(gradient_image,127,255,cv.THRESH_BINARY)
ret,threshold2 = cv.threshold(gradient_image,127,255,cv.THRESH_BINARY_INV)
ret,threshold3 = cv.threshold(gradient_image,127,255,cv.THRESH_TOZERO)
ret,threshold4 = cv.threshold(gradient_image,127,255,cv.THRESH_TOZERO_INV)
ret,threshold5 = cv.threshold(gradient_image,127,255,cv.THRESH_TRUNC)
#cv.imshow("THRESH_BINARY1",threshold1)

Page_image = cv.imread('C:/Users/LENOVO/Desktop/Page.jpg',0) # 0 for flag to convert it into gray scale
new_page = cv.GaussianBlur(Page_image,(3,3),0)

threshoooold = cv.adaptiveThreshold(new_page,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,3,5)
rat,thresho1 = cv.threshold(new_page,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
#cv.imshow("thres",new_page)

