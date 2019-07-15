import cv2 as cv
import numpy as np


Part27 = "Histogram and back projection"
image = cv.imread('C:/Users/LENOVO/Desktop/Pictures and videos/goalkeeper.jpg')
image2 = cv.imread('C:/Users/LENOVO/Desktop/Pictures and videos/pitch_ground.jpg')

hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
hsv_image2 = cv.cvtColor(image2,cv.COLOR_BGR2HSV)

# to calculate histo
# we will calculate histo to make a mask in which we will apply on the original image to select only the colors wanted
image2_hist = cv.calcHist([hsv_image2],[0,1],None , [180, 256],[0, 180 , 0 , 256])

# if we put the original image instead of hsv of it, we won't be able to detect anything from it
mask = cv.calcBackProject([hsv],[0 , 1],image2_hist,[0,180,0,255],1)
# as the output gave us a huge noise, we will apply a filter
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
mask = cv.filter2D(mask,-1,kernel)

# I think that we also have to apply threshold,to select a certain colors and get mor improvement
_,mask = cv.threshold(mask,150,255,cv.THRESH_BINARY)

# usually we apply bitwise operations on a gray scal image but here we have a 3 channel, so we have to convert our mask to 3 channels
mask = cv.merge((mask,mask,mask))

result = cv.bitwise_and(image,mask)



cv.imshow("window",result)
cv.waitKey()