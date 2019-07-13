import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part24 = "Blob detection"

# blob => is a group of connected Pixels, that share a similar property

image = cv.imread('C:/Users/LENOVO/Desktop/flowers.jpg')

image = cv.pyrUp(image)
gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

# I used SimpleBlobDetector_create() instead of SimpleBlobDetector() because of my opencv version
detector = cv.SimpleBlobDetector_create()
key_points = detector.detect(gray_image)

blank = np.zeros((1,1))

# for DRAW_MATCHES_FLAGS_DEFAULT ther's not a big difference as the difference is in the circle size
blobs = cv.drawKeypoints(image,key_points,blank,(0,255,0,cv.DRAW_MATCHES_FLAGS_DEFAULT))

cv.imshow("blob",blobs)
cv.waitKey()