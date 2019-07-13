import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part9 = "Rotation of te image"
######## M = [cos theta,- sin theta] [ sin theta, cos theta]
######### to fix the space that will show up in the image you can modify the scale or set new shape for the window in the rotated image variable
hight , widt = image.shape[:2]
rotation_matrix = cv.getRotationMatrix2D((hight/2,widt/2),90,1)
rotated_image = cv.warpAffine(image,rotation_matrix,(width,hight))
#cv.imshow("rotation",rotated_image)
#cv.waitKey()

########## another way to do such a thing is to use transpose function which will rotates it for 90 degrees and full fill the space
rotate_image = cv.transpose(image)

