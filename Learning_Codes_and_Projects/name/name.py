import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part8 = "Translation of the image"
######### in this Part all what we are going to do is to shift the imagre into some certain distance and its affain type
########## T sould be like T = [1 0 TH] [ 0 1 TW]
height, width = image.shape[:2]
T = np.float32([[1,0,height/4],[0,1,width/4]])
img_translation = cv.warpAffine(image,T,(width,height))
#cv.imshow("ue",img_translation)
#cv.waitKey()
