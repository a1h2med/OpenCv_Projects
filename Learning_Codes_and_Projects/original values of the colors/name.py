import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part5 = "we were trying to print the original values of the colors"
# here all what we are trying to do is to print the original value of Red , Green, BLUE
# so we created a zero list with the saame length and dimension of the original image
# and made it into R G B but replacing R G B to ZEROS and the required color to show
Zeros = np.zeros(image.shape[:2],dtype = "uint8")
#cv.imshow("red",cv.merge([Zeros,Zeros,R]))
#cv.imshow("Green",cv.merge([Zeros,G,Zeros]))
#cv.imshow("Blue",cv.merge([B,Zeros,Zeros]))
#cv.waitKey()

