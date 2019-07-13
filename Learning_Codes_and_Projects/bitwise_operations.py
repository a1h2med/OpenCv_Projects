import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part14 = "we will do some bit wise operations on the window"

square = np.zeros((300,300),dtype="uint8")
cv.rectangle(square,(50,50),(250,250),205,-2)

ellipse = np.zeros((300,300),dtype="uint8")
cv.ellipse(ellipse,(150,150),(150,150),30,0,180,152,-1)

And = cv.bitwise_and(square,ellipse)
OR = cv.bitwise_or(square,ellipse)
XOR = cv.bitwise_xor(square,ellipse)
Not_square = cv.bitwise_not(square)
Not_Ellipse = cv.bitwise_not(ellipse)

#cv.imshow("square",square)
#cv.imshow("ellipse",ellipse)
#cv.imshow("AND",And)
#cv.imshow("OR",OR)
#cv.imshow("XOR",XOR)
#cv.imshow("Not_square",Not_square)
#cv.imshow("Not_ellipse",Not_Ellipse)
