import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part1 = "we will only show the image"

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image
cv.imshow("NAME",image)                                         # used to show the image
cv.waitKey()                                                    # used to wait for a certain time (if you gave it the values) or till you take an action (if you left it empty

print(image.shape)                                              # used to print the image dimensions
B, G ,R = image[10 , 50]                                        # used to print the values of a certain Pixel

