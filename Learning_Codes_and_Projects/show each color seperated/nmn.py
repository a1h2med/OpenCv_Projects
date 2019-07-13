import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part4 = "we were trying to show each color seperated then merge it so we can modify it as we want and control each color intensity"

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

###########       BGR is stored in computer as RGB

B , G , R = cv.split(image)                                 # here we are splitting the RGB values and storing it in var
#cv.imshow("red",R)                                         # the output will Be gray in all of the coming because
#cv.imshow("Blue",B)                                        # we are printing one dimension of color
#cv.imshow("Green",G)
#cv.waitKey()

merged = cv.merge([B+100 , G+50 , R+50])                    # it's used to combine the colorsback again
#cv.imshow("merged",merged)
#cv.waitKey()
