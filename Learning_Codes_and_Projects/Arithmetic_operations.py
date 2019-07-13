import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part13 = "here we will do some Arithmetic operation on the image"
Add_mat = np.ones(image.shape,dtype="uint8") * 85
new_ADDED_ima = cv.add(image,Add_mat)
#cv.imshow("ADDED_IMAGE",new_ADDED_ima)
#cv.waitKey()
new_SUBTRACTED_ima = cv.subtract(image,Add_mat)
#cv.imshow("ADDED_IMAGE",new_SUBTRACTED_ima)
#cv.waitKey()

