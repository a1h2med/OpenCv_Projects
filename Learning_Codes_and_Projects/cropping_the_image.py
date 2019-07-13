import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

hight , width = image.shape[0],image.shape[1]

Part12 = "here we wanna crop the image but as we don't have a direct crop tool we will use Numpy to do so"
########### here we took the starting row in which we wanna crop our image and the starting col
########### then we took the end in which we wanna end our cropping, then specify those colimns and rows from our image
Start_row , Start_col = int(hight*.1), int(width*.2)
end_row , end_col = int(hight*.75), int(width*.75)
new_image = image[Start_row:end_row, Start_col:end_col]

