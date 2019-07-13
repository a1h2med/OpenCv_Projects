import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part15 = "convolution and bluring"

elephant_image = cv.imread("C:/Users/LENOVO/Desktop/elephant1.png")

bigger = cv.pyrUp(elephant_image)

kernel_filter1 = np.ones((3,3),np.float32) / 45

blurred_image = cv.filter2D(bigger,-1,kernel_filter1)

kernel_filter2 = np.ones((5,5),np.float32) / 45

blurred_image2 = cv.filter2D(bigger,-1,kernel_filter2)

blur = cv.blur(elephant_image,(3,3))                # all what it does is to take the pixels under the box and replace the central element
blur2 = cv.GaussianBlur(elephant_image,(3,3),0)     # instead of box filter it will be gaussian filter in which it will have sigma x Or y
blur3 = cv.medianBlur(elephant_image,3)             # it's said that it's the best one for filtring and getting rid out of noise as it takes all the pixels under the kernel area and central
blur4 = cv.bilateralFilter(elephant_image,9,45,45)         # it's very effective in noise removing with edge sharping
very_beutiful_looking_image = cv.fastNlMeansDenoisingColored(elephant_image,None,10,10,17,20)       # the idea behind that is just imagine that you have a static camera and you took som many Pics for the same scene and added all of them together to get one cool Picture(if you compared it with the original you will find out that there's a reduction in noise
#cv.imshow("origin",elephant_image)
#cv.imshow("blurred",very_beutiful_looking_image)
