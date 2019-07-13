import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

############  here we will try to plot the histogram forcolors using opencv

Part6 = "we will try to plot histogram"
############## image , Number of channels , mask if wanted , hist size , ranges

#histogram = cv.calcHist([image],[2],None,[256], [0, 256])
######### ravel function is used to expand the list on the histogram
######## we can play in the ranges to select a specific values

#plt.hist(image.ravel(),1000 , [0,256])
#plt.show()
#color =('b' , 'g' , 'r')

##### i is used to loop on color as channel , col is used is used to loop on the color contents
#for i,col in enumerate(color):
#    histogram2 = cv.calcHist([image],[i],None,[256],[0,256])
#   plt.plot(histogram2,color = col)
#    plt.xlim([0,256])
#plt.show()
