import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part11 = "here we are trying to make something called pyramid for the picture which scales it"
smaller = cv.pyrDown(image)
bigger = cv.pyrUp(image)
hight , width = image.shape[0],image.shape[1]
