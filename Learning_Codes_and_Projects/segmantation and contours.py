import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part20 = "Segmantation and Contours"
####### Segmantation means detecting each part of a certain object
####### Contours are continuous lines or curves that bound or cover the full boundary of an object in an image,it's quitly used in object detection and Shape analysis
####### the function findContours can only deals with gray scale images

threesquares = np.zeros((500,1000,3),np.uint8)              # here i tried to create a full black window to draw three squares on it
cv.rectangle(threesquares,(10,10),(200,200),255,3)
cv.rectangle(threesquares,(210,210),(400,400),255,3)
cv.rectangle(threesquares,(220,10),(400,200),255,3)

gray = cv.cvtColor(threesquares,cv.COLOR_BGR2GRAY)          # I have to convert the image into gray scale as the function (find contours) can only deals with gray images

edged = cv.Canny(gray,30,200)                               # it's better to use it to reduce noise and help the function from getting rid of unneeded edges

#if you print contours you will see list of lists which have multiple points inside it which represents that it keeps tracking the lines
# hirachey stores the relationship between childs and parents in contours

# CHAIN_APPROX_NONE gives you hundreds of points which corresponds to a certain shape whether it's a line or anytging else
# CHAIN_APPROX_SIMPLE gives you only the starting and ending points of the shape if you have a rectangle it will give you only four points

# Retreval mode there's two main types that you will keep using and dealing with, but it has 4 types
# RETR_List: which give you all the contours,RETR_EXTERNAL : which give you only external contours, neglecting what's inside the shape
# RETR_COMP: Retreives all in a 2-level hirachey , RETR_TREE : Retreives all in full hirachy
#### RETR_EXTERNAL , RETR_TREE are the most usefull
# RETR_TREE returns the hirachey layered like that [Next,Previous,First Child,Parent] which is very useful (something like graph

contours , hirachey = cv.findContours(edged,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

# if you don't want contours to edit on your image just type (image name).copy()
cv.drawContours(threesquares,contours,-1,(0,255,0),3)       # it's a funcction used to draw the contours output on the image as the previous function edits on the original image
