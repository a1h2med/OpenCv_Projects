import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('C:/Users/LENOVO/Desktop/ME.jpg')             # used to read an image

Part7 = "we will try to draw on images"

######## here we created a black rectangle
black_rec = np.zeros((500,500,3),np.uint8)


black = np.zeros((200,200),np.uint8)    # we can draw a line in here but it will draw onlly one channel which is B from BGR as it's not BGR it's only two channels
#v.imshow("black",black_rec)
#cv.imshow("black_with out 3 channels",black_rec)
#cv.waitKey()

cv.line(black,(0,0),(199,199),(100,0,0),5)
#cv.imshow("black",black)
#cv.waitKey()

cv.rectangle(black_rec,(20,20),(400,400),(200,155,100),5)
#cv.imshow("black",black_rec)
#cv.waitKey()
########################## we also have cv.puttext , cv.circle , cv.polygon

