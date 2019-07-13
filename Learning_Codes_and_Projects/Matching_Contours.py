import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part23 = "Shape Match"
###### it's trying to find the closest shape of a found contour
###### I have a shape and another image and trying to search for that shape in my image, so i did such a thing using contour Matching

image = cv.imread('C:/Users/LENOVO/Desktop/star.jpg')
target = cv.imread('C:/Users/LENOVO/Desktop/target.jpg')
target2 = cv.imread('C:/Users/LENOVO/Desktop/target2.jpg')

cv.imshow("original",image)

grey = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
canny = cv.Canny(grey,20,200)
contourr , cont = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

greyy = cv.cvtColor(target2,cv.COLOR_BGR2GRAY)
cannyy = cv.Canny(greyy,20,200)
contouurr , connt = cv.findContours(cannyy,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)


sorted_contour = sorted(contourr,key=cv.contourArea,reverse=True)

template_contour = sorted_contour[1]

gray = cv.cvtColor(target,cv.COLOR_BGR2GRAY)
cany = cv.Canny(gray,20,200)
contour , contt = cv.findContours(cany,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

for c in contouurr:
    # 1 in here refers to type of contour matching

    # for match , lower value means a close match
    match = cv.matchShapes(template_contour,c,1,0.0)    # we put zero as it's not fully utilized and ther's still an updates on it
    if match < .15:
        closest_Contour = c

cv.drawContours(target2, [closest_Contour], -1, (0, 255, 0), 3)
cv.imshow("window",target2)
cv.waitKey()