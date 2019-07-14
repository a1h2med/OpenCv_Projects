import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part24 = "Find bart"

image = cv.imread('C:/Users/LENOVO/Desktop/simson.jpg')
gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

template = cv.imread('C:/Users/LENOVO/Desktop/barts.jpg',cv.IMREAD_GRAYSCALE)

# if we showed the result it will give us an ugly image with a small point of light in it
# that point represents the top left of the image
result = cv.matchTemplate(gray_image,template,cv.TM_CCOEFF_NORMED)

# if you decreased the threshold a lot and put it lets say (.1) it will give you a huge mess
# it will cover all the screen with green(the color you selected) as it got hundreds of points and locations which he didn't know what should i do
# if you choose it to be (1) which will make it very very wide you will get nothing detected
# although if you made it lets say (.6) you still get a clear rectangle, so you have to be aware when choosing your threshold
loc = np.where(result > .8)     # the higher the threshold value is the wider the values you will get

# I wanted to get w,h to draw a rectangle around the matched image
w , h = template.shape[::-1]    # actually ther's nothing special in here all what I wanted is to get w then h ^_^
# also I can do it r, c (h,w) = template.shape which will gave me the same result I believe

# zip is made to get the points as it does something like zipping the (loc)
for c in zip(*loc[::-1]):        # I put [::-1] in here because the width and hight are inverted in here and if i didn't put it it will draw a wrong rectangle although it detected the shape correctly
    cv.rectangle(image,c,(c[0] + w,c[1]+h),(0,255,0),3)


cv.imshow("window",image)
cv.waitKey()