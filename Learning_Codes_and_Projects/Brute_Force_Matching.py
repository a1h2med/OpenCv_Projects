import cv2 as cv
import numpy as np

Part26 = "Brute Force Matching"
image = cv.imread('C:/Users/LENOVO/Desktop/the_book_thief.jpg',cv.IMREAD_GRAYSCALE)
image2 = cv.imread('C:/Users/LENOVO/Desktop/me_holding_book.jpg',cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()

# each feature found it has a descriptor

kp1, des1 = orb.detectAndCompute(image,None)
kp2, des2 = orb.detectAndCompute(image2,None)

# lets make a brute force matching
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

# all what it do is to compare the descriptors with each other
matches = bf.match(des1,des2)

# we sorted it just to get the best results up, and the others down
matches = sorted(matches,key=lambda x:x.distance)

matchingResult = cv.drawMatches(image,kp1,image2,kp2,matches[:20],None)

cv.imshow("image",matchingResult)
cv.waitKey()
