import cv2 as cv
import numpy as np

Part26 = "Object detection"
image = cv.imread('C:/Users/LENOVO/Desktop/Car.jpg',cv.IMREAD_GRAYSCALE)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    new_image = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()

    kp1,des1 = orb.detectAndCompute(new_image,None)
    kp2, des2 = orb.detectAndCompute(image, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1,des2)
    matches = sorted(matches,key=lambda x:x.distance)
    matched_frame = cv.drawMatches(new_image,kp1,image,kp2,matches[:20],None)

    cv.imshow("Live",frame)
    cv.imshow("window",matched_frame)
    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()