import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part25 = "Corners detection"

#image = cv.imread('C:/Users/LENOVO/Desktop/squares.jpg')

cap = cv.VideoCapture(0)

def nothing():
    pass
cv.namedWindow("Live")
cv.createTrackbar("quality","Live",1,100,nothing)

while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    value = cv.getTrackbarPos("quality","Live")
    value = value / 100 if value > 0 else .01
    corners = cv.goodFeaturesToTrack(gray,100,value,20)

    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            x,y = corner.ravel()
            cv.circle(frame,(x,y),3,(0,0,255),-1)


    cv.imshow("Live",frame)
    key = cv.waitKey(1)

    if key == 1:
        break

cap.release()
cv.destroyAllWindows()
