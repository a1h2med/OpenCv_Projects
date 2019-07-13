import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part24 = "detecting certain colors from a live video"

def nothing():
    pass

cv.namedWindow("trackbars")
cv.createTrackbar("L - H", "trackbars",0,179,nothing)
cv.createTrackbar("L - S", "trackbars",0,255,nothing)
cv.createTrackbar("L - V", "trackbars",0,255,nothing)
cv.createTrackbar("H - H", "trackbars",179,179,nothing)
cv.createTrackbar("H - S", "trackbars",255,255,nothing)
cv.createTrackbar("H - V", "trackbars",255 ,255,nothing)

video = cv.VideoCapture(0)

while True:
    ret, frame = video.read()
    cv.imshow("window",frame)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    l_h = cv.getTrackbarPos("L - H", "trackbars")
    l_s = cv.getTrackbarPos("L - S", "trackbars")
    l_v = cv.getTrackbarPos("L - V", "trackbars")
    h_h = cv.getTrackbarPos("H - H", "trackbars")
    h_s = cv.getTrackbarPos("H - S", "trackbars")
    h_v = cv.getTrackbarPos("H - V", "trackbars")

    lower_blue = np.array([l_h,l_s,l_v])
    upper_blue = np.array([h_h,h_s,h_v])

    mask = cv.inRange(hsv,lower_blue,upper_blue)
    cv.imshow("mask",mask)
    final = cv.bitwise_and(frame,frame,mask=mask)
    cv.imshow("FINAL",final)
    key = cv.waitKey(1)
    if key == 27:
        break

video.release()
