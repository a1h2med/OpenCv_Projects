import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part23 = "Line Detection from a live video"

# we need to get the value in which I can see only yellow colors

def nothing():
    pass

cv.namedWindow("trackback")

cv.createTrackbar("L - H","trackback",0,179,nothing)
cv.createTrackbar("L - S","trackback",0,255,nothing)
cv.createTrackbar("L - V","trackback",0,255,nothing)
cv.createTrackbar("H - H","trackback",179,179,nothing)
cv.createTrackbar("H - S","trackback",255,255,nothing)
cv.createTrackbar("H - V","trackback",255,255,nothing)

video = cv.VideoCapture('C:/Users/LENOVO/Desktop/road_car_view.mp4')

# to get yellow value it should be from (20 - 89) for H all the others are from 0 to 255
while True:
    ret, ORIGINAL_frame = video.read()
    if not ret:
        video = cv.VideoCapture('C:/Users/LENOVO/Desktop/road_car_view.mp4')
        continue

    # I think I'm gonna apply a gaussian blur as the video has a lot of noise

    frame= cv.GaussianBlur(ORIGINAL_frame, (3, 3), 0)

    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    l_h = cv.getTrackbarPos("L - H","trackback")
    l_s = cv.getTrackbarPos("L - S", "trackback")
    l_v = cv.getTrackbarPos("L - V", "trackback")
    h_h = cv.getTrackbarPos("H - H", "trackback")
    h_s = cv.getTrackbarPos("H - S", "trackback")
    h_v = cv.getTrackbarPos("H - V", "trackback")

    # values are [18,94,140],[48,255,255]

    low_yellow = np.array([18,94,140])
    upper_yellow = np.array([48,255,255])

    mask = cv.inRange(hsv,low_yellow,upper_yellow)

    final = cv.bitwise_and(frame,frame,mask=mask)

    Canny = cv.Canny(mask,75,150)

    lines = cv.HoughLinesP(Canny,1,np.pi/180,50,maxLineGap=50)

    # as we are tracking a line on a video, so if it doesn't have a line how won't be able to draw anything
    # for the video I have it won't give me an error I believe
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
            
    cv.imshow("final_video",final)
    cv.imshow("mask", mask)
    cv.imshow("Video", frame)
    cv.imshow("canny",Canny)

    cv.waitKey(25)

video.release()
