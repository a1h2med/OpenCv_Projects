import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

Part23 = "Line Detection from a live video"

video = cv.VideoCapture('C:/Users/LENOVO/Desktop/road_car_view.mp4')

while True:
	# ret tells us whether there's still a frame or not, as it only returns true or false
	# if you want to make it a continuos video that runs forever just check on ret value if it false
	# that means that the video ended, so we load it again and continue
	# if we wanna break the loop we can set cv.waitKey(25) to a variable, then checks if the value is higher than 25 
	# so it should be no frame to execute, then we break the loop
	# or we can simply check on ret value
    ret, frame = video.read()

    cv.imshow("window",frame)
    cv.waitKey(25)

video.release()
