import cv2 as cv
import numpy as np
import dlib as dl
import matplotlib.pyplot as plt

Part29 = "Line detection"
#Car_image = cv.imread('C:/Users/LENOVO/Desktop/Image/test_image.jpg')

def Make_Coordinates(image,Average):
    slope,intercept = Average
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_Slope(image,Lines):
    Left_Fit = []
    Right_Fit = []
    for line in Lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameter = np.polyfit((x1,x2),(y1,y2),1)
        Slope = parameter[0]
        intercept = parameter[1]
        if Slope < 0:
            Left_Fit.append((Slope,intercept))
        else:
            Right_Fit.append((Slope,intercept))
    Left_Fit_average = np.average(Left_Fit,axis=0)
    Right_Fit_average = np.average(Right_Fit,axis=0)
    Left_Line = Make_Coordinates(image,Left_Fit_average)
    Right_Line = Make_Coordinates(image, Right_Fit_average)
    return np.array([Left_Line,Right_Line])

def Canny_detector (image):
    return cv.Canny(image, 50, 150)

def Filter (image):
    return cv.GaussianBlur(image, (5, 5), 0)

def grey (image):
    return cv.cvtColor(image,cv.COLOR_BGR2GRAY)

def Mask(image,Canny):
    height = image.shape[0]
    triangle = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(Canny)
    cv.fillPoly(mask, triangle, 255)
    new_image = cv.bitwise_and(Canny, mask)
    return new_image

def Line_detecting(image):
    Lines = cv.HoughLinesP(image, 2, np.pi / 180, 100, maxLineGap=5, minLineLength=50)
    return Lines

def Draw_Lines(image,Lines):
    if Lines is not None:
        for line in Lines:
            x1, y1, x2, y2 = line
            cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 10)


#Filtered_Car_image = Filter (Car_image)
#Grey_Car_image = grey(Filtered_Car_image)
#Canny = Canny_detector (Grey_Car_image)
#Masked_image = Mask(Car_image,Canny)
#Lines = Line_detecting(Masked_image)
#averaged_Lines = average_Slope(Car_image,Lines)
#Draw_Lines(Car_image,averaged_Lines)

Cap = cv.VideoCapture('C:/Users/LENOVO/Desktop/Image/test2.mp4')

while True:
    _,frame = Cap.read()
    Filtered_Car_image = Filter (frame)
    Grey_Car_image = grey(Filtered_Car_image)
    Canny = Canny_detector (Grey_Car_image)
    Masked_image = Mask(frame,Canny)
    Lines = Line_detecting(Masked_image)
    averaged_Lines = average_Slope(frame,Lines)
    Draw_Lines(frame,averaged_Lines)
    cv.imshow("window", frame)
    key = cv.waitKey(1)
    if key == 13:
        break

Cap.release()
cv.destroyAllWindows()