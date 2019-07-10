import cv2 as cv

def sketch(IMAGE):
    ######## if we didn't convert it into gray then blurring it we will get a lot of noise in our video stream

    img_gray = cv.cvtColor(IMAGE,cv.COLOR_BGR2GRAY)

    ######## I tried to use all the filters and they all almost gave me the same results
    filter = cv.GaussianBlur(img_gray,(5,5),0)
    #filter = cv.bilateralFilter(img_gray,5,75,75) # gave me the same result

    ########## gave me a lot of noise when i used sobel
    #filtered  = cv.Sobel(filter, cv.CV_64F, 1, 0, ksize=5)
    #filtered2 = cv.Sobel(filter, cv.CV_64F, 0, 1, ksize=5)
    #filter1 = cv.bitwise_or(filtered,filtered2)
    ####### it gave me a quite good image but when you stop moving, using CV_64F only, as it's not clear at all when i used CV_8U, also when i used the image directly it was better than using it with filter
    #laplac = cv.Laplacian(IMAGE,cv.CV_64F)

    cany_image = cv.Canny(filter, 10, 70)
    ret,mask = cv.threshold(cany_image,70,255,cv.THRESH_BINARY_INV)     # I used that just to invert the result which I see on the screen
    return mask

capture = cv.VideoCapture(0)
while True:
    ret,frame = capture.read()
    cv.imshow("LIVE",sketch(frame))
    if cv.waitKey(1)==13:
        break
capture.release()
cv.destroyAllWindows()