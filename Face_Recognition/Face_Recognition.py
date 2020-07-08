'''
this project I made as part of smart door unlock.
It will get an image using a camera and checks the person's image.
If he/she is a member of house, door will open otherwise it will inform the house manager.

In this project I've used (face_recognition) from dlib as it's fast, and reliable.
Imaged of the house members should exist in (images) folder so I can read it.
I've used opencv for image reading and some basic analysis.
'''
import cv2
import numpy as np
import face_recognition
import os

# this function encoding all the images passed to it and returns encoding list.
def Encoding_images(images):
    encoding_list = []
    for i in images:
        encoded_image = face_recognition.face_encodings(i)[0]
        encoding_list.append(encoded_image)
    return encoding_list

# here I'm just getting the path for the images and the names assigned to it
# for example (I will read bill gates's image and store its name as bill gates), so when I detect his image
# I will print his name or store his name in wherever I want.

path = 'images'
images = []
image_name = []
for c in os.listdir(path):
    img = cv2.imread(f'{path}/{c}')
    images.append(img)
    image_name.append(os.path.splitext(c)[0])

# sending images to be ecoded
encde_List_KImages = Encoding_images(images)

# starting a real time video
# this real time video is made just for testing, but in real time I will use a real camera
# which will send me the image then I'll check on it

capture = cv2.VideoCapture(0)
while True:
    _, frame = capture.read()
    frameS = cv2.resize(frame, (0, 0), None, .25, .25)
    frameS = cv2.cvtColor(frameS,cv2.COLOR_BGR2RGB)

    # Please note that here I didn't use [0] like above or like you will see in online tutorials
    # As If you tried to put it and used it in real time it will work fine as long you just putting the image
    # Directly to it, not moving not doing anything else as if you moved it will crash as it's getting the array
    # Inside the list, So we can use it if we are getting just an image, but we are in real time and we don't
    # Want it to crash, so I will get the whole list so when there's a movememt I will get an empty list
    # Not causing crashing for my project.
    face_frame_location = face_recognition.face_locations(frameS)
    face_frame_encodings = face_recognition.face_encodings(frameS)
#    print(face_frame_encodings)
    # Here I'm just looping on encodings and compare it, then getting the distance
    # Print the name of the person with the lowest distance.

    for encodings in face_frame_encodings:
        match = face_recognition.compare_faces(encde_List_KImages, encodings)
        faceDis = face_recognition.face_distance(encde_List_KImages, encodings)

        matched_index = np.argmin(faceDis)
        print(matched_index)
        # Print the name.
        if match[matched_index]:
            name = image_name[matched_index].upper()
            print(name)

    cv2.imshow("LIVE", frame)
    if cv2.waitKey(1) == 13:
        break
capture.release()
cv2.destroyAllWindows()

'''
Imgelon = cv2.imread('images/elon.jfif')
ImgelonTest = cv2.imread('images/elonmask.jfif')
billgates = cv2.imread('images/bill gates.jfif')

faceloc = face_recognition.face_locations(Imgelon)[0]
encodeElon = face_recognition.face_encodings(Imgelon)[0]
cv2.rectangle(Imgelon, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (0, 255, 0), 2)

facelocTest = face_recognition.face_locations(ImgelonTest)[0]
encodeElonTest = face_recognition.face_encodings(ImgelonTest)[0]
cv2.rectangle(ImgelonTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (0, 255, 0), 2)

facelocbill = face_recognition.face_locations(billgates)[0]
encodebill = face_recognition.face_encodings(billgates)[0]
cv2.rectangle(billgates, (facelocbill[3], facelocbill[0]), (facelocbill[1], facelocbill[2]), (0, 255, 0), 2)

result = face_recognition.compare_faces([encodeElon], encodebill)
faceDis = face_recognition.face_distance([encodeElon],encodebill)
print(result, faceDis)

cv2.imshow('window', ImgelonTest)
cv2.imshow('wind', Imgelon)
cv2.waitKey()
'''
