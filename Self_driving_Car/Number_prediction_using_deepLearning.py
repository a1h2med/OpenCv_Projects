import cv2 as cv
import numpy as np
import keras

# It's used for creating layers
from keras.models import Sequential

from keras.datasets import mnist
# dense layer is used for creating a fully connected perceptron
from keras.layers import Dense
from keras.optimizers import Adam

# it's used to show the hot encoded of the input labels
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import datasets
#import requests
import random
#Car_image = cv.imread('C:/Users/LENOVO/Desktop/Image/test_image.jpg')

def create_model():
    input_pixels = 28*28
    Number_of_Col = 10
    model = Sequential()
    model.add(Dense(10,input_dim=input_pixels,activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(Number_of_Col, activation='relu'))
    model.compile(Adam(lr=.01),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

np.random.seed(0)
(x_train , y_train), (x_test,y_test) = mnist.load_data()
assert (x_train.shape[0] == y_train.shape[0]),"there's a difference between the labels and the trained data"
assert (x_test.shape[0] == y_test.shape[0]),"there's a difference between the labels and the tested data"
assert (x_train.shape[1:] == (28,28)), "the shape of the training data is not 28*28"
assert (x_test.shape[1:] == (28,28)), "the shape of the testing data is not 28*28"
num_of_samples = []
cols = 5
num_classes = 10
fig , axs = plt.subplots(nrows=num_classes,ncols=cols,figsize= (5,10))
for i in range(cols):
    for j in range(num_classes):
        x_selected = x_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0,len(x_selected - 1)),:,:],cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")

y_train = to_categorical(y_train , 10)
y_test = to_categorical(y_test , 10)
x_test = x_test/255
x_train = x_train/255
num_pixels = 28*28
x_train = x_train.reshape(x_train.shape[0],num_pixels)
x_test = x_test.reshape(x_test.shape[0],num_pixels)

model = create_model()
model.fit(x_train,y_train,validation_split=.1,epochs=10,batch_size=200,verbose=1,shuffle=1)
image = cv.imread('C:/Users/LENOVO/Desktop/Handwritten-digit-2.png')
resized = cv.resize(image,(28,28))
gray = cv.cvtColor(resized,cv.COLOR_RGB2GRAY)
final_image = cv.bitwise_not(gray)

final_image = final_image/255
final_image = final_image.reshape(1,28*28)
prediction = model.predict_classes(final_image)
print(str(prediction))
plt.show()