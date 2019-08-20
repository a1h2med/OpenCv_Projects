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
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn import datasets
import random

#Car_image = cv.imread('C:/Users/LENOVO/Desktop/Image/test_image.jpg')
def Lenet_model():
    model = Sequential()
    model.add(Conv2D(30,(5,5),input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(10,activation='softmax'))
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
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
model = Lenet_model()
model.fit(x_train,y_train,epochs=10,validation_split=.1,verbose=1,batch_size=400,shuffle=1)

img = cv.imread('C:/Users/LENOVO/Desktop/Handwritten-digit-2.png')

img = cv.resize(img,(28,28))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.bitwise_not(img)

img = img/255
img = img.reshape(1, 28,28,1)

prediction = model.predict_classes(img)
print("predicted digit:", str(prediction))
