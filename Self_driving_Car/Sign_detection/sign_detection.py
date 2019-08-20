import cv2 as cv
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten, Dropout, Dense
from keras.layers.convolutional import Conv2D ,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random
import pickle
import pandas as pd

#Car_image = cv.imread('C:/Users/LENOVO/Desktop/Image/test_image.jpg')

def Lenet_model():
    model = Sequential()
    model.add(Conv2D(60,(5,5),input_shape=(32,32,1),activation='relu'))
    model.add(Conv2D(60, (5, 5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(30,(3,3),activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(.5))

    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(43,activation='softmax'))
    model.compile(Adam(lr=.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def gray_cvt(img):
    return cv.cvtColor(img,cv.COLOR_BGR2GRAY)

def Hist_equalization(img):
    return cv.equalizeHist(img)

def preprocessing(img):
    image = gray_cvt(img)
    final_image = Hist_equalization(image)
    final_image = final_image/255
    return final_image

np.random.seed(0)

with open('master/test.p','rb') as f:
    test_data = pickle.load(f)

with open('master/valid.p','rb') as f:
    val_data = pickle.load(f)

with open('master/train.p','rb') as f:
    train_data = pickle.load(f)

x_train, y_train = train_data['features'], train_data['labels']
x_val, y_val = val_data['features'], val_data['labels']
x_test, y_test = test_data['features'], test_data['labels']

assert (x_train.shape[0] == y_train.shape[0]), "error in the training"
assert (x_val.shape[0] == y_val.shape[0]), "error in the validation data"
assert (x_test.shape[0] == y_test.shape[0]), "error in the testing data"
assert (x_train.shape[1:] == (32,32,3)), "error in the training image shape"
assert (x_val.shape[1:] == (32,32,3)), "error in the validation image shape"
assert (x_test.shape[1:] == (32,32,3)), "error in the testing image shape"

data = pd.read_csv('master/signnames.csv')

n_cls = 5
n_rows = 43
num_of_samples = []

fig, axs = plt.subplots(nrows=n_rows,ncols=n_cls,figsize=(5,50))
fig.tight_layout()
for i in range(n_cls):
    for j,row in data.iterrows():
        x_selected = x_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0,len(x_selected)-1),:,:],cmap= plt.get_cmap('gray'))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j)+"-"+ row["SignName"])


x_train = np.array(list(map(preprocessing,x_train)))
x_val = np.array(list(map(preprocessing,x_val)))
x_test = np.array(list(map(preprocessing,x_test)))

x_train = x_train.reshape(x_train.shape[0],32,32,1)
x_test = x_test.reshape(x_test.shape[0],32,32,1)
x_val = x_val.reshape(x_val.shape[0],32,32,1)

y_train = to_categorical(y_train,43)
y_val = to_categorical(y_val,43)
y_test = to_categorical(y_test,43)

model = Lenet_model()

datagen = ImageDataGenerator(width_shift_range=.1 , height_shift_range=.1 , rotation_range=10 , shear_range=.1 , zoom_range=.2)
datagen.fit(x_train)
batches = datagen.flow(x_train,y_train,batch_size=50)
x_batch, y_batch = next(batches)

model.fit_generator(batches,steps_per_epoch=2000,epochs=10,validation_data=(x_val,y_val),shuffle=1)
score = model.evaluate(x_test,y_test)
print("test score",score[0])
print("test accuracy",score[1])

sign1 = cv.imread('C:/Users/LENOVO/Desktop/test1')
img = np.asarray(sign1)
img = cv.resize(img, (32, 32))
img = preprocess(img)
img = img.reshape(1, 32, 32, 1)
print("predicted sign1: "+ str(model.predict_classes(img)))

sign2 = cv.imread('C:/Users/LENOVO/Desktop/test2')
img = np.asarray(sign2)
img = cv.resize(img, (32, 32))
img = preprocess(img)
img = img.reshape(1, 32, 32, 1)
print("predicted sign2: "+ str(model.predict_classes(img)))


sign3 = cv.imread('C:/Users/LENOVO/Desktop/test3')
img = np.asarray(sign3)
img = cv.resize(img, (32, 32))
img = preprocess(img)
img = img.reshape(1, 32, 32, 1)
print("predicted sign3: "+ str(model.predict_classes(img)))


sign4 = cv.imread('C:/Users/LENOVO/Desktop/test4')
img = np.asarray(sign2)
img = cv.resize(img, (32, 32))
img = preprocess(img)
img = img.reshape(1, 32, 32, 1)
print("predicted sign4: "+ str(model.predict_classes(img)))


sign5 = cv.imread('C:/Users/LENOVO/Desktop/test5')
img = np.asarray(sign2)
img = cv.resize(img, (32, 32))
img = preprocess(img)
img = img.reshape(1, 32, 32, 1)
print("predicted sign5: "+ str(model.predict_classes(img)))
