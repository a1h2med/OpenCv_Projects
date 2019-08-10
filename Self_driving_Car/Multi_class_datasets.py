import cv2 as cv
import numpy as np
import keras

# It's used for creating layers
from keras.models import Sequential

# dense layer is used for creating a fully connected perceptron
from keras.layers import Dense
from keras.optimizers import Adam

# it's used to show the hot encoded of the input labels
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn import datasets

#Car_image = cv.imread('C:/Users/LENOVO/Desktop/Image/test_image.jpg')


def plot_decision_boundary(X,y_cat,model):
    x_span = np.linspace(min(X[:,0]) - 1,max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1,max(X[:,1]) + 1)
    xx , yy = np.meshgrid(x_span,y_span)
    XX , YY = xx.ravel() , yy.ravel()
    grid = np.c_[XX,YY]
    predict = model.predict_classes(grid)
    z = predict.reshape(xx.shape)
    plt.contourf(xx,yy,z)

N_pts = 500
centers = [[-1,1] , [-1,-1] , [1,-1]]
X , y = datasets.make_blobs(N_pts ,random_state=123 ,centers=centers ,cluster_std= .4)
y_cat = to_categorical(y,3)
model = Sequential()
model.add(Dense(units=3,input_shape=(2,),activation='softmax'))
model.compile(Adam(lr=0.1),loss='categorical_crossentropy',metrics=['accuracy'])
h = model.fit(x=X,y=y_cat,verbose=1,batch_size=50,epochs=100)
#print(y_cat)
plot_decision_boundary(X,y_cat,model)
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.scatter(X[y==2,0],X[y==2,1])
plt.show()