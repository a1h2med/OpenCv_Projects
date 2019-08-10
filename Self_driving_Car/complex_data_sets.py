import cv2 as cv
import numpy as np
import keras

# It's used for creating layers
from keras.models import Sequential

# dense layer is used for creating a fully connected perceptron
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn import datasets

#Car_image = cv.imread('C:/Users/LENOVO/Desktop/Image/test_image.jpg')

def plot_decision_boundary(X,y,model):
    x_span = np.linspace(min(X[:,0]),max(X[:,0]))
    y_span = np.linspace(min(X[:,1]),max(X[:,1]))
    xx , yy = np.meshgrid(x_span,y_span)
    XX , YY = xx.ravel(), yy.ravel()
    grid = np.c_[XX,YY]
    prediction = model.predict(grid)
    z = prediction.reshape(xx.shape)
    # there's no difference between contourf and contour but in the appearence
    plt.contourf(xx,yy,z)

n_pts = 500
np.random.seed(0)

X , y = datasets.make_circles(n_pts,noise=.1,random_state=123,factor=.2)
model = Sequential()
model.add(Dense(4,input_shape=(2,),activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# loss function is mainly used to backproject the network
# metrics is used to check how the output accuracy matches our needs
model.compile(Adam(lr=.01),'binary_crossentropy',metrics=['accuracy'])
h = model.fit(x=X,y=y,verbose=1,batch_size=20,epochs=150,shuffle='true')

plot_decision_boundary(X,y,model)
plt.scatter(X[:n_pts, 0], X[:n_pts,1])
plt.scatter(X[n_pts:, 0], X[n_pts:,1])
#plt.scatter(X[y==0, 0], X[y==0,1])
#plt.scatter(X[y==1, 0], X[y==1,1])

plt.show()