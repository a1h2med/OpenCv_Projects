import cv2 as cv
import numpy as np
import keras
# It's used for creating layers
from keras.models import Sequential
# dense layer is used for creating a fully connected perceptron
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#Car_image = cv.imread('C:/Users/LENOVO/Desktop/Image/test_image.jpg')

def plot_decision_boundary(X,y,model):
    x_span = np.linspace(min(X[:,0]) - 1,max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1,max(X[:,1]) + 1)
    xx , yy = np.meshgrid(x_span,y_span)
    XX , YY = xx.ravel() , yy.ravel()
    grid = np.c_[XX,YY]
    predict = model.predict(grid)
    z = predict.reshape(xx.shape)
    plt.contourf(xx,yy,z)

n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T

X = np.vstack((Xa, Xb))
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

model = Sequential()
model.add(Dense(units=1,input_shape=(2,),activation='sigmoid'))
adam = Adam(lr=.1)
model.compile(adam , loss='binary_crossentropy' , metrics=['accuracy'] )
model.fit(x=X,y=y,verbose=1,batch_size=50,epochs=500,shuffle='true')
#plt.plot(h.history['acc'])

x = 7.5
y = 7.5
point = np.array([[x , y]])
prediction = model.predict(point)
print(prediction)
plot_decision_boundary(X,y,model)

plt.plot([x],[y],marker= "o", color = "red")

plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

plt.show()