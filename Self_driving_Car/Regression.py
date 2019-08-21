import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(0)
points = 500
x = np.linspace(-3, 3, points)
y = np.sin(x) + np.random.uniform(-.5,.5,points)
plt.scatter(x,y)

model = Sequential()
model.add(Dense(50,input_dim= 1,activation= 'sigmoid'))
model.add(Dense(30,activation= 'sigmoid'))
model.add(Dense(1))
model.compile(Adam(lr=.01),loss='mse')
model.fit(x,y,epochs= 50)

predictions = model.predict(x)
plt.scatter(x,y)
plt.plot(x,predictions,'ro')
plt.show