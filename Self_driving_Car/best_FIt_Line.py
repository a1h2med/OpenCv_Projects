import cv2 as cv
import numpy as np
import dlib as dl
import matplotlib.pyplot as plt
Part29 = "Best Fit Line"
#Car_image = cv.imread('C:/Users/LENOVO/Desktop/Image/test_image.jpg')

def draw_Line(x1,x2):
    ln = plt.plot(x1, x2)
    plt.pause(.0001)
    ln[0].remove()
def sigmoid(parameter):
    return 1/(1+np.exp(-parameter))

def Calculating_error (Prob,m , y):
    cross_entropy = -(1 / m) * ((np.log(Prob).T) * y +  (np.log(1-Prob).T) * (1 - y))
    return cross_entropy

def gradient_desent(line_parameters , all_points, y , alpha):
    m = all_points.shape[0]
    for i in range(500):
        P = sigmoid(all_points * line_parameters)
        gradient = (alpha/m) * (all_points.T * (P - y))
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([all_points[:,0].min() ,all_points[:,0].max()])
        x2 = -b/w2 + x1*(-w1)/w2
    draw_Line(x1, x2)

N_pts = 100
np.random.seed(0)
random_x1 = np.random.normal(10,2,N_pts)
random_x2 = np.random.normal(12,2,N_pts)
bias = np.ones(N_pts)
top_region = np.array([random_x1,random_x2,bias]).T
bottom_region = np.array([np.random.normal(5,2,N_pts),np.random.normal(6,2,N_pts),bias]).T

# we commented those as it's not our job to put them
# w1 = -.2
# b = 5
# w2 = -.35
# w1 * x1 + w2 * x2 + B = 0
#x1 = np.array([bottom_region[:,0].min() ,top_region[:,0].max()])
#x2 = -b/w2 + x1*(-w1)/w2


line_parameters = np.matrix([np.zeros(3)]).T
all_points = np.vstack([bottom_region,top_region])


# we will multiply line_parameters with all_points as
# (all_points) contains x1,x2 values, (line_parameters) contains the weights
# So if we multiplied them we will get (w1 * x1 + w2 * x2 + B ) which is the chracterstic equation for us which defines
# Is it above the line or Not
Linear_Compination = all_points*line_parameters
probability = sigmoid(Linear_Compination)
# we made that zeros as in the above I know that it's diabetic
# and in the bottom is one as I know that in the bottom it's not so it should be multiplied by one
y = np.array([np.ones(N_pts),np.zeros(N_pts)]).reshape(N_pts*2 ,1)

error = Calculating_error(probability,all_points.shape[0],y)

_,axs = plt.subplots(figsize = (4,4))
axs.scatter(top_region[:,0],top_region[:,1],color = 'r')
axs.scatter(bottom_region[:,0],bottom_region[:,1],color = 'b')
gradient_desent(line_parameters,all_points,y,.06)
plt.show()