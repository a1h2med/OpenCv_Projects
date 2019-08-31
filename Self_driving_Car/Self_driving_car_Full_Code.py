import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D,Convolution2D,MaxPooling2D,Dense,Flatten,Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2 as cv
import pandas as pd
import random
import ntpath

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail


def load_img_steering(datadir,df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center , left , right = indexed_data[0] , indexed_data[1] , indexed_data[2]
    image_path.append(os.path.join(datadir,center.strip()))
    steering.append(float(indexed_data[3]))
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths ,steerings


def zoom(image):
  zoom = iaa.Affine(scale=(1,1.3))
  image = zoom.augment_image(image)
  return image

def pan(image):
  pan = iaa.Affine(translate_percent= {"x":(-.1,.1),"y":(-.1,.1)})
  image = pan.augment_image(image)
  return image

def image_brightness(image):
  brightness = iaa.Multiply((.2,1.2))
  image = brightness.augment_image(image)
  return image

def image_flip(image, steering_angle):
  image = cv.flip(image, 1)                       # 0 for vertical flipping, 1 for horizontal , -1 for both, but actually here we need horizontal flipping
  steering_angle = -steering_angle
  return image,steering_angle


def random_augment(image, steering_angle):
    image = npimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = image_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = image_flip(image, steering_angle)

    return image, steering_angle

def img_preprocess(image):
  image = image[60:130,:,:]
  image = cv.cvtColor(image,cv.COLOR_RGB2YUV)
  image = cv.GaussianBlur(image,(3,3), 0)
  image = cv.resize(image, (200,66))
  image = image/255
  return image


def batch_generator(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)

            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])

            else:
                im = npimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]

            im = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))


def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))

    # model.add(Dropout(.5))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))

    # model.add(Dropout(.5))

    model.add(Dense(15, activation='elu'))

    # model.add(Dropout(.5))

    model.add(Dense(10, activation='elu'))

    # model.add(Dropout(.5))

    model.add(Dense(1))
    model.compile(Adam(lr=.0001), loss='mse')
    return model




datadir = 'DATA'
columns = ['center','left','right','steering','throttle','reverse','speed']
data = pd.read_csv(os.path.join(datadir,'DATA/driving_log.csv'), names = columns)
pd.set_option('display.max_colwidth',-1)
data.head()

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()

num_bins = 25
hist, bins = np.histogram(data['steering'],num_bins)
center = (bins[:-1]+bins[1:]) * .5
plt.bar(center,hist,width=.05)

print("total_data= ",len(data))
remove_list = []
for i in range (num_bins):
  List = []
  for j in range (len(data['steering'])):
    if data['steering'][j] >= bins[i] and data['steering'][j] <= bins[i+1]:
      List.append(j)
  List = shuffle(List)
  List = List[400:]
  remove_list.extend(List)

print("removed",len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print("remaining",len(data))

hist, _ = np.histogram(data['steering'],num_bins)
plt.bar(center,hist,width=.05)

print(data.iloc[1])
image_paths, steerings = load_img_steering(datadir + '/DATA/IMG',data)

x_train, x_val, y_train, y_val = train_test_split(image_paths,steerings,test_size=.25,random_state = 6)
_, axs = plt.subplots(1, 2 ,figsize = (12,4))
axs[0].hist(y_train,bins=num_bins,width=.05,color='red')
axs[1].hist(y_val,bins=num_bins,width=.05,color='blue')

image = image_paths[random.randint(0,1000)]
original_image = npimg.imread(image)
zoomed_image = zoom(original_image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original')
axs[1].imshow(zoomed_image)
axs[1].set_title('zoomed')

image = image_paths[random.randint(0,1000)]
original_image = npimg.imread(image)
panned_image = pan(original_image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original')
axs[1].imshow(panned_image)
axs[1].set_title('panned')

image = image_paths[random.randint(0,1000)]
original_image = npimg.imread(image)
brightness_image = image_brightness(original_image)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original')
axs[1].imshow(brightness_image)
axs[1].set_title('brightness')

random_index = random.randint(0,1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]
original_image = npimg.imread(image)
flipped_image, flipped_steering_angle = image_flip(original_image,steering_angle)
fig,axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('original')
axs[1].imshow(flipped_image)
axs[1].set_title('flipped')

image = image_paths[160]
original_image = npimg.imread(image)
preprocessed_image = img_preprocess(original_image)

fig, axs = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title("original")
axs[1].imshow(preprocessed_image)
axs[1].set_title("preprocessed")

random_index = random.randint(0, len(image_paths) - 1)
image = image_paths[random_index]
image = npimg.imread(image)

x_train_gen, y_train_gen = next(batch_generator(x_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(x_val, y_val, 1, 0))

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(x_train_gen[0])
axs[0].set_title('Training Image')

axs[1].imshow(x_valid_gen[0])
axs[1].set_title('Validation Image')

#x_train = np.array(list(map(img_preprocess,x_train)))
#x_val = np.array(list(map(img_preprocess,x_val)))

model = nvidia_model()
print(model.summary())

history = model.fit_generator(batch_generator(x_train,y_train,batch_size=100 ,istraining=1) ,steps_per_epoch=300 ,epochs=10,validation_data=batch_generator(x_val,y_val,100,0),validation_steps=200,verbose=1,shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epochs')