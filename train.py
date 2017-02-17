from __future__ import division
from skimage.io import imread, imsave
from keras.models import Sequential
from keras.layers import Activation, Dense
import numpy as np

file_path = 'keyboard.png'
image = imread(file_path, as_grey=True)
img_width, img_height = image.shape


x = []
y = []
for i in range(img_height):
    for j in range(img_width):
        x.append([i / img_height, j / img_width])
        y.append([image[i][j]])
x = np.array(x)
y = np.array(y)

model = Sequential()
model.add(Dense(20, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop'
)

history = model.fit(
    x,
    y,
    batch_size=64,
    nb_epoch=300,
    shuffle=True
)
print(history)

predicted_image = np.copy(image)
for i in range(img_height):
    for j in range(img_width):
        predicted_image[i][j] = model.predict_proba(
            np.array([[i / img_height, j / img_width]]),
            verbose=False
        )[0]
predicted_image = np.clip(predicted_image, 0, 1)

imsave('keyboard_predicted.png', predicted_image)
