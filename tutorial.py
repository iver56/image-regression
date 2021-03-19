# This is the code from the tutorial at https://github.com/iver56/image-regression/wiki/Tutorial
import numpy as np

image = [[0, 130, 255], [40, 170, 255], [80, 210, 255]]
image = np.array(image)
image = np.divide(image, 255.0)
image_width, image_height = image.shape
print("Image with shape {0}:".format(image.shape))
print(image)

x = []
y = []
for i in range(image_height):
    for j in range(image_width):
        x.append([i / image_height, j / image_width])
        y.append([image[i][j]])
x = np.array(x)
y = np.array(y)

print("\nScaled coordinates (input):")
print(x)

print("\nScaled pixel brightness values (output):")
print(y)

from keras.models import Sequential
from keras.layers import Activation, Dense

model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("relu"))

model.compile(loss="mean_squared_error", optimizer="sgd")

model.fit(x, y, epochs=500)
# epochs=500 means it'll sweep the data 500 times during the training process
# The loss should go down from around [0.5, 0.3] to somewhere around [0.004, 0.01]

predicted_image = model.predict(x, verbose=False).reshape(image.shape)
print("\nPredicted image:")
print(predicted_image)
