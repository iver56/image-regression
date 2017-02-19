from __future__ import division
from skimage.io import imread, imsave
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import Callback
import numpy as np
import warnings
import os
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i', help='File name of input image', dest='input_filename', type=str, default='keyboard.png')
arg_parser.add_argument('--num-epochs', help='Number of epochs', dest='num_epochs', type=int, default=1000)
args = arg_parser.parse_args()

image = imread(args.input_filename, as_grey=True, plugin='pil')
if str(image.dtype) == 'uint8':
    image = np.divide(image, 255.0)
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
model.add(Dense(30, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop'
)


class CheckpointOutputs(Callback):
    def __init__(self):
        super(CheckpointOutputs, self).__init__()
        self.last_loss_checkpoint = 9001  # it's over 9000!
        self.loss_change_threshold = 0.05

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        loss_change = 1 - logs['loss'] / self.last_loss_checkpoint

        if loss_change > self.loss_change_threshold:
            self.last_loss_checkpoint = logs['loss']
            predicted_image = self.model.predict(x, verbose=False)
            predicted_image = np.clip(predicted_image, 0, 1)
            predicted_image = predicted_image.reshape(image.shape)

            with warnings.catch_warnings():
                output_file_path = os.path.join(
                    'output',
                    '{0}_predicted_{1:04d}.png'.format(args.input_filename, epoch)
                )
                imsave(output_file_path, predicted_image)


checkpoint_outputs = CheckpointOutputs()
history = model.fit(
    x,
    y,
    batch_size=128,
    nb_epoch=args.num_epochs,
    shuffle=True,
    callbacks=[checkpoint_outputs]
)
