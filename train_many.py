from __future__ import division
import os
import warnings
import argparse
import hashlib
import json

from skimage.io import imread, imsave
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import Callback
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '-i',
    dest='image_filenames',
    nargs='+',
    type=str,
    help='File names of input images',
    default=['keyboard.png']
)
arg_parser.add_argument(
    '--num-epochs',
    help='Number of epochs',
    dest='num_epochs',
    type=int,
    default=750
)
args = arg_parser.parse_args()

images = []
for image_filename in args.image_filenames:
    image = imread(image_filename, as_grey=True, plugin='pil')
    if str(image.dtype) == 'uint8':
        image = np.divide(image, 255.0)
    images.append(image)

img_height, img_width = images[0].shape
# check that all images have the same dimensions
for image in images:
    assert image.shape == images[0].shape

image_filenames_hash = hashlib.md5(json.dumps(args.image_filenames).encode('utf-8')).hexdigest()[:8]
num_images = len(images)

x = []
y = []
image_datasets = []
for k, image in enumerate(images):
    image_dataset_x = []
    image_dataset_y = []
    one_hot_vector = [0] * num_images
    one_hot_vector[k] = 1
    for i in range(img_height):
        for j in range(img_width):
            coordinate_y = 2 * (i / (img_height - 1) - 0.5)
            coordinate_x = 2 * (j / (img_width - 1) - 0.5)
            vector = [coordinate_y, coordinate_x] + one_hot_vector
            image_dataset_x.append(vector)
            image_dataset_y.append([image[i][j]])

    x += image_dataset_x
    y += image_dataset_y
    image_dataset_x = np.array(image_dataset_x)
    image_dataset_y = np.array(image_dataset_y)
    image_datasets.append(
        (
            image_dataset_x,
            image_dataset_y
        )
    )

x = np.array(x)
y = np.array(y)

model = Sequential()
model.add(Dense(150, input_dim=2 + num_images))
model.add(Activation('relu'))
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dense(150))
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
        self.last_loss_checkpoint = 9001.0  # it's over 9000!
        self.loss_change_threshold = 0.05

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        loss_change = 1.0 - logs['loss'] / self.last_loss_checkpoint

        if loss_change > self.loss_change_threshold:
            self.last_loss_checkpoint = logs['loss']
            model_file_path = os.path.join(
                'output',
                'train_many_{0}_{1:04d}.h5'.format(image_filenames_hash, epoch)
            )
            model.save(model_file_path)
            for k, image_filename in enumerate(args.image_filenames):

                that_x = image_datasets[k][0]
                predicted_image = self.model.predict(that_x, verbose=False)
                predicted_image = np.clip(predicted_image, 0, 1)
                predicted_image = predicted_image.reshape(image.shape)

                with warnings.catch_warnings():
                    output_file_path = os.path.join(
                        'output',
                        '{0}_predicted_{1:04d}.png'.format(image_filename, epoch)
                    )
                    imsave(output_file_path, predicted_image)


checkpoint_outputs = CheckpointOutputs()
history = model.fit(
    x,
    y,
    batch_size=128,
    epochs=args.num_epochs,
    shuffle=True,
    callbacks=[checkpoint_outputs]
)
