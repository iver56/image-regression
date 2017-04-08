from __future__ import division
import os
import warnings
import argparse
import sys

from keras.models import load_model
from skimage.io import imread, imsave
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--num-images',
    help='Number of images',
    dest='num_images',
    type=int,
    default=12
)
arg_parser.add_argument(
    '--model',
    help='Filename of model',
    dest='model_filename',
    type=str,
    required=True
)
args = arg_parser.parse_args()

image_shape = (256, 256)
img_height, img_width = image_shape

model = load_model(args.model_filename)

image_datasets = []
steps_per_image = 10

for k in range(args.num_images):
    for l in range(steps_per_image):
        x = []
        one_hot_vector = [0] * args.num_images
        progress = l / steps_per_image
        current_value = 1 - progress
        next_value = progress
        one_hot_vector[k] = current_value
        if k < args.num_images - 1:
            one_hot_vector[k + 1] = next_value
        if k == args.num_images - 1 and l > 0:
            continue
        for i in range(img_height):
            for j in range(img_width):
                coordinate_y = 2 * (i / (img_height - 1) - 0.5)
                coordinate_x = 2 * (j / (img_width - 1) - 0.5)
                vector = [coordinate_y, coordinate_x] + one_hot_vector
                x.append(vector)

        x = np.array(x)

        predicted_image = model.predict(x, verbose=False)
        predicted_image = np.clip(predicted_image, 0, 1)
        predicted_image = predicted_image.reshape(image_shape)

        with warnings.catch_warnings():
            output_file_path = os.path.join(
                'output',
                '{0}_{1}_interpolated.png'.format(k, l)
            )
            imsave(output_file_path, predicted_image)
