from __future__ import division
import os
import warnings
import argparse
import math

from keras.models import load_model
from skimage.io import imsave
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
    '--width',
    help='Image width',
    dest='image_width',
    type=int,
    default=128
)
arg_parser.add_argument(
    '--height',
    help='Image height',
    dest='image_height',
    type=int,
    default=128
)
arg_parser.add_argument(
    '--model',
    help='Filename of model',
    dest='model_filename',
    type=str,
    required=True
)
args = arg_parser.parse_args()

image_shape = (args.image_height, args.image_width)

model = load_model(args.model_filename)

image_datasets = []
steps_per_image = 10


def smoothstep(minimum, maximum, value):
    that_x = max(0, min(1, (value - minimum) / (maximum - minimum)))
    return that_x * that_x * (3 - 2 * that_x)


for k in range(args.num_images):
    for l in range(steps_per_image):
        x = []
        one_hot_vector = [0] * args.num_images
        progress = l / steps_per_image
        progress = smoothstep(0, 1, progress)
        progress = smoothstep(0, 1, progress)

        current_value = 1 - progress
        current_value = (math.sqrt(current_value) + current_value) / 2.0
        next_value = (math.sqrt(progress) + progress) / 2.0
        one_hot_vector[k] = current_value
        one_hot_vector[(k + 1) % args.num_images] = next_value
        for i in range(args.image_height):
            for j in range(args.image_width):
                coordinate_y = 2 * (i / (args.image_height - 1) - 0.5)
                coordinate_x = 2 * (j / (args.image_width - 1) - 0.5)
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
