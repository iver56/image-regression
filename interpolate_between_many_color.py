from __future__ import division
import os
import warnings
import argparse
import math

from keras.models import load_model
from skimage.io import imsave, imread
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--num-images',
    help='Number of images',
    dest='num_images',
    type=int,
    default=16
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
    '--steps-per-image',
    help='Number of interpolations between each image',
    dest='steps_per_image',
    type=int,
    default=30
)
arg_parser.add_argument(
    '--model',
    help='Filename of model',
    dest='model_filename',
    type=str,
    required=True
)
arg_parser.add_argument(
    '--ordering',
    dest='index_map',
    nargs='+',
    type=int,
    help='Indexes of the order the images should appear in',
    default=[8, 12, 11, 7, 3, 4, 13, 2, 5, 14, 0, 1, 15, 6, 9, 10]
)
arg_parser.add_argument(
    '--images',
    dest='image_filenames',
    nargs='+',
    type=str,
    help='Image filenames if you want to include originals in the series',
    default=['darklite.png', 'desire.png ', 'farbrausch.png ', 'gargaj.png ', 'idle.png ',
             'kvasigen.png ', 'lft.png ', 'logicoma.png ', 'mercury.png ', 'mrdoob.png ',
             'outracks.png ', 'pandacube.png ', 'revision.png ', 'rohtie.png ', 'sandsmark.png ',
             't-101.png']
)
arg_parser.add_argument(
    '--num-channels',
    help='Number of channels in the output images',
    dest='num_channels',
    type=int,
    default=4
)
args = arg_parser.parse_args()

image_shape = (args.image_height, args.image_width, args.num_channels)

model = load_model(args.model_filename)

image_datasets = []


def smoothstep(minimum, maximum, value):
    that_x = max(0, min(1, (value - minimum) / (maximum - minimum)))
    return that_x * that_x * (3 - 2 * that_x)


image_counter = 0
for k in range(-1, args.num_images):
    for l in range(args.steps_per_image):
        x = []
        one_hot_vector = [0] * args.num_images
        progress = l / args.steps_per_image
        progress = smoothstep(0, 1, progress)
        progress = smoothstep(0, 1, progress)

        current_value = 1 - progress
        current_value = 0.7 * math.sqrt(current_value) + 0.3 * current_value
        next_value = 0.7 * math.sqrt(progress) + 0.3 * progress

        if k >= 0:
            current_index = args.index_map[k]
            one_hot_vector[current_index] = current_value
        next_index = args.index_map[(k + 1) % args.num_images]
        one_hot_vector[next_index] = next_value
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
                '{0:03d}_interpolated.png'.format(image_counter)
            )
            if l == 0 and k >= 0:
                filename = args.image_filenames[args.index_map[k]]
                original_image = imread(filename, as_grey=False, plugin='pil')
                imsave(output_file_path, original_image)
            else:
                imsave(output_file_path, predicted_image)
            image_counter += 1
