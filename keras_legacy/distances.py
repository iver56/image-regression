from __future__ import division
import argparse

from skimage.io import imread
import numpy as np
from tsp_solver.greedy import solve_tsp

"""
Naively calculate a short path through the images
"""


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-i",
    dest="image_filenames",
    nargs="+",
    type=str,
    help="File names of input images",
    required=True,
)
args = arg_parser.parse_args()

images = []
for image_filename in args.image_filenames:
    image = imread(image_filename, as_grey=True, plugin="pil")
    if str(image.dtype) == "uint8":
        image = np.divide(image, 255.0)
    images.append(image)

num_images = len(images)

differences = np.zeros((num_images, num_images))

for i, image in enumerate(images):
    for j in range(i, len(images)):
        other_image = images[j]
        difference = ((image - other_image) ** 2).sum()
        differences[i, j] = difference
        differences[j, i] = difference

differences_matrix = differences.tolist()

path = solve_tsp(differences_matrix)
print(path)

ordered_image_filenames = [args.image_filenames[i] for i in path]
for filename in ordered_image_filenames:
    print(filename)
