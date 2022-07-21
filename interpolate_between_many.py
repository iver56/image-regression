import argparse
import os

import math
import numpy as np
import onnxruntime
import torch
from PIL import Image
from tqdm import tqdm

from utils.gif import make_gif

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--num-images", help="Number of images", dest="num_images", type=int, default=1
    )
    arg_parser.add_argument("--width", help="Image width", dest="image_width", type=int)
    arg_parser.add_argument(
        "--height", help="Image height", dest="image_height", type=int
    )
    arg_parser.add_argument(
        "--steps-per-image",
        help="Number of interpolations between each image",
        dest="steps_per_image",
        type=int,
        default=30,
    )
    arg_parser.add_argument(
        "--model",
        help="Filename of model",
        dest="model_filename",
        type=str,
        required=True,
    )
    arg_parser.add_argument(
        "--ordering",
        dest="index_map",
        nargs="+",
        type=int,
        help="List of indexes that denote the order the images should appear in",
    )
    arg_parser.add_argument(
        "--images",
        dest="image_filenames",
        nargs="+",
        type=str,
        help="Image filenames if you want to include originals in the series",
    )
    arg_parser.add_argument(
        "--use-cuda",
        dest="use_cuda",
        default=1,
        type=int,
        help="Use CUDA (GPU) or not?",
    )
    args = arg_parser.parse_args()

    batch_size = 128

    index_map = args.index_map if args.index_map else list(range(args.num_images))

    image_shape = (args.image_height, args.image_width)

    model_file_path = os.path.join("models", args.model_filename)
    model = onnxruntime.InferenceSession(model_file_path)
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    image_datasets = []

    use_cuda = args.use_cuda
    if use_cuda and not torch.cuda.is_available():
        print("Warning: Trying to use CUDA, but it is not available")
        use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    output_dir = os.path.join("interpolated_output", args.model_filename)
    os.makedirs("interpolated_output", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    def smoothstep(minimum, maximum, value):
        that_x = max(0, min(1, (value - minimum) / (maximum - minimum)))
        return that_x * that_x * (3 - 2 * that_x)

    image_counter = 0
    for k in tqdm(range(-1, args.num_images)):
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
                current_index = index_map[k]
                one_hot_vector[current_index] = current_value
            next_index = index_map[(k + 1) % args.num_images]
            one_hot_vector[next_index] = next_value
            for i in range(args.image_height):
                for j in range(args.image_width):
                    vector = [i, j] + one_hot_vector
                    x.append(vector)

            x = np.array(x, dtype=np.float32)

            predicted_pixels = np.zeros(
                shape=(image_shape[0] * image_shape[1], 1),
                dtype=np.float32,
            )
            with torch.no_grad():
                for offset in range(0, x.shape[0], batch_size):
                    pred = model.run(
                        [output_name], {input_name: x[offset : offset + batch_size]}
                    )[0]
                    predicted_pixels[offset : offset + pred.shape[0]] = pred

            predicted_image = predicted_pixels.reshape(image_shape)

            output_file_path = os.path.join(
                output_dir, "{0:03d}_interpolated.png".format(image_counter)
            )

            if l == 0 and k >= 0 and args.image_filenames:
                filename = args.image_filenames[index_map[k]]
                Image.open(filename).save(output_file_path)
            else:
                Image.fromarray(
                    np.clip(predicted_image * 256, 0, 255).astype(np.uint8)
                ).save(output_file_path)
            image_counter += 1

    make_gif(output_dir)
