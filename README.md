# Image regression

A toy application that learns a mapping from (x, y) coordinates to color. Uses pytorch and pytorch-lightning.

## Setup

Set up your python environment with conda like this:

`conda env create`

## Usage

### Train on a single greyscale image

`python train.py -i keyboard.png --num-epochs 150`

### Train on a single color image (RGBA)

TODO

### Train on multiple images

`python train.py -i boxy_stripes2.png boxy_stripes2_30.png boxy_stripes2_60.png boxy_stripes2_90.png`

### Train on multiple color images (RGBA)

TODO

### Interpolate between the images:

`python interpolate_between_many.py --width 64 --height 64 --num-images 4 --model boxy_stripes2_boxy_stripes2_30_boxy_stripes2_60_bo_2f59b848.onnx --use-cuda 0`

## Examples

| Name | Original | Learned image |
| ---- | -------- | ------------- |
| Keyboard | ![Original image](input_images/keyboard.png) | ![Learned image](demo/keyboard-learned.gif) |
| 8x8 Checkerboard | ![Original image](input_images/chess.png) | ![Learned image](demo/chess-learned.gif) |

The following animation visualizes the output of a neural network that was trained on 12 different images (different rotations of boxy stripes). The input vectors are constructed to interpolate between the 12 images, so we get a kind of morphing effect.

![Boxy stripes](demo/boxy_stripes_interpolation.gif)
