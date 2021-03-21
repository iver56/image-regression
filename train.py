from __future__ import division

import argparse
import hashlib
import json
import os
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from siren_pytorch import Siren
from skimage.io import imread
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset, DataLoader

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-i",
    dest="image_filenames",
    nargs="+",
    type=str,
    help="File name(s) of input image(s)",
    default=["keyboard.png"],
)
arg_parser.add_argument(
    "--num-epochs", help="Number of epochs", dest="num_epochs", type=int, default=750
)
args = arg_parser.parse_args()

images = []
dx_images = []
dy_images = []
for image_filename in args.image_filenames:
    image = imread(image_filename, as_grey=True, plugin="pil")
    if str(image.dtype) == "uint8":
        image = np.divide(image, 255.0)
    images.append(image)

    image_shifted_right = np.copy(image)
    image_shifted_right[:, 1:] = image[:, :-1]
    image_shifted_down = np.copy(image)
    image_shifted_down[1:, :] = image[:-1, :]

    dx_gt = image - image_shifted_right
    dx_images.append(dx_gt)
    dy_gt = image - image_shifted_down
    dy_images.append(dy_gt)

    del dx_gt, dy_gt

img_height, img_width = images[0].shape
# check that all images have the same dimensions
for image in images:
    assert image.shape == images[0].shape

dx_images = torch.from_numpy(np.array(dx_images, dtype=np.float32))
dy_images = torch.from_numpy(np.array(dy_images, dtype=np.float32))

image_filenames_hash = hashlib.md5(
    json.dumps(args.image_filenames).encode("utf-8")
).hexdigest()[:8]
num_images = len(images)
one_hot_vector_size = num_images if num_images > 1 else 0

# Prepare dataset
x = []
y = []
image_datasets = []
for k, image in enumerate(images):
    image_dataset_x = []
    image_dataset_y = []
    one_hot_vector = [0.0] * one_hot_vector_size
    if one_hot_vector_size >= 1:
        one_hot_vector[k] = 1.0
    for i in range(img_height):
        for j in range(img_width):
            vector = [i, j] + one_hot_vector
            image_dataset_x.append(vector)
            image_dataset_y.append([image[i][j]])

    x += image_dataset_x
    y += image_dataset_y
    image_dataset_x = np.array(image_dataset_x)
    image_dataset_y = np.array(image_dataset_y)
    image_datasets.append((image_dataset_x, image_dataset_y))

tensor_x = torch.from_numpy(np.array(x, dtype=np.float32))
tensor_y = torch.from_numpy(np.array(y, dtype=np.float32))


os.makedirs("output", exist_ok=True)


class SimpleNeuralNetwork(pl.LightningModule):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.half_height = self.height / 2
        self.width = width
        self.half_width = self.width / 2
        num_hidden_nodes = 128
        self.net = nn.Sequential(
            Siren(dim_in=2 + one_hot_vector_size, dim_out=num_hidden_nodes),
            Siren(dim_in=num_hidden_nodes, dim_out=num_hidden_nodes),
            Siren(dim_in=num_hidden_nodes, dim_out=num_hidden_nodes),
            Siren(dim_in=num_hidden_nodes, dim_out=num_hidden_nodes),
            Siren(dim_in=num_hidden_nodes, dim_out=num_hidden_nodes),
            nn.Linear(num_hidden_nodes, 1),
            nn.LeakyReLU(),
        )

    def forward(self, inputs):
        inputs = torch.clone(inputs)
        inputs[:, 0] = inputs[:, 0] / self.half_height
        inputs[:, 1] = inputs[:, 1] / self.half_width
        inputs[:, 0:2] = 1 - inputs[:, 0:2]
        return self.net(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_coordinates = x[:, 0].long()
        x_coordinates = x[:, 1].long()
        x_coords_shifted_left = torch.clamp(x_coordinates - 1, min=0)
        y_coords_shifter_up = torch.clamp(y_coordinates - 1, min=0)

        image_indexes = torch.argmax(x[:, 2:], dim=1)

        x_left_coords = torch.column_stack(
            (y_coordinates, x_coords_shifted_left)
        ).float()
        x_left = torch.clone(x)
        x_left[:, 0:2] = x_left_coords
        x_up_coords = torch.column_stack((y_coords_shifter_up, x_coordinates)).float()
        x_up = torch.clone(x)
        x_up[:, 0:2] = x_up_coords

        y_pred = self(x)
        y_pred_left = self(x_left)
        y_pred_up = self(x_up)
        dx_pred = y_pred - y_pred_left
        dy_pred = y_pred - y_pred_up

        dx_gt = dx_images[image_indexes, y_coordinates, x_coordinates]
        dy_gt = dy_images[image_indexes, y_coordinates, x_coordinates]

        pixelwise_loss = mse_loss(y_pred, y)
        dx_loss = mse_loss(dx_pred.flatten(), dx_gt)
        dy_loss = mse_loss(dy_pred.flatten(), dy_gt)
        return 0.5 * pixelwise_loss + 0.25 * dx_loss + 0.25 * dy_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
        return [optimizer], [scheduler]

    def on_train_epoch_end(self, outputs: Any) -> None:
        # We must implement this, or else SaveCheckpointImages.on_train_epoch_end
        # won't get any outputs
        pass


class SaveCheckpointImages(pl.Callback):
    def __init__(self):
        self.last_loss_checkpoint = float("inf")
        self.loss_change_threshold = 0.05

    def on_train_epoch_end(
        self, trainer, pl_module: pl.LightningModule, outputs: Any
    ) -> None:
        step_losses = [step[0]["minimize"].item() for step in outputs[0]]
        avg_loss = np.mean(step_losses)
        print("Loss: {:.6f}".format(avg_loss))
        loss_change = 1 - avg_loss / self.last_loss_checkpoint

        if loss_change > self.loss_change_threshold:
            self.last_loss_checkpoint = avg_loss

            with torch.no_grad():
                predicted_images = pl_module.forward(tensor_x).numpy()

            predicted_images = predicted_images.reshape(
                (num_images, image.shape[0], image.shape[1])
            )
            for img_idx in range(num_images):
                output_file_path = os.path.join(
                    "output",
                    "{0}_predicted_{1:04d}.png".format(
                        args.image_filenames[img_idx], trainer.current_epoch
                    ),
                )
                Image.fromarray(
                    np.clip(predicted_images[img_idx] * 256, 0, 255).astype(np.uint8)
                ).save(output_file_path)


nn = SimpleNeuralNetwork(img_height, img_width)
trainer = pl.Trainer(
    max_epochs=args.num_epochs,
    checkpoint_callback=False,
    logger=False,
    callbacks=[SaveCheckpointImages()],
)

tensor_dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(tensor_dataset, batch_size=128, shuffle=True)

# Train
trainer.fit(nn, train_loader)

# Predict
"""
with torch.no_grad():
    predicted_image = nn(tensor_x).numpy().reshape(image.shape)
    print("\nPredicted image:")
    print(predicted_image)
"""
