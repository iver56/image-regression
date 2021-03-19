from __future__ import division

import argparse
import os
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from skimage.io import imread
from torch import nn
from torch.nn.functional import mse_loss, smooth_l1_loss
from torch.utils.data import TensorDataset, DataLoader

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-i",
    help="File name of input image",
    dest="input_filename",
    type=str,
    default="keyboard.png",
)
arg_parser.add_argument(
    "--num-epochs", help="Number of epochs", dest="num_epochs", type=int, default=1000
)
args = arg_parser.parse_args()

image = imread(args.input_filename, as_grey=True, plugin="pil")
if str(image.dtype) == "uint8":
    image = np.divide(image, 255.0, dtype=np.float32)
img_height, img_width = image.shape

image_tensor = torch.from_numpy(image)
image_shifted_right = torch.clone(image_tensor)
image_shifted_right[:, 1:] = image_tensor[:, :-1]
image_shifted_down = torch.clone(image_tensor)
image_shifted_down[1:, :] = image_tensor[:-1, :]

x = []
y = []
for i in range(img_height):
    for j in range(img_width):
        x.append([i, j])
        y.append([image[i][j]])
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)
tensor_x = torch.from_numpy(x)
tensor_y = torch.from_numpy(y)


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
            nn.Linear(2, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(num_hidden_nodes, 1),
            nn.LeakyReLU(),
        )

    def forward(self, inputs):
        inputs = torch.clone(inputs)
        inputs[:, 0] = inputs[:, 0] / self.half_height
        inputs[:, 1] = inputs[:, 1] / self.half_width
        inputs = 1 - inputs
        return self.net(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_coordinates = x[:, 0].long()
        x_coordinates = x[:, 1].long()
        x_coords_shifted_left = torch.clamp(x_coordinates - 1, min=0)
        y_coords_shifter_up = torch.clamp(y_coordinates - 1, min=0)

        x_left = torch.column_stack((y_coordinates, x_coords_shifted_left)).float()
        x_up = torch.column_stack((y_coords_shifter_up, x_coordinates)).float()
        y_pred = self(x)
        y_pred_left = self(x_left)
        y_pred_up = self(x_up)
        dx_pred = y_pred - y_pred_left
        dy_pred = y_pred - y_pred_up

        pixels = image_tensor[y_coordinates, x_coordinates]
        pixels_left_gt = image_shifted_right[y_coordinates, x_coordinates]
        pixels_top_gt = image_shifted_down[y_coordinates, x_coordinates]
        # TODO: gt_dx and gt_dy can be cached
        dx_gt = pixels - pixels_left_gt
        dy_gt = pixels - pixels_top_gt

        pixelwise_loss = mse_loss(y_pred, y)
        dx_loss = mse_loss(dx_pred.flatten(), dx_gt)
        dy_loss = mse_loss(dy_pred.flatten(), dy_gt)
        return 0.5 * pixelwise_loss + 0.25 * dx_loss + 0.25 * dy_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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
        loss_change = 1 - avg_loss / self.last_loss_checkpoint

        if loss_change > self.loss_change_threshold:
            self.last_loss_checkpoint = avg_loss

            with torch.no_grad():
                predicted_image = pl_module.forward(tensor_x).numpy()

            predicted_image = predicted_image.reshape(image.shape)

            output_file_path = os.path.join(
                "output",
                "{0}_predicted_{1:04d}.png".format(
                    args.input_filename, trainer.current_epoch
                ),
            )
            Image.fromarray(
                np.clip(predicted_image * 256, 0, 255).astype(np.uint8)
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
with torch.no_grad():
    predicted_image = nn(tensor_x).numpy().reshape(image.shape)
    print("\nPredicted image:")
    print(predicted_image)
