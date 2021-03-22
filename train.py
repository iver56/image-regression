from __future__ import division

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from siren_pytorch import Siren
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset, DataLoader

from utils.differentiable_clamp import differential_clamp

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
    "--edge-loss-factor",
    dest="edge_loss_factor",
    type=float,
    help="How much should edges (differences between 4-connected neighbour pixels) be weighted",
    default=0.0,
)
arg_parser.add_argument(
    "--num-epochs", help="Number of epochs", dest="num_epochs", type=int, default=750
)
arg_parser.add_argument(
    "--batch-size",
    dest="batch_size",
    help="How many samples should be in each training batch?",
    type=int,
    default=256,
)
arg_parser.add_argument(
    "--hidden-nodes",
    dest="hidden_nodes",
    help="How many nodes should be in each hidden layer?",
    type=int,
    default=150,
)
arg_parser.add_argument(
    "--use-cuda", dest="use_cuda", default=1, type=int, help="Use CUDA (GPU) or not?"
)
args = arg_parser.parse_args()

use_cuda = args.use_cuda
if use_cuda and not torch.cuda.is_available():
    print("Warning: Trying to use CUDA, but it is not available")
    use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

half_edge_loss_factor = args.edge_loss_factor / 2

images = []
dx_images = []
dy_images = []
for image_filename in args.image_filenames:
    image = np.array(Image.open(image_filename).convert("L"))
    image = np.divide(image, 255.0, dtype=np.float32)
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

dx_images = torch.from_numpy(np.array(dx_images, dtype=np.float32)).to(device)
dy_images = torch.from_numpy(np.array(dy_images, dtype=np.float32)).to(device)

image_filenames_hash = (
    "_".join(Path(filename).stem for filename in args.image_filenames)[0:200]
    + "_"
    + hashlib.md5(json.dumps(args.image_filenames).encode("utf-8")).hexdigest()[:8]
)
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

tensor_x = torch.from_numpy(np.array(x, dtype=np.float32)).to(device)
tensor_y = torch.from_numpy(np.array(y, dtype=np.float32)).to(device)

os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)


def save_model(model, model_name, input_example, device):
    print("Saving model...")
    model_training = model.training
    model_device = model.device

    model_on_device = model.to(device)
    if model_training:
        model_on_device.eval()  # Set eval mode
    with torch.no_grad():
        traced_model = torch.jit.trace(model_on_device, input_example.to(device))
    torch.jit.save(
        traced_model,
        os.path.join("models", "{}_{}.torchscript".format(model_name, device)),
    )

    model.to(model_device)
    if model_training:
        model.train()  # Set back to train mode


class SimpleNeuralNetwork(pl.LightningModule):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.half_height = self.height / 2
        self.width = width
        self.half_width = self.width / 2
        self.net = nn.Sequential(
            Siren(dim_in=2 + one_hot_vector_size, dim_out=args.hidden_nodes),
            Siren(dim_in=args.hidden_nodes, dim_out=args.hidden_nodes),
            Siren(dim_in=args.hidden_nodes, dim_out=args.hidden_nodes),
            Siren(dim_in=args.hidden_nodes, dim_out=args.hidden_nodes),
            Siren(dim_in=args.hidden_nodes, dim_out=args.hidden_nodes),
            nn.Linear(args.hidden_nodes, 1),
        )

    def forward(self, inputs):
        inputs = torch.clone(inputs)
        inputs[:, 0] = inputs[:, 0] / self.half_height
        inputs[:, 1] = inputs[:, 1] / self.half_width
        inputs[:, 0:2] = 3.14 * (1 - inputs[:, 0:2])
        net_output = self.net(inputs)
        if self.training:
            return differential_clamp(net_output, min_val=0.0, max_val=1.0)
        else:
            return torch.clamp(net_output, min=0.0, max=1.0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        pixelwise_loss = mse_loss(y_pred, y)

        if half_edge_loss_factor > 0.0:
            y_coordinates = x[:, 0].long()
            x_coordinates = x[:, 1].long()
            x_coords_shifted_left = torch.clamp(x_coordinates - 1, min=0)
            y_coords_shifted_up = torch.clamp(y_coordinates - 1, min=0)

            image_indexes = torch.argmax(x[:, 2:], dim=1)

            x_left_coords = torch.column_stack(
                (y_coordinates, x_coords_shifted_left)
            ).float()
            x_left = torch.clone(x)
            x_left[:, 0:2] = x_left_coords
            x_up_coords = torch.column_stack(
                (y_coords_shifted_up, x_coordinates)
            ).float()
            x_up = torch.clone(x)
            x_up[:, 0:2] = x_up_coords

            y_pred_left = self(x_left)
            y_pred_up = self(x_up)
            dx_pred = y_pred - y_pred_left
            dy_pred = y_pred - y_pred_up

            dx_gt = dx_images[image_indexes, y_coordinates, x_coordinates]
            dy_gt = dy_images[image_indexes, y_coordinates, x_coordinates]
            dx_loss = mse_loss(dx_pred.flatten(), dx_gt)
            dy_loss = mse_loss(dy_pred.flatten(), dy_gt)

            return (
                0.5 * pixelwise_loss
                + half_edge_loss_factor * dx_loss
                + half_edge_loss_factor * dy_loss
            )
        else:
            return pixelwise_loss

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

            predicted_pixels = np.zeros(
                shape=tensor_y.shape,
                dtype=np.float32,
            )
            with torch.no_grad():
                for offset in range(0, tensor_x.shape[0], args.batch_size):
                    pred = (
                        pl_module.forward(tensor_x[offset : offset + args.batch_size])
                        .cpu()
                        .numpy()
                    )
                    predicted_pixels[offset : offset + pred.shape[0]] = pred

            predicted_images = predicted_pixels.reshape(
                (num_images, img_height, img_width)
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
    gpus=1 if use_cuda else 0,
)

tensor_dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=True)

# Train
trainer.fit(nn, train_loader)

save_model(
    model=nn,
    model_name=image_filenames_hash,
    input_example=tensor_x[0:128],
    device=device,
)

# Predict
"""
with torch.no_grad():
    predicted_image = nn(tensor_x).numpy().reshape(image.shape)
    print("\nPredicted image:")
    print(predicted_image)
"""
