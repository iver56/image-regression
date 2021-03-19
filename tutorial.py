import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, TensorDataset

image = [[0, 130, 255], [40, 170, 255], [80, 210, 255]]
image = np.array(image)
image = np.divide(image, 255.0)
image_width, image_height = image.shape
print("Image with shape {0}:".format(image.shape))
print(image)

x = []
y = []
for i in range(image_height):
    for j in range(image_width):
        x.append([i / image_height, j / image_width])
        y.append([image[i][j]])
x = np.array(x)
y = np.array(y)

print("\nScaled coordinates (input):")
print(x)

print("\nScaled pixel brightness values (output):")
print(y)


class SimpleNeuralNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        num_hidden_nodes = 10
        self.net = nn.Sequential(
            nn.Linear(2, num_hidden_nodes),
            nn.SELU(),
            nn.Linear(num_hidden_nodes, 1),
            nn.SELU(),
        )

    def forward(self, inputs):
        inputs = self.net(inputs)
        return inputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = mse_loss(y_pred, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=1e-3, momentum=0.9, nesterov=True
        )
        return optimizer


nn = SimpleNeuralNetwork()
trainer = pl.Trainer(max_epochs=300, checkpoint_callback=False, logger=False)

tensor_x = torch.from_numpy(x.astype(np.float32))
tensor_y = torch.from_numpy(y.astype(np.float32))
tensor_dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(tensor_dataset)

# Train
trainer.fit(nn, train_loader)

# Predict
with torch.no_grad():
    predicted_image = nn(tensor_x).numpy().reshape(image.shape)
    print("\nPredicted image:")
    print(predicted_image)
