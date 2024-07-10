#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: torchflow.py
#   Author: xyy15926
#   Created: 2024-06-24 11:54:56
#   Updated: 2024-07-10 10:44:37
#   Description:
# ---------------------------------------------------------

# %%
import os
import logging
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

if __name__ == "__main__":
    from importlib import reload
    from suitbear import finer
    reload(finer)

from suitbear.finer import get_tmp_path

DEVICE = ("cuda"
          if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu")
CLASSES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

logging.basicConfig(
    # format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    format="%(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# Reference: <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>
#
# A custom Dataset must implements:
# - `__init__`: Initialize the Dataset object.
# - `__len__`: Return the number of samples in dataset.
# - `__getitem__`: Return the sample and its label.
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir,
                 transform=None,
                 target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def load_data():
    training_data = datasets.FashionMNIST(
        root=get_tmp_path() / "data",
        train=True,
        # ToTensor converts PIL image or np.ndarray into FloatTensor.
        transform=ToTensor(),
        download=True)
    test_data = datasets.FashionMNIST(
        root=get_tmp_path() / "data",
        train=False,
        transform=ToTensor(),
        download=True)

    # Wrap with DataLoader for automatic batching, sampling and multiprocessing
    # data loading.
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


# %%
# Reference: <https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html>
def build_nn():
    train_dataloader, test_dataloader = load_data()
    X_batch, y_batch = next(iter(test_dataloader))

    # nn.Flatten
    flatten = nn.Flatten()
    flat_image = flatten(X_batch)
    assert len(flat_image.shape) == 2

    # nn.Linear
    linear = nn.Linear(28 * 28, 20)
    hidden1 = linear(flat_image)
    assert hidden1.shape[1] == 20

    # nn.ReLU
    relu = nn.ReLU()
    hidden2 = relu(hidden1)
    assert torch.all(hidden2 >= 0)

    # nn.Sequential
    seq_modules = nn.Sequential(
        flatten,
        linear,
        relu,
        nn.Linear(20, 10))
    logits = seq_modules(X_batch)

    # nn.Softmax
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)
    assert logits.shape == pred_probab.shape


# %%
# Reference: <https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html>
#
# A custom nn.Module must implements:
# - `__init__`: Initialize the module.
# - `forward`: Apply forward pass procedure to compute the result.
#   `forward` will be executed automatically when pass data to the model
#   instance along with some background operations and should be called
#   direclty.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# %%
# Reference: <https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html>
def train_model(dataloader, model, loss_fn, optimizer,
                epoch=0, writer=None, device=DEVICE):
    # device = DEVICE
    # dataloader = train_dataloader
    size = len(dataloader.dataset)

    # Set the model to training model, which is important for batch-norm and
    # dropout.
    # Here, just for best practice.
    model.train()

    logger.info(f"Epoch {epoch+1}\n-----------------------------")
    # Inside train loop, optimization happens in three steps:
    # 1. Call `optimizer.zero_grad()` to reset gradient of model parameters to
    #   prevent double counting.
    # 2. Call `loss.backward()` to backpropagate the prediction loss.
    # 3. Call `optimizer.step()` to adjust parameters by the gradients
    #   collected in the backward pass.
    for batch, (X, y) in enumerate(dataloader):
        # X, y = next(iter(dataloader))
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f} [{current: >5d} / {size:>5d}]")

            if writer is not None:
                writer.add_scalar("train_loss", loss,
                                  epoch * len(dataloader) + batch)


def validate_model(dataloader, model, loss_fn,
                   epoch=0, writer=None, device=DEVICE):
    # device = DEVICE
    # dataloader = test_dataloader
    classes = CLASSES
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # Set the model to test model, which is important for batch-norm and
    # dropout.
    # Here, just for best practice.
    model.eval()
    test_loss, correct = 0, 0

    batch_probs = []
    batch_labels = []
    # Evaluating the model with `torch.no_grad()` to prevent gradient
    # computation to reduce overhead.
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = next(iter(dataloader))
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            batch_probs.append(pred)
            batch_labels.append(y)

    probs = torch.cat(batch_probs)
    labels = torch.cat(batch_labels)
    if writer is not None:
        for idx, class_ in enumerate(classes):
            writer.add_pr_curve(class_, labels == idx, probs[:, idx], global_step=epoch)

    correct += (probs.argmax(dim=1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(f"Test Error: \n"
                f"Accuracy: {(100 * correct):>0.1f}%, "
                f"Avg loss: {test_loss:>8f} \n")


# %%
# Reference: <https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html>
def fit_model():
    device = DEVICE
    classes = CLASSES
    train_dataloader, test_dataloader = load_data()

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Log with TensorBoard.
    images, labels = next(iter(train_dataloader))
    writer = SummaryWriter(get_tmp_path() / "runs/fashion_mnist_experiment_1")
    img_grid = make_grid(images)
    writer.add_image("fashion_mnist_images", img_grid)
    writer.add_graph(model, images)
    # TensorBoard may be need to restart to load projector.
    writer.add_embedding(images.view(-1, 28 * 28),
                         metadata=[classes[lab] for lab in labels],
                         label_img=images,
                         global_step=0)

    epochs = 5
    for t in range(epochs):
        train_model(train_dataloader, model, loss_fn, optimizer, t, writer)
        validate_model(test_dataloader, model, loss_fn, t, writer)

    torch.save(model.state_dict(), get_tmp_path() / "fashion_mnist_model.pth")


# %%
# Reference: <https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html>
def load_model():
    device = DEVICE
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(get_tmp_path() / "fashion_mnist_model.pth"))
    train_dataloader, test_dataloader = load_data()
    X_batch, y_batch = next(iter(test_dataloader))
    with torch.no_grad():
        x = X_batch[0].to(device)
        pred = model(x)
        predicted = pred[0].argmax(0)
        assert(y_batch[0] == predicted)
    # Model Parameters
    for name, param in model.named_parameters():
        logger.info(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
