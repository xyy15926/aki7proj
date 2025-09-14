#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_unet.py
#   Author: xyy15926
#   Created: 2025-09-10 19:20:48
#   Updated: 2025-09-12 11:17:28
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
# https://pytorch.org/blog/extending-torchvisions-transforms-to-object-detection-segmentation-and-video-tasks/#the-new-transforms-api
import torchvision.transforms.v2 as transforms
from torchvision.utils import save_image
from PIL import Image

if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import unet
    from ubears.nutsbear import trainer
    reload(unet)
    reload(trainer)

from ubears.nutsbear.unet import UNet
from ubears.nutsbear.trainer import Trainer
from ubears.flagbear.slp.finer import get_tmp_path, get_assets_path


# %%
class ISBIImageDataset(Dataset):
    def __init__(self, path:str, transform=None):
        images = get_assets_path() / path / "train/image"
        labels = get_assets_path() / path / "train/label"
        self.images = list(images.iterdir())
        self.labels = list(labels.iterdir())
        self.images.sort()
        self.labels.sort()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        lbl = Image.open(self.labels[idx])
        if self.transform:
            return self.transform(img, lbl)
        else:
            return img, lbl


# %%
@pytest.mark.skip(reason="Time comsuming.")
def test_unet():
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    train_dataset = ISBIImageDataset(
        path="images/ISBI",
        transform=transform,
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=2)

    def pred_loss_fn(mod, img, label):
        pred = mod(img)
        return F.binary_cross_entropy(pred, label)

    unet = UNet(1, 1)
    optm = optim.Adam(unet.parameters())
    trainer = Trainer(
        unet,
        pred_loss_fn,
        optm,
        "ISBI_seg",
    )
    trainer.fit(train_loader, 40)

    with torch.no_grad():
        train_dataset = ISBIImageDataset(
            path="images/ISBI",
            transform=transform,
        )
        train_loader = DataLoader(dataset=train_dataset, batch_size=4)
        sample = next(iter(train_loader))
        pred = unet(sample[0])
        pred_01 = torch.zeros_like(pred)
        pred_01[pred > 0.5] = 1
        ret = torch.concat([sample[0], sample[1], pred, pred_01], dim=0)
        save_image(
            ret,
            get_tmp_path() / "torch/unetimg.png",
            nrow=4,
        )
