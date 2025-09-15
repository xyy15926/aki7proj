#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_diffusion.py
#   Author: xyy15926
#   Created: 2025-09-10 22:27:58
#   Updated: 2025-09-15 17:57:36
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
# https://pytorch.org/blog/extending-torchvisions-transforms-to-object-detection-segmentation-and-video-tasks/#the-new-transforms-api
import torchvision.transforms.v2 as transforms
from torchvision.utils import save_image
from PIL import Image

if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import trainer, unet, diffusion, posemb
    reload(trainer)
    reload(unet)
    reload(diffusion)
    reload(posemb)

from ubears.flagbear.slp.finer import get_tmp_path, get_assets_path, tmp_file
from ubears.nutsbear.trainer import Trainer
from ubears.nutsbear.unet import DoubleConv, UNetDown, UNetUp, UNet
from ubears.nutsbear.posemb import SinusoidalPECache
from ubears.nutsbear.diffusion import GaussianDiffusion


# %%
class UNetCond(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        bilinear: bool = False,
        class_n: int = None,
        embed_sz: int = 8,
    ):
        """UNet initialization.
        """
        super().__init__()
        self.cin = cin
        self.cout = cout

        # The in-channel and out-channel must be set carefully.
        self.in_conv = DoubleConv(cin, 4)
        self.dnc1 = UNetDown(4, 8)
        self.dnc2 = UNetDown(8, 16)
        self.upc2 = UNetUp(16, 8, bilinear)
        self.upc1 = UNetUp(8, 4, bilinear)
        self.out_conv = nn.Conv2d(4, cout, kernel_size=1)

        # Diffusion related attrs.
        self.class_n = class_n
        self.embed_sz = embed_sz
        self.tsinpe = SinusoidalPECache()
        if class_n is not None:
            self.class_emb = nn.Embedding(class_n, embed_sz)
        self.dnc1_pe = nn.Linear(embed_sz, 8)
        self.dnc2_pe = nn.Linear(embed_sz, 16)
        self.upc2_pe = nn.Linear(embed_sz, 8)
        self.upc1_pe = nn.Linear(embed_sz, 4)

    def forward(
        self,
        inp: torch.Tensor,
        tsteps: torch.Tensor,
        labels: torch.Tensor,
    ):
        """UNet forward.

        Shape:
        ---------------------------
        inp: [bsz, channel_n, height, width]
        tsteps: [bsz,]
        """
        bsz, esz, *____ = inp.size()
        # Apply (position) embedding for `tsteps` and labels.
        temb = self.tsinpe.get_pe(tsteps, self.embed_sz)
        if self.class_n is not None and labels is not None:
            lemb = self.class_emb(labels)
            temb = temb + lemb

        inp = self.in_conv(inp)
        dn1 = self.dnc1(inp)
        dn1 = dn1 + self.dnc1_pe(temb)[:, :, None, None]

        dn2 = self.dnc2(dn1)
        dn2 = dn2 + self.dnc2_pe(temb)[:, :, None, None]

        up2 = self.upc2(dn2, dn1)
        up2 = up2 + self.upc2_pe(temb)[:, :, None, None]

        up1 = self.upc1(up2, inp)
        up1 = up1 + self.upc1_pe(temb)[:, :, None, None]
        oup = self.out_conv(up1)

        return F.sigmoid(oup)


# %%
@pytest.mark.skip(reason="Time comsuming.")
def test_DDPM():
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5,], [0.5,]),
    ])
    mnist = datasets.MNIST(
        get_assets_path() / "images",
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        mnist,
        batch_size=100,
        shuffle=True,
    )
    unetm = UNetCond(1, 1, False, 10, 8)
    optm = optim.Adam(unetm.parameters())
    ddpm = GaussianDiffusion(100)

    # x, y = next(iter(train_loader))
    def pred_loss_fn(model, inp, labels):
        bsz = inp.size(0)
        steps_n = ddpm.steps_n
        t = torch.randint(1, steps_n, size=(bsz,))
        noised, zt = ddpm.addnoise(inp, t)
        pred_zt = model(noised, t, labels)
        loss = F.mse_loss(zt, pred_zt)
        return loss

    trn = Trainer(
        unetm,
        pred_loss_fn,
        optm,
        "tboard/DDPM_UNet",
    )
    trn.fit(train_loader, 5)

    with torch.no_grad():
        x, y = next(iter(train_loader))
        xs = [x,]
        # Add noise and denoise.
        for n in range(10, 101, 10):
            xn, zt = ddpm.addnoise(x, n)
            xs.append(xn)
            xd = ddpm.denoise(unetm, xn, n, y)
            xs.append(xd)
        # Denoise from random.
        rnd = torch.randn_like(x)
        xs.append(rnd)
        rnd_ded = ddpm.denoise(unetm, rnd, 100, y)
        xs.append(rnd_ded)

        save_image(
            torch.concat(xs, dim=0),
            tmp_file("torch/ddpm_unet.png"),
            nrow=100,
        )

    trn.save()
