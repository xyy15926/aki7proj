#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_vae.py
#   Author: xyy15926
#   Created: 2025-09-02 09:16:19
#   Updated: 2025-09-03 23:36:47
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import vae
    from ubears.nutsbear import trainer
    reload(vae)
    reload(trainer)

from ubears.nutsbear.vae import VAEBase
from ubears.nutsbear.trainer import Trainer
from ubears.flagbear.slp.finer import get_tmp_path


# %%
class MNISTEnc(nn.Module):
    def __init__(
        self,
        inp_sz: int = (1, 28, 28),
        oup_sz: int = 128,
    ):
        super().__init__()
        self.fc1 = nn.Linear(np.multiply.reduce(inp_sz), oup_sz)

    def forward(self, inp):
        inp = inp.flatten(1, -1)
        oup = F.relu(self.fc1(inp))
        return oup


class MNISTDec(nn.Module):
    def __init__(
        self,
        inp_sz: int = 128,
        oup_sz: int = (1, 28, 28),
    ):
        super().__init__()
        fsz = np.multiply.reduce(oup_sz)
        self.fc1 = nn.Linear(inp_sz, fsz)
        self.fc2 = nn.Linear(fsz, fsz)
        self.oup_sz = oup_sz

    def forward(self, inp):
        oup = F.relu(self.fc1(inp))
        oup = F.sigmoid(self.fc2(oup))
        return oup.view(-1, *self.oup_sz)


# %%
@pytest.mark.skip(reason="Time comsuming.")
def test_VAEBase():
    bsz = 100
    mnist = datasets.MNIST(
        get_tmp_path() / "torch",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    train_loader = DataLoader(
        mnist,
        batch_size=bsz,
        shuffle=True,
    )

    lat_sz = 10
    enc_out_sz = 256
    mnist_sz = 1, 28, 28
    enc = MNISTEnc(mnist_sz, enc_out_sz)
    dec = MNISTDec(lat_sz, mnist_sz)
    vaeb = VAEBase(lat_sz, enc_out_sz, enc, dec)

    # Specify latent mu for each digit.
    mu_emb = nn.Embedding(10, lat_sz)
    opt = optim.Adam(vaeb.parameters())
    # opt.add_param_group({"params": mu_emb.parameters()})

    # x, y = next(iter(train_loader))
    # recon, mu, logsig = vaeb(x)
    # mu_ = mu_emb[y]
    # loss = VAEBase.vae_loss(recon, x, mu, logsig, mu_)

    def pred_fn(mod, x, y):
        return mod(x)

    def loss_fn(ret, x, y):
        recon, mu, logsig = ret
        mu_ = mu_emb(y)
        loss = VAEBase.vae_loss(recon, x, mu, logsig, mu_)
        return loss

    trainer = Trainer(
        vaeb,
        loss_fn,
        opt,
        "vaetest",
        pred_fn,
    )
    trainer.fit(train_loader, 1, 10)

    with torch.no_grad():
        # sample = torch.randn(64, lat_sz)
        sample = mu_emb.weight
        sample = vaeb.decode(sample)
        save_image(
            sample,
            get_tmp_path() / "torch/vaeimg.png"
        )
