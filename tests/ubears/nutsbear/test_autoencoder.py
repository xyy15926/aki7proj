#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_autoencoder.py
#   Author: xyy15926
#   Created: 2025-09-02 09:16:19
#   Updated: 2025-11-21 22:42:32
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import fixture
    from ubears.nutsbear import autoencoder
    from ubears.nutsbear import trainer
    reload(fixture)
    reload(autoencoder)
    reload(trainer)

from ubears.nutsbear.fixture import (
    fkwargs_32_cpu,
    fkwargs_64_cpu,
    fkwargs_32_dml,
    fkwargs_64_dml,
    all_close,
)
from ubears.nutsbear.autoencoder import VAEBase
from ubears.nutsbear.trainer import Trainer
from ubears.flagbear.slp.finer import get_tmp_path, get_assets_path, tmp_file

torch.autograd.set_detect_anomaly(False)


# %%
if fkwargs_32_dml:
    torch_fkwargs_params = [fkwargs_64_cpu, fkwargs_32_dml]
else:
    torch_fkwargs_params = [fkwargs_64_cpu, ]
@pytest.fixture(params=torch_fkwargs_params)
def torch_fkwargs(request):
    return request.param
# torch_fkwargs = fkwargs_32_dml
# torch_fkwargs = fkwargs_64_cpu


# %%
class MNISTEnc(nn.Module):
    def __init__(
        self,
        inp_sz: int = (1, 28, 28),
        oup_sz: int = 128,
        device: str = None,
        dtype: str = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fc1 = nn.Linear(np.multiply.reduce(inp_sz), oup_sz, **factory_kwargs)

    def forward(self, inp):
        inp = inp.flatten(1, -1)
        oup = F.relu(self.fc1(inp))
        return oup


class MNISTDec(nn.Module):
    def __init__(
        self,
        inp_sz: int = 128,
        oup_sz: int = (1, 28, 28),
        device: str = None,
        dtype: str = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        fsz = np.multiply.reduce(oup_sz)
        self.fc1 = nn.Linear(inp_sz, fsz, **factory_kwargs)
        self.fc2 = nn.Linear(fsz, fsz, **factory_kwargs)
        self.oup_sz = oup_sz

    def forward(self, inp):
        oup = F.relu(self.fc1(inp))
        oup = F.sigmoid(self.fc2(oup))
        return oup.view(-1, *self.oup_sz)


# %%
@pytest.mark.skip(reason="Time comsuming.")
def test_VAEBase(torch_fkwargs):
    device, dtype = torch_fkwargs["device"], torch_fkwargs["dtype"]
    bsz = 100
    mnist = datasets.MNIST(
        get_assets_path() / "images",
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
    enc = MNISTEnc(mnist_sz, enc_out_sz, **torch_fkwargs)
    dec = MNISTDec(lat_sz, mnist_sz, **torch_fkwargs)
    vaeb = VAEBase(lat_sz, enc_out_sz, enc, dec, **torch_fkwargs)

    # Specify latent mu for each digit.
    mu_emb = nn.Embedding(10, lat_sz, **torch_fkwargs)
    opt = optim.Adam(vaeb.parameters())
    opt.add_param_group({"params": mu_emb.parameters()})

    # x, y = next(iter(train_loader))
    # recon, mu, logsig = vaeb(x)
    # mu_ = mu_emb[y]
    # loss = VAEBase.vae_loss(recon, x, mu, logsig, mu_)

    def pred_loss_fn(mod, x, y):
        x = x.to(**torch_fkwargs)
        y = y.to(**torch_fkwargs)
        recon, mu, logsig = mod(x)
        mu_ = mu_emb(y)
        loss, ce, kl = VAEBase.vae_loss(recon, x, mu, logsig, mu_)
        return loss, {"CE": ce, "KLD": kl}

    trainer = Trainer(
        vaeb,
        pred_loss_fn,
        opt,
        "vaetest",
    )
    trainer.fit(train_loader, 1, 10)

    with torch.no_grad():
        sample = torch.randn(10, 10, lat_sz, **torch_fkwargs)
        sample = sample + mu_emb.weight.unsqueeze(1)
        sample = vaeb.decode(sample).to("cpu")
        save_image(
            sample,
            tmp_file("torch/vaeimg.png"),
            nrow=10,
        )
