#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_posemb.py
#   Author: xyy15926
#   Created: 2025-09-12 20:07:17
#   Updated: 2025-11-17 22:45:52
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark
import numpy as np
import torch
from torch import nn, optim
from torch import nested
from torch.nn import functional as F

if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import posemb
    reload(posemb)

from ubears.nutsbear.posemb import (
    SinPE,
    RotaryPE,
)
torch.autograd.set_detect_anomaly(False)


# %%
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py#L117
class TimestepSinPE(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # Why????
        # embeddings = np.log(10000) / (half_dim - 1)
        embeddings = np.log(10000) / half_dim
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def test_SinPE():
    esz = 32
    tstep = torch.randint(0, 100, (100,))

    sinpe = SinPE()
    x = torch.zeros(100, esz, dtype=torch.int)
    xped = sinpe(x, tstep)

    # Test the timestep embedding.
    tespe = TimestepSinPE(esz)
    xtped = tespe(tstep)
    assert torch.allclose(xped[:, ::2], xtped[:, :16], rtol=1e-4)
    assert torch.allclose(xped[:, 1::2], xtped[:, -16:], rtol=1e-4)

    # The cache won't be updated(regenerated).
    cache = sinpe.pe_cache
    tstep = torch.randint(0, 50, (100,))
    xped2 = sinpe(x, tstep)
    assert cache is sinpe.pe_cache

    # The cache will be updated(regenerated).
    cache = sinpe.pe_cache
    tstep = torch.randint(100, 200, (100,))
    xped3 = sinpe(x, tstep)
    assert cache is not sinpe.pe_cache
    assert sinpe.pe_cache.size(0) > 100
    assert sinpe.pe_cache.size(1) == esz

    # The cache will be updated(regenerated).
    cache = sinpe.pe_cache
    x = torch.zeros(100, esz + 2, dtype=torch.int)
    tstep = torch.randint(0, 50, (100,))
    xped4 = sinpe(x, tstep)
    assert cache is not sinpe.pe_cache
    assert sinpe.pe_cache.size(0) > 100
    assert sinpe.pe_cache.size(1) == esz + 2


# %%
def test_RotaryPE():
    esz = 32
    rpe = RotaryPE(esz)

    # The absolute position N1, N2 doesn't matter.
    x = torch.ones(1000, esz)
    x_rped = rpe(x)
    N1, N2 = torch.randint(40, (2,)).tolist()
    ret = x_rped[N1:] @ x_rped[N2:].transpose(0, 1)
    # The diagonal elements's relative position gaps are all `N2 - N1`.
    assert torch.all(torch.isclose(ret[0, 0], torch.diag(ret)))
    assert torch.all(torch.isclose(ret[0, 1], torch.diag(ret, 1)))

    x = torch.randn((100, 32))
    N1, N2 = torch.randint(40, (2,)).tolist()
    x1_rped = rpe(x, torch.arange(N1, N1 + 100))
    ret1 = x1_rped[:50] @ x1_rped[50:].transpose(0, 1)
    x2_rped = rpe(x, torch.arange(N2, N2 + 100))
    ret2 = x2_rped[:50] @ x2_rped[50:].transpose(0, 1)
    assert torch.all(torch.isclose(ret1, ret2, rtol=1e-2))

    # Multi dimensions.
    x = torch.ones(100, 100, esz)
    x_rped = rpe(x)
    N1, N2 = torch.randint(40, (2,)).tolist()
    ret = x_rped[:, N1:] @ x_rped[:, N2:].transpose(-1, -2)
    assert torch.all(torch.isclose(ret[0].expand(100, -1, -1), ret))
    assert torch.all(torch.isclose(ret[0, 0, 0], torch.diag(ret[0])))
    assert torch.all(torch.isclose(ret[0, 0, 1], torch.diag(ret[0], 1)))
