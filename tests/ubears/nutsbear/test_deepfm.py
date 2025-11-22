#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_deepfm.py
#   Author: xyy15926
#   Created: 2025-06-19 09:27:32
#   Updated: 2025-11-22 21:02:34
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import torch
if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import fixture
    from ubears.nutsbear import deepfm
    reload(fixture)
    reload(deepfm)

from ubears.nutsbear.fixture import (
    fkwargs_32_cpu,
    fkwargs_64_cpu,
    fkwargs_32_dml,
    fkwargs_64_dml,
    all_close,
)
from ubears.nutsbear.deepfm import DeepFM
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
def mock_ctr(
    fea_catn: int = 5,
    sample_n: int = 100,
    sparse_n: int = 4,
    dense_n: int = 5,
    device: str = None,
    dtype: str = None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    inp_idx = torch.randint(
        0, fea_catn, (sample_n, sparse_n + dense_n),
        device=device,
    )
    inp_idx[:, sparse_n:] = 0
    inp_val = torch.ones(sample_n, sparse_n + dense_n, **factory_kwargs)
    torch.rand((sample_n, dense_n), out=inp_val[:, sparse_n:])
    label = torch.randint(0, 2, (sample_n,), device=device)

    return inp_idx, inp_val, label


# %%
def test_DeepFM(torch_fkwargs):
    sparse_n, dense_n = 4, 5
    sample_n = 100
    fea_catn = 5
    inp_idx, inp_val, label = mock_ctr(
        fea_catn,
        sample_n,
        sparse_n,
        dense_n,
        **torch_fkwargs,
    )

    mod = DeepFM([fea_catn] * sparse_n + [1] * dense_n, **torch_fkwargs)
    ret = mod(inp_idx, inp_val)
    assert ret.size(0) == sample_n
