#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_fixture.py
#   Author: xyy15926
#   Created: 2025-11-22 20:25:52
#   Updated: 2025-11-22 20:47:08
#   Description:
# ---------------------------------------------------------


# %%
import pytest
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import fixture
    reload(fixture)

from ubears.nutsbear.fixture import (
    fkwargs_32_cpu,
    fkwargs_64_cpu,
    fkwargs_32_dml,
    fkwargs_64_dml,
    all_close,
    ssoftmax,
)

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
def test_all_close():
    tt_32_cpu = torch.ones((4, 5), **fkwargs_32_cpu)
    assert all_close(tt_32_cpu, 1.0)

    tt_32_cpu[0, 0] = torch.nan
    tt_32_cpu[0, 1] = torch.inf
    tt_64_cpu = tt_32_cpu.to(**fkwargs_64_cpu)
    tt_32_dml = tt_32_cpu.to(**fkwargs_32_dml)
    tt_64_dml = tt_32_dml.to(**fkwargs_64_dml)

    assert all_close(tt_32_cpu, tt_32_cpu)
    assert all_close(tt_32_cpu, tt_64_cpu)
    assert all_close(tt_32_cpu, tt_32_dml)
    assert all_close(tt_32_cpu, tt_64_dml)
    assert all_close(tt_64_dml, tt_32_cpu)

    assert not all_close(tt_32_cpu, tt_32_cpu, equal_nan=False)
    assert not all_close(tt_32_cpu, tt_64_cpu, equal_nan=False)
    assert not all_close(tt_32_cpu, tt_32_dml, equal_nan=False)
    assert not all_close(tt_32_cpu, tt_64_dml, equal_nan=False)
    assert not all_close(tt_64_dml, tt_32_cpu, equal_nan=False)


# %%
def test_ssoftmax(torch_fkwargs):
    # Check common result.
    ori = torch.rand(5, 3, 4, **torch_fkwargs)
    for dim in (0, 1, 2, -1):
        ret1 = F.softmax(ori, dim)
        ret2 = ssoftmax(ori, dim)
        assert torch.all(torch.isclose(ret1, ret2))

    # Check all-NInf input.
    addup = torch.tensor([[float("-inf"), 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]], **torch_fkwargs)
    inp = ori + addup
    for dim in (1, 2):
        ret1 = F.softmax(inp, dim)
        ret2 = ssoftmax(inp, dim)
        assert torch.all(torch.isclose(ret1, ret2))
    ret1 = F.softmax(inp, 0)
    ret2 = ssoftmax(inp, 0)
    assert all_close(ret1[:, :, 1:], ret2[:, :, 1:])
    assert torch.all(torch.isnan(ret1[:, 0, 0]))
    assert all_close(ret2[:, 0, 0], 0.0)

    # Check the autograd.
    ori = torch.rand(5, 3, 4, requires_grad=True, **torch_fkwargs)
    inp = ori + addup
    loss = F.softmax(inp, 0).sum()
    loss.backward()
    assert torch.all(torch.isnan(ori.grad[:, 0, 0]))

    ori = torch.rand(5, 3, 4, requires_grad=True, **torch_fkwargs)
    inp = ori + addup
    loss = ssoftmax(inp, 0).sum()
    loss.backward()
    assert all_close(ret2[:, 0, 0], 0.0)


