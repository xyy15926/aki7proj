#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_deepfm.py
#   Author: xyy15926
#   Created: 2025-06-19 09:27:32
#   Updated: 2025-07-08 16:11:35
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import torch
if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import deepfm
    reload(deepfm)

from ubears.nutsbear.deepfm import DeepFM


# %%
def mock_ctr(
    fea_catn: int = 5,
    sample_n: int = 100,
    sparse_n: int = 4,
    dense_n: int = 5,
):
    inp_idx = torch.randint(0, fea_catn, (sample_n, sparse_n + dense_n))
    inp_idx[:, sparse_n:] = 0
    inp_val = torch.ones(sample_n, sparse_n + dense_n, dtype=torch.float)
    torch.rand((sample_n, dense_n), out=inp_val[:, sparse_n:])
    label = torch.randint(0, 2, (sample_n,))

    return inp_idx, inp_val, label


# %%
def test_DeepFM():
    sparse_n, dense_n = 4, 5
    sample_n = 100
    fea_catn = 5
    inp_idx, inp_val, label = mock_ctr(fea_catn, sample_n, sparse_n, dense_n)

    mod = DeepFM([fea_catn] * sparse_n + [1] * dense_n)
    ret = mod(inp_idx, inp_val)
    assert ret.size(0) == sample_n
