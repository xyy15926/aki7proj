#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_trainer.py
#   Author: xyy15926
#   Created: 2025-07-08 16:09:07
#   Updated: 2025-07-21 21:41:41
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import deepfm
    from ubears.nutsbear import trainer
    reload(deepfm)
    reload(trainer)

from ubears.nutsbear.deepfm import DeepFM
from ubears.nutsbear.trainer import Trainer


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
    label = torch.randint(0, 2, (sample_n,), dtype=torch.float16)

    return inp_idx, inp_val, label


# %%
def test_Trainer():
    sparse_n, dense_n = 4, 5
    sample_n = 100
    fea_catn = 5
    inp_idx, inp_val, label = mock_ctr(fea_catn, sample_n, sparse_n, dense_n)
    dataset = TensorDataset(inp_idx, inp_val, label)
    dloader = DataLoader(dataset, 5)

    # Trainer with default pred_fn and loss_fn.
    mod_name = "test"
    mod = DeepFM([fea_catn] * sparse_n + [1] * dense_n)
    trn = Trainer(
        mod,
        F.binary_cross_entropy_with_logits,
        mod_name=mod_name
    )
    trn.fit(dloader, 2, 4)

    # Trainer with customed pred_fn and loss_fn.
    mod_name = "test"
    mod = DeepFM([fea_catn] * sparse_n + [1] * dense_n)
    trn = Trainer(
        mod,
        loss_fn=lambda x, *ele: F.binary_cross_entropy_with_logits(x, ele[-1]),
        mod_name=mod_name,
        pred_fn=lambda mod, *ele: mod(*ele[:-1]),
    )
    trn.fit(dloader, 2, 4)

    # Save and load test.
    trn.save()
    sdict = mod.state_dict()
    new_mod = DeepFM([fea_catn] * sparse_n + [1] * dense_n)
    Trainer.load(new_mod, mod_name)
    new_sdict = new_mod.state_dict()
    for key, val in sdict.items():
        assert torch.all(val == new_sdict[key])
