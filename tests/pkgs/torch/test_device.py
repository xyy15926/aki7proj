#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_device.py
#   Author: xyy15926
#   Created: 2025-11-20 15:45:38
#   Updated: 2025-11-21 21:59:46
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark
from packaging.version import Version
import torch
from torch import nn, optim
from torch.nn import functional as F

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
)

# DML Attention:
# 1. `torch.flip` will lead panic crush in DML.
#
# 2. `torch.isnan` doesn't support `torch.float64` in DML, and so don't the
#   following:
# 2.1 `torch.allclose` with `equal_nan=True`,
# 2.2 `torch.arange`
# 2.3 `torch.isinf`
#
# 3. `.to()` can't cast dtype and move tensor from CPU to GPU simutaneously,
# 3.1 But `.to()` can cast dtype and move tensor from GPU to CPU simutaneously.
# 3.2 And this apply to `Module.to` !!!!!
#
# 4. Assgignment for slice in DML has been restricted that only slice on
#   the 1st demension could be assigned freely in DML.
#
# 5. `torch.eye` hasn't been implemented for DML.
# 5.1 But just like the assgignment, the `torch.eye(out=some_slice)` DML has
#   been restricted that only slice on the 1st demension could be assigned
#   freely.
#   And so do other inplace calculations such as `ones()`, `zeros()`.
#
# 6. `torch.full` doesn't support `torch.floating` in DML fully that
#   unexpected result will come up for tensor with `inf`, `nan`.
# 
# 7. `torch.triu` will set the lower-triangle with `nan` wrongly.
# 7.1 `nn.Transformer.generate_square_subsequent_mask` will return `nan` for
#   non-masked position in DML.
# 7.1.1 And `tgt_is_causal = None` instead of explicitly `True` or `False`
#    will call `_detect_is_causal_mask` and this sequncely with `attn_mask`
#    passed in TransformerDecoderLayer.


# %%
# 2. `torch.isnan` doesn't support `torch.float64` in DML.
def test_isnan():
    if not fkwargs_64_dml:
        return

    for tag, fk in [
        ("yes", fkwargs_32_cpu),
        ("yes", fkwargs_64_cpu),
        ("yes", fkwargs_32_dml),
        ("no", fkwargs_64_dml),
    ]:
        t = torch.ones(3, **fk)
        if tag == "yes":
            assert not torch.any(torch.isnan(t))
        else:
            with pytest.raises(RuntimeError):
                assert not torch.any(torch.isnan(t))


# %%
# 2.1 `torch.allclose` doesn't support `equal_nan=True` for `torch.float64`
#   in DML.
def test_allclose_equal_nan_float64_DML():
    if not fkwargs_64_dml:
        return

    t = torch.tensor([float("nan")], **fkwargs_64_dml)
    assert not torch.allclose(t, t, equal_nan=False)

    # 1. `torch.allclose` doesn't support `equal_nan=True` in for `torch.float64`
    #   DML.
    for tag, fk in [
        ("yes", fkwargs_32_cpu),
        ("yes", fkwargs_64_cpu),
        ("yes", fkwargs_32_dml),
        ("no", fkwargs_64_dml),
    ]:
        t = torch.ones(3, **fk)
        if tag == "yes":
            assert torch.allclose(t, t, equal_nan=True)
        else:
            with pytest.raises(RuntimeError):
                assert torch.allclose(t, t, equal_nan=True)


# %%
# 2.2 `torch.arange` doesn't support `torch.float64` in DML.
def test_arange_float64_DML():
    if not fkwargs_64_dml:
        return

    t = torch.arange(0, 100, **fkwargs_64_cpu)

    # 1. `torch.arange` doesn't support `torch.float64` in DML.
    for tag, fk in [
        ("yes", fkwargs_32_cpu),
        ("yes", fkwargs_64_cpu),
        ("yes", fkwargs_32_dml),
        ("no", fkwargs_64_dml),
    ]:
        if tag == "yes":
            tt = torch.arange(0, 100, **fk)
            assert all_close(t, tt)
        else:
            with pytest.raises(RuntimeError):
                tt = torch.arange(0, 100, **fk)
                assert all_close(t, tt)


# %%
# 3. `.to()` can't cast dtype and move tensor from CPU to GPU simutaneously,
# 3.1 But `.to()` can cast dtype and move tensor from GPU to CPU simutaneously.
# 3.2 And this apply to `Module.to` !!!!!
def test_to_dtype_DML():
    if not fkwargs_64_dml:
        return

    for tag, ori_fk, dest_fk in [
        ("yes", fkwargs_32_cpu, fkwargs_64_dml),
        ("yes", fkwargs_64_cpu, fkwargs_32_dml),
        ("no", fkwargs_32_dml, fkwargs_64_cpu),
        ("no", fkwargs_64_dml, fkwargs_32_cpu)
    ]:
        query = torch.randn(3, 4, 8, **ori_fk)
        q1 = query.to(**dest_fk)
        assert q1.device == dest_fk["device"]
        # 1. `.to()` can't cast dtype and move tensor from CPU to DML simutaneously,
        if tag == "yes":
            assert q1.dtype is dest_fk["dtype"]
        # 2. But `.to()` can cast dtype and move tensor from DML to CPU
        #   simutaneously.
        else:
            assert q1.dtype is not dest_fk["dtype"]
            q2 = q1.to(**dest_fk)
            assert q2.device == dest_fk["device"]
            assert q2.dtype is dest_fk["dtype"]


# %%
# 4. Assgignment for slice in DML has been restricted that only slice on
#   the 1st demension could be assigned freely in DML.
def test_assginment_DML():
    if not fkwargs_64_dml:
        return

    comp_dest = torch.ones((4, 4), **fkwargs_32_cpu)
    for tag, fk in [
        ("yes", fkwargs_32_cpu),
        ("yes", fkwargs_64_cpu),
        ("no", fkwargs_32_dml),
        ("no", fkwargs_64_dml),
    ]:
        mt = torch.rand((16, 4), **fk)
        assert mt.size() == (16, 4)

        # 1. Assgignment for slice in DML has been restricted that only slice on
        #   the 1st demension could be assigned freely.
        mt[0:4, :] = comp_dest.to(**fk).to(**fk)
        assert all_close(comp_dest, mt[:4])

        mt[4:8, :] = comp_dest.to(**fk).to(**fk)
        if tag == "yes":
            # Attention: Always put `comp_dest` first as `torch.allclose` doesn't
            #  support `torch.float64` with `equal_nan` set.
            assert all_close(comp_dest, mt[4:8])
        # 2. Assignment will fail.
        else:
            assert not all_close(comp_dest, mt[4:8])

        # 3. But assignment will succeed with second `:` omitted.
        mt[4:8] = comp_dest.to(**fk).to(**fk)
        assert all_close(comp_dest, mt[4:8])

# %%
# 5. `torch.eye` hasn't been implemented for DML.
# 5.1 But just like the assgignment, the `torch.eye(out=some_slice)` DML has
#   been restricted that only slice on the 1st demension could be assigned
#   freely.
#   And so do other inplace calculations such as `ones()`, `zeros()`.
@mark.filterwarnings("ignore: The operator")
def test_eye_DML():
    if not fkwargs_64_dml:
        return

    # 1. `torch.eye` hasn't been implemented for DML and fallback failed too.
    eye_dml = torch.eye(8, **fkwargs_32_dml)
    assert eye_dml.size() == (0,)

    # 2. But just like the assgignment, the `torch.eye(out=some_slice)` DML has
    #   been restricted that only slice on the 1st demension could be assigned
    #   freely.
    comp_dest = torch.eye(4, **fkwargs_32_cpu)
    for tag, fk in [
        ("yes", fkwargs_32_cpu),
        ("yes", fkwargs_64_cpu),
        ("no", fkwargs_32_dml),
        ("no", fkwargs_64_dml),
    ]:
        mt = torch.rand((16, 4), **fk)
        assert mt.size() == (16, 4)

        # 2.1 `torch.eye(out=...)` will succeed.
        torch.eye(4, out=mt[0:4, :], **fk)
        assert all_close(comp_dest, mt[:4])

        # 2.2 `torch.eye(out=...)` will fail.
        torch.eye(4, out=mt[4:8, :], **fk)
        if tag == "yes":
            # Attention: Always put `comp_dest` first as `torch.allclose` doesn't
            #  support `torch.float64` with `equal_nan` set.
            assert all_close(comp_dest, mt[4:8])
        else:
            assert not all_close(comp_dest, mt[4:8])

        # 2.3 `torch.eye(out=...)` will succeed with the second `:` omitted.
        torch.eye(4, out=mt[4:8], **fk)
        assert all_close(comp_dest, mt[4:8])


# %%
# 6. `torch.full(shape, NaN or Inf)` will raise RuntimeError.
def test_full_inf_nan_error():
    if not fkwargs_64_dml:
        return

    for tag, fk in [
        ("yes", fkwargs_32_cpu),
        ("yes", fkwargs_64_cpu),
        ("no", fkwargs_32_dml),
        ("no", fkwargs_64_dml),
    ]:
        device = fk["device"]
        # `full` with normal float works fine.
        tt = torch.full((2, 2), 1., device=device, dtype=torch.bool)
        assert torch.all(tt)

        if tag == "yes":
            tt = torch.full((2, 2), float("nan"), device=device, dtype=torch.bool)
            assert torch.all(tt)
        else:
            # `full` will Inf or NaN with `dtype=torch.bool` will raise
            # RuntimeError in DML.
            with pytest.raises(RuntimeError):
                tt = torch.full((2, 2), float("nan"), device=device, dtype=torch.bool)
            with pytest.raises(RuntimeError):
                tt = torch.full((2, 2), float("inf"), device=device, dtype=torch.bool)


# %%
# 7. `torch.triu` will set the lower-triangle with `nan` wrongly.
def test_triu_nan():
    if not fkwargs_64_dml:
        return

    for tag, fk in [
        ("yes", fkwargs_32_cpu),
        ("yes", fkwargs_64_cpu),
        ("no", fkwargs_32_dml),
    ]:
        onet = torch.full((2, 2), 1., **fk)
        assert all_close(onet, 1.)
        onetu = torch.triu(onet)
        assert all_close(onetu, torch.tensor([[1., 1.], [0., 1.]]))

        inft = torch.full((2, 2), float("inf"), **fk)
        assert torch.all(torch.isinf(inft))
        inftu = torch.triu(inft)
        if tag == "yes":
            ret_ = torch.tensor([
                [float("inf"), float("inf")],
                [0., float("inf")]
            ])
            assert all_close(inftu, ret_)
        else:
            ret_ = torch.tensor([
                [float("inf"), float("inf")],
                [float("nan"), float("inf")]
            ])
            # `torch.triu` will set the lower-triangle with `nan` wrongly.
            assert all_close(inftu, ret_)


    # In fact, `nan` will come up even in the following case.
    inft = torch.full((2, 2), 1., **fkwargs_32_dml)
    inft[1] = torch.inf
    inftu = torch.triu(inft)
    assert torch.isnan(inftu[1, 0])


# %%
# 7.1 `nn.Transformer.generate_square_subsequent_mask` will return `nan` for
#   non-masked position in DML.
def test_nn_Transformer_generate_square_subsequent_mask():
    if not fkwargs_64_dml:
        return

    for tag, fk in [
        ("yes", fkwargs_32_cpu),
        ("yes", fkwargs_64_cpu),
        ("no", fkwargs_32_dml),
        # ("no", fkwargs_64_dml),
    ]:
        mask = nn.Transformer.generate_square_subsequent_mask(8, **fk)
        if tag == "yes":
            assert all_close(torch.tril(mask), 0.0)
        else:
            assert torch.all(torch.isnan(mask[:, 0]))
