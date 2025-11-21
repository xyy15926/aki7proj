#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_attn.py
#   Author: xyy15926
#   Created: 2025-11-17 15:22:39
#   Updated: 2025-11-20 19:42:37
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
    from ubears.nutsbear import attention
    reload(fixture)
    reload(attention)

from ubears.nutsbear.fixture import (
    fkwargs_32_cpu,
    fkwargs_64_cpu,
    fkwargs_32_dml,
    fkwargs_64_dml,
    all_close,
)
from ubears.nutsbear.attention import (
    MultiheadAttention,
)

torch.autograd.set_detect_anomaly(False)


# %%
# 1. `F.test_F_scaled_dot_product_attention` doesn't support the float64
#   in GPU, which may be just caused by same reason as `torch.allclose`.
def test_F_scaled_dot_product_attention_float64():
    query = torch.randn(3, 4, 8, **fkwargs_64_cpu)
    key = torch.randn(3, 6, 8, **fkwargs_64_cpu)
    value = torch.randn(3, 6, 8, **fkwargs_64_cpu)
    fret_64_cpu = F.scaled_dot_product_attention(query, key, value)

    for tag, fk in [
        ("yes", fkwargs_32_cpu),
        ("yes", fkwargs_32_dml),
        ("no", fkwargs_64_dml),
    ]:
        query = query.to(**fk)
        key = key.to(**fk)
        value = value.to(**fk)
        if tag == "yes":
            fret = F.scaled_dot_product_attention(query, key, value)
            assert all_close(fret, fret_64_cpu)
        # 1. `F.test_F_scaled_dot_product_attention` doesn't support the
        #   float64 in GPU.
        else:
            with pytest.raises(RuntimeError):
                fret = F.scaled_dot_product_attention(query, key, value)


# %%
def test_F_scaled_dot_product_attention_is_causal_3D_4D():
    query = torch.randn(3, 4, 8, **fkwargs_32_cpu)
    key = torch.randn(3, 6, 8, **fkwargs_32_cpu)
    value = torch.randn(3, 6, 8, **fkwargs_32_cpu)

    key_padding_mask = torch.randint(0, 2, (3, 6)).to(torch.bool)
    key_padding_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
    ]).to(torch.bool).logical_not()
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    attn_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
    ]).to(torch.bool).logical_not()

    mask = MultiheadAttention.merge_masks(
        key_padding_mask,
        attn_mask,
        False,
        query
    )
    # 1. `F.scaled_dot_product_attention` will raise error for
    #   3D-QKV when `attn_mask` and `is_causal` is provided together.
    with pytest.raises(RuntimeError):
        fret = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            is_causal=True,
        )

    # 2. But SDPA won't raise error for 4D-QKV in CPU.
    query = query.unsqueeze(1)
    key = key.unsqueeze(1)
    value = value.unsqueeze(1)
    mask = mask.unsqueeze(1)
    fret_ = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=mask,
        is_causal=False,
    )
    fret = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=mask,
        is_causal=True,
    )
    # And the result won't be the same obviously.
    assert not torch.all(torch.isclose(fret, fret_, rtol=1e-3))

    # 3. But SDPA will also raise error for 4D-QKV in GPU.
    query = query.to(**fkwargs_32_dml)
    key = key.to(**fkwargs_32_dml)
    value = value.to(**fkwargs_32_dml)
    with pytest.raises(RuntimeError):
        fret = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask.to(fkwargs_32_dml["device"]),
            is_causal=True,
        )


# %%
def test_F_scaled_dot_product_attention_all_ninf():
    query = torch.randn(3, 4, 2, dtype=torch.float64, requires_grad=True)
    key = torch.randn(3, 6, 2, dtype=torch.float64, requires_grad=True)
    value = torch.randn(3, 6, 2, dtype=torch.float64, requires_grad=True)
    sgd = optim.SGD((query, key, value))
    # attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    # attn_mask[0, :] = False
    # attn_mask[:, 0] = False
    attn_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
    ]).to(torch.bool)

    foutp = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
    )
    fos = foutp.sum()
    fos.backward()

    # -----------------------------------------------------------------
    # Version-based branches:
    # -----------------------------------------------------------------
    # In pytorch <= 2.4:
    # `F.scaled_dot_product_attention` can't handle non-attention query
    # in backward by set the grad with 0.
    if Version(torch.__version__) < Version("2.7"):
        assert torch.isnan(fos)
        assert torch.all(query.grad[:, 0, :].isnan())
        sgd.step()
        assert torch.all(query[:, 0, :].isnan())
    # In pytorch >= 2.8:
    # `F.scaled_dot_product_attention` could handle non-attention query
    # in backward by set the grad with 0.
    else:
        assert torch.all(query.grad[:, 0, :] == 0)
        assert torch.all(query.grad.isnan().logical_not())
        # Key and value's grad are 0 because no query attention to 2nd key.
        assert torch.all(key.grad[:, 2, :] == 0)
        assert torch.all(value.grad[:, 2, :] == 0)
        sgd.step()
        assert torch.all(query.isnan().logical_not())
        assert torch.all(key.isnan().logical_not())
        assert torch.all(value.isnan().logical_not())


# %%
# ATTENTION: `is_causal` in `nn.MultiheadAttention` is just a hint.
def test_nn_MultiHeadAttention_is_causal_hint():
    query = torch.randn(3, 4, 8, dtype=torch.float32)
    key = torch.randn(3, 6, 8, dtype=torch.float32)
    value = torch.randn(3, 6, 8, dtype=torch.float32)

    nnmha = nn.MultiheadAttention(8, 1, batch_first=True)
    nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(8, 1)
    sd = {
        "in_proj.weight": nn_sd["in_proj_weight"],
        "in_proj.bias": nn_sd["in_proj_bias"],
        "out_proj.weight": nn_sd["out_proj.weight"],
        "out_proj.bias": nn_sd["out_proj.bias"],
    }
    mha.load_state_dict(sd)
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)

    # Forward with causal attention-mask.
    # 1. `is_causal` in `nn.MultiheadAttention` is just a hint and
    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=False,
        is_causal=False
    )
    nnattn_, nnw_ = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=False,
        is_causal=True
    )
    assert not all_close(nnattn, nnattn_, 1, 1)

    # 2. `nn.MultiheadAttention`(and `F.multi_head_attention_foward`) may
    #   ignore `is_causal` if `attn_mask` is set in non-flash-SDPA cases,
    #   for`need_weights=True` for example.
    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=True,
        is_causal=False)
    nnattn_, nnw_ = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=True,
        is_causal=True)
    assert all_close(nnattn, nnattn_)


# %%
def test_nn_MultiHeadAttention_all_ninf():
    query = torch.randn(3, 4, 8, dtype=torch.float32)
    key = torch.randn(3, 6, 8, dtype=torch.float32)
    value = torch.randn(3, 6, 8, dtype=torch.float32)
    nnmha = nn.MultiheadAttention(8, 2, batch_first=True)

    # Check `key_padding_mask`:
    key_padding_mask = torch.randint(0, 2, (3, 6)).to(torch.bool)
    key_padding_mask[0, :] = True
    nnattn, nnw = nnmha(
        query, key, value,
        key_padding_mask=key_padding_mask,
        # need_weights=True,
    )
    assert torch.all(torch.isnan(nnattn[0]))
    assert torch.all(torch.isnan(nnw[0]))

    # -----------------------------------------------------------------
    # Version-based branches:
    # -----------------------------------------------------------------
    if Version(torch.__version__) < Version("2.7"):
        nnattn, nnw = nnmha(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        assert torch.all(torch.isnan(nnattn[0]))
        assert nnw is None
    # In pytorch >= 2.8:
    # `nn.MultiHeadAttention` could handle non-attention query in forward
    # and backward process just like `F.scaled_dot_product_attention`.
    else:
        nnattn, nnw = nnmha(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        assert all_close(nnattn[0], 0.0)
        assert nnw is None

    # Check `attn_mask`:
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    attn_mask[0, :] = True
    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        # need_weights=True,
    )
    assert torch.all(torch.isnan(nnattn[:, 0]))
    assert torch.all(torch.isnan(nnw[:, 0]))

    # -----------------------------------------------------------------
    # Version-based branches:
    # -----------------------------------------------------------------
    if Version(torch.__version__) < Version("2.7"):
        nnattn, nnw = nnmha(
            query, key, value,
            attn_mask=attn_mask,
            need_weights=False,
        )
        assert torch.all(torch.isnan(nnattn[:, 0]))
        assert nnw is None
    # In pytorch >= 2.8:
    # `nn.MultiHeadAttention` could handle non-attention query in forward
    # and backward process just like `F.scaled_dot_product_attention`.
    else:
        nnattn, nnw = nnmha(
            query, key, value,
            attn_mask=attn_mask,
            need_weights=False,
        )
        assert all_close(nnattn[:, 0], 0.0)
        assert nnw is None


# %%
# `F.scaled_dot_product_attention` will be called when `need_weights` is not
# set for acceleration and no-NaN in `.grad` after `.backward`, which is
# just like the `nn.MultiheadAttention`.
def test_nn_MultiHeadAttention_all_ninf_backward_grad():
    dtype = torch.float64

    def _grad_data():
        query = torch.randn(3, 4, 2, dtype=dtype, requires_grad=True)
        key = torch.randn(3, 6, 2, dtype=dtype, requires_grad=True)
        value = torch.randn(3, 6, 2, dtype=dtype, requires_grad=True)
        sgd = optim.SGD((query, key, value))

        return query, key, value, sgd

    nnmha = nn.MultiheadAttention(2, 1, batch_first=True, dtype=dtype)

    # Set mask with a query with all key masked.
    key_padding_mask = torch.randint(0, 2, (3, 6)).to(torch.bool)
    key_padding_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
    ]).to(torch.bool).logical_not()
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    attn_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
    ]).to(torch.bool).logical_not()

    # `nn.MultiheadAttention` will return NaN in attention result and NaN in
    # `.grad` for the query with all key masked if `need_weights` is set,
    # in which case the customed SDPA will be called in
    # `F.multi_head_attention_forward` instead of
    # `F.scaled_dot_product_attention`
    # as `F.softmax` will return NaN for all NInf.
    query, key, value, sgd = _grad_data()
    nnattn, nnw = nnmha(
        query, key, value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=True,
    )
    nnattn.sum().backward()
    assert torch.all(query.grad[:, 0, :].isnan())
    assert torch.all(query.grad[0, :, :].isnan())
    assert torch.all(key.grad.isnan())
    assert torch.all(value.grad.isnan())
    sgd.step()
    assert torch.all(query[:, 0, :].isnan())
    assert torch.all(query[0, :, :].isnan())
    assert torch.all(key.isnan())
    assert torch.all(value.isnan())

    # -----------------------------------------------------------------
    # Version-based branches:
    # -----------------------------------------------------------------
    query, key, value, sgd = _grad_data()
    nnattn, nnw = nnmha(
        query, key, value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=False,
    )
    nnattn.sum().backward()
    if Version(torch.__version__) < Version("2.7"):
        assert torch.all(query.grad[:, 0, :].isnan())
        assert torch.all(query.grad[0, :, :].isnan())
        assert torch.all(key.grad.isnan())
        assert torch.all(value.grad.isnan())
        sgd.step()
        assert torch.all(query[:, 0, :].isnan())
        assert torch.all(query[0, :, :].isnan())
        assert torch.all(key.isnan())
        assert torch.all(value.isnan())
    # While `nn.MultiheadAttention` will return 0 in attention result and 0 in
    # `.grad` for the query with all key masked if `need_weights` is not set
    # in which case the calling and the implmentation of
    # `F.scaled_dot_product_attention`.
    else:
        assert torch.all(query.grad[:, 0, :] == 0)
        assert torch.all(query.grad[0, :, :] == 0)
        assert torch.all(key.grad[0, :, :] == 0)
        assert torch.all(key.grad[:, 2, :] == 0)
        assert torch.all(value.grad[0, :, :] == 0)
        assert torch.all(value.grad[:, 2, :] == 0)
        sgd.step()
        assert torch.all(query.isnan().logical_not())
        assert torch.all(key.isnan().logical_not())
        assert torch.all(value.isnan().logical_not())


# %%
def test_nn_TransformerEncoderLayer_all_ninf():
    bsz, slen, tlen, mlen = 3, 4, 6, 4
    src = torch.randn(bsz, slen, 8, dtype=torch.float32)
    tgt = torch.randn(bsz, tlen, 8, dtype=torch.float32)
    mem = torch.randn(bsz, mlen, 8, dtype=torch.float32)

    nntel = nn.TransformerEncoderLayer(8, 2, 16, 0, batch_first=True)
    src_mask = torch.randint(0, 2, (slen, slen)).to(torch.bool)
    src_mask[0] = 1
    nnret = nntel(src, src_mask=src_mask)
    src_key_padding_mask = torch.randint(0, 2, (bsz, slen)).to(torch.bool)
    src_key_padding_mask[0] = 1
    nnret2 = nntel(src, src_key_padding_mask=src_key_padding_mask)
    # -----------------------------------------------------------------
    # Version-based branches:
    # -----------------------------------------------------------------
    if Version(torch.__version__) < Version("2.7"):
        assert torch.all(nnret.isnan()[:, 0])
        assert torch.all(nnret2.isnan()[0])
    else:
        assert not torch.any(nnret.isnan())
        assert not torch.any(nnret2.isnan())
