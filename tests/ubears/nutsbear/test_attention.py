#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_attention.py
#   Author: xyy15926
#   Created: 2025-06-17 15:58:08
#   Updated: 2025-07-14 19:35:47
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
from torch.nested._internal import sdpa
from torch.backends.cuda import SDPAParams

if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import attention
    reload(attention)

from ubears.nutsbear.attention import (
    scaled_dot_product_attention,
    MultiheadAttention,
)


# %%
def test_scaled_dot_product_attention():
    query = torch.randn(3, 4, 5, dtype=torch.float64)
    key = torch.randn(3, 6, 5, dtype=torch.float64)
    value = torch.randn(3, 6, 5, dtype=torch.float64)
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    attn_mask_3D = torch.randint(0, 2, (3, 4, 6)).to(torch.bool)

    # Attention mask SDPA.
    outp, ws = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask.logical_not())
    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask)
    assert torch.all(torch.isclose(torch.nan_to_num(outp, 0.0), foutp, rtol=1e-3))

    outp, ws = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask_3D.logical_not())
    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask_3D)
    assert torch.all(torch.isclose(torch.nan_to_num(outp, 0.0), foutp, rtol=1e-3))

    # Causal attention mask SDPA.
    outp, ws = scaled_dot_product_attention(query, key, value, is_causal=True)
    foutp = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    assert torch.all(torch.isclose(torch.nan_to_num(outp, 0.0), foutp, rtol=1e-3))

    # Mixed attention mask SDPA.
    # `attn_mask` and `is_causal` shouldn't be set simultaneously in
    # `F.scaled_dot_product_attention`.
    with pytest.raises(RuntimeError):
        foutp = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, is_causal=True)
    outp, ws = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, is_causal=True)
    # Construct `attn_mask` for `F.scaled_dot_product_attention` manually.
    F_attn_mask = attn_mask + 1 - torch.ones(4, 6, dtype=torch.int).tril(diagonal=0)
    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=F_attn_mask.logical_not())
    assert torch.all(torch.isclose(torch.nan_to_num(outp, 0.0), foutp, rtol=1e-3))


# %%
def test_scaled_dot_product_attention_backward_grad():
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

    output, ws = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask.logical_not()
    )
    os = output.sum()
    os.backward()
    assert torch.all(query.grad[:, 0, :].isnan())
    assert torch.all(query.grad[:, 1:, :].isnan().logical_not())
    assert torch.all(key.grad.isnan())
    assert torch.all(value.grad.isnan())
    sgd.step()
    assert torch.all(query[:, 0, :].isnan())
    assert torch.all(key.isnan())
    assert torch.all(value.isnan())

    # `F.scaled_dot_product_attention` could handle non-attention query
    # in backward by set the grad with 0.
    query = torch.randn(3, 4, 2, dtype=torch.float64, requires_grad=True)
    key = torch.randn(3, 6, 2, dtype=torch.float64, requires_grad=True)
    value = torch.randn(3, 6, 2, dtype=torch.float64, requires_grad=True)
    sgd = optim.SGD((query, key, value))

    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask
    )
    fos = foutp.sum()
    fos.backward()
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
@mark.filterwarnings("ignore: .*PyTorch API of nested tensors is in prototype.*")
def test_scaled_dot_product_attention_njt():
    # 1. The size of last dimension must be 8n.
    query = torch.randn(3, 4, 8, dtype=torch.float64)
    key = torch.randn(3, 6, 8, dtype=torch.float64)
    value = torch.randn(3, 6, 8, dtype=torch.float64)

    def build_nested(query, key, value, qlens, kvlens):
        # 2. The ragged-dim can't be the dimension for `heads_n`, which will be
        #   regarded as the dim-1 for 3D Tensor.
        # (bsz, slen, dim) => (bsz, slen, heads_n, qsz)
        # (3, 4, 8) => (3, j2, 8) => (3, j2, 1, 8)
        query = nested.narrow(
            query, dim=1, start=0, length=qlens, layout=torch.jagged
        ).contiguous().unflatten(-1, (-1, 8)).transpose(2, 1)
        key = nested.narrow(
            key, dim=1, start=0, length=kvlens, layout=torch.jagged
        ).contiguous().unflatten(-1, (-1, 8)).transpose(2, 1)
        value = nested.narrow(
            value, dim=1, start=0, length=kvlens, layout=torch.jagged
        ).contiguous().unflatten(-1, (-1, 8)).transpose(2, 1)

        return query, key, value

    qlens = torch.IntTensor([2, 4, 3])
    kvlens = qlens
    q, k, v = build_nested(query, key, value, qlens, kvlens)
    params = SDPAParams(q, k, v, None, 0.0, False, False)
    assert sdpa._can_use_flash_sdpa_jagged(params, True)
    assert sdpa._can_use_efficient_sdpa_jagged(params, True)
    assert sdpa._can_use_math_sdpa_jagged(params, True)

    # `F.scaled_dot_product_attention` does a lot of preprocess for
    # NestedTensor.
    # https://github.com/pytorch/pytorch/blob/main/torch/nested/_internal/sdpa.py
    foutp = F.scaled_dot_product_attention(q, k, v)
    with pytest.raises(RuntimeError):
        # `@` doesn't support (..., j1, 8n) (..., 8n, j2)
        scaled_dot_product_attention(q, k, v)

    kvlens = torch.IntTensor([3, 5, 4])
    q, k, v = build_nested(query, key, value, qlens, kvlens)
    params = SDPAParams(q, k, v, None, 0.0, False, False)
    assert sdpa._can_use_flash_sdpa_jagged(params, True)
    assert sdpa._can_use_efficient_sdpa_jagged(params, True)
    assert sdpa._can_use_math_sdpa_jagged(params, True)
    foutp = F.scaled_dot_product_attention(q, k, v)
    assert foutp is not None


# %%
def test_MultiHeadAttention():
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

    # Default forward.
    nnattn, nnw = nnmha(query, key, value)
    attn, attn_ws = mha(query, key, value)
    assert torch.all(torch.isclose(nnattn, attn, rtol=1e-3))

    # Forward with key-padding-mask.
    key_padding_mask = torch.randint(0, 2, (3, 6)).to(torch.bool)
    nnattn, nnw = nnmha(query, key, value, key_padding_mask=key_padding_mask)
    attn, attn_ws = mha(query, key, value, key_padding_mask=key_padding_mask)
    assert torch.all(torch.isclose(torch.nan_to_num(nnattn, 0.0), attn, rtol=1e-3))

    nnattn, nnw = nnmha(
        query, key, value,
        key_padding_mask=key_padding_mask.logical_not())
    attn, attn_ws = mha(
        query, key, value,
        key_padding_mask=key_padding_mask.logical_not())
    assert torch.all(torch.isclose(torch.nan_to_num(nnattn, 0.0), attn, rtol=1e-3))

    # Forward with attention-mask only.
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=False,
    )
    nnattn_w, nnw_w = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=True,
    )
    attn, attn_ws = mha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=False,
    )
    attn_w, attn_ws_w = mha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=True,
    )
    assert torch.all(torch.isclose(nnattn, attn, rtol=1e-3))
    assert torch.all(torch.isclose(nnattn_w, attn_w, rtol=1e-3, equal_nan=True))
    assert torch.all(torch.isclose(torch.nan_to_num(attn, 0.0), attn_w, rtol=1e-3))

    # Forward with attention-mask(or mixed mask).
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask)
    attn, attn_ws = mha(
        query, key, value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask)
    assert torch.all(torch.isclose(torch.nan_to_num(nnattn, 0.0), attn, rtol=1e-3))

    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=attn_mask.logical_not(),
        key_padding_mask=key_padding_mask.logical_not())
    attn, attn_ws = mha(
        query, key, value,
        attn_mask=attn_mask.logical_not(),
        key_padding_mask=key_padding_mask.logical_not())
    assert torch.all(torch.isclose(torch.nan_to_num(nnattn, 0.0), attn, rtol=1e-3))


# %%
def test_MultiHeadAttention_qkv_diffsz():
    query = torch.randn(3, 4, 8, dtype=torch.float32)
    key = torch.randn(3, 6, 16, dtype=torch.float32)
    value = torch.randn(3, 6, 16, dtype=torch.float32)

    nnmha = nn.MultiheadAttention(
        8, 1,
        bias=True,
        kdim=16, vdim=16,
        batch_first=True)
    nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(
        8, 1,
        ksz=16, vsz=16)
    qb, kb, vb = nn_sd["in_proj_bias"].chunk(3)
    sd = {
        "q_proj.weight": nn_sd["q_proj_weight"],
        "k_proj.weight": nn_sd["k_proj_weight"],
        "v_proj.weight": nn_sd["v_proj_weight"],
        "q_proj.bias": qb,
        "k_proj.bias": kb,
        "v_proj.bias": vb,
        "out_proj.weight": nn_sd["out_proj.weight"],
        "out_proj.bias": nn_sd["out_proj.bias"],
    }
    mha.load_state_dict(sd)

    # Default forward.
    nnattn, nnw = nnmha(query, key, value)
    attn, attn_ws = mha(query, key, value)
    assert torch.all(torch.isclose(nnattn, attn, rtol=1e-3))


# %%
def test_MHA_merge_masks():
    dtype = torch.float64

    # 4D-QKV and mask will be used here so that
    # `F.scaled_dot_product_attention` won't raise RuntimeError.
    query = torch.randn(3, 1, 4, 2, dtype=dtype, requires_grad=True)
    key = torch.randn(3, 1, 6, 2, dtype=dtype, requires_grad=True)
    value = torch.randn(3, 1, 6, 2, dtype=dtype, requires_grad=True)

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

    # Causality from `is_causal` or `attn_mask` lead to the same result.
    def check_SDPA_and_causal(query, key, value, non_causal_mask, causal_mask):
        nc_ret, nc_ws = scaled_dot_product_attention(
            query, key, value,
            attn_mask=non_causal_mask,
            is_causal=True,
        )
        cn_ret, c_ws = scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_mask,
            is_causal=False,
        )
        cc_ret, c_ws = scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_mask,
            is_causal=True,
        )
        nc_fret = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=non_causal_mask,
            is_causal=True,
        )
        cn_fret = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_mask,
            is_causal=False,
        )
        cc_fret = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_mask,
            is_causal=True,
        )
        assert torch.all(torch.isclose(nc_ret, cn_ret, rtol=1e-3, equal_nan=True))
        assert torch.all(torch.isclose(nc_ret, cc_ret, rtol=1e-3, equal_nan=True))
        assert torch.all(torch.isclose(nc_fret, cn_fret, rtol=1e-3))
        assert torch.all(torch.isclose(nc_fret, cc_fret, rtol=1e-3))
        assert torch.all(torch.isclose(torch.nan_to_num(nc_ret, 0.0), nc_fret, rtol=1e-3))

    # Check `.merge_masks` with different kinds of parameters.
    non_causal_mask = MultiheadAttention.merge_masks(
        key_padding_mask, attn_mask, False, query, key
    ).unsqueeze(1)
    causal_mask = MultiheadAttention.merge_masks(
        key_padding_mask, attn_mask, True, query, key
    ).unsqueeze(1)
    check_SDPA_and_causal(query, key, value, non_causal_mask, causal_mask)

    non_causal_mask = MultiheadAttention.merge_masks(
        key_padding_mask, None, False, query, key
    ).unsqueeze(1)
    causal_mask = MultiheadAttention.merge_masks(
        key_padding_mask, None, True, query, key
    ).unsqueeze(1)
    check_SDPA_and_causal(query, key, value, non_causal_mask, causal_mask)

    non_causal_mask = MultiheadAttention.merge_masks(
        None, attn_mask, False, query, key
    ).unsqueeze(1)
    causal_mask = MultiheadAttention.merge_masks(
        None, attn_mask, True, query, key
    ).unsqueeze(1)
    check_SDPA_and_causal(query, key, value, non_causal_mask, causal_mask)


# %%
def test_F_scaled_dot_product_attention_is_causal():
    dtype = torch.float64
    query = torch.randn(3, 4, 8, dtype=dtype, requires_grad=True)
    key = torch.randn(3, 6, 8, dtype=dtype, requires_grad=True)
    value = torch.randn(3, 6, 8, dtype=dtype, requires_grad=True)

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
    # `F.scaled_dot_product_attention` will raise error for
    # 3D-QKV when `attn_mask` and `is_causal` is provided together.
    with pytest.raises(RuntimeError):
        fret = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            is_causal=True,
        )
    # But not for 4D-QKV.
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


# %%
# ATTENTION: `is_causal` in `nn.MultiheadAttention` is just a hint.
def test_NN_MultiHeadAttention_is_causal_hint():
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
    key_padding_mask = torch.randint(0, 2, (3, 6)).to(torch.bool)

    # Construct `attn_mask` for `F.scaled_dot_product_attention` manually.
    F_attn_mask = attn_mask.logical_or(
        torch.ones(4, 6, dtype=torch.int).tril(diagonal=0)
        .logical_not()
    )
    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=F_attn_mask,
        key_padding_mask=key_padding_mask.logical_not()
    )
    # `is_causal` and `attn_mask` are merged.
    attn, attn_ws = mha(
        query, key, value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask.logical_not(),
        is_causal=True
    )
    assert torch.all(torch.isclose(torch.nan_to_num(nnattn, 0.0), attn, rtol=1e-3))

    # Forward with causal attention-mask.
    # 1. `is_causal` in `nn.MultiheadAttention` is just a hint and
    # 2. `nn.MultiheadAttention`(and `F.multi_head_attention_foward`) may
    #   ignore `is_causal` if `attn_mask` is set in non-flash in some cases.
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
    assert not torch.all(torch.isclose(torch.nan_to_num(nnattn, 0.0),
                                       torch.nan_to_num(nnattn_, 0.0),
                                       rtol=1e-3))
    # 3. `need_weights=True` for example.
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
    assert torch.all(torch.isclose(torch.nan_to_num(nnattn, 0.0),
                                   torch.nan_to_num(nnattn_, 0.0),
                                   rtol=1e-3))


# %%
def test_NN_MultiHeadAttention_backward_grad():
    dtype = torch.float64

    def _grad_data():
        query = torch.randn(3, 4, 2, dtype=dtype, requires_grad=True)
        key = torch.randn(3, 6, 2, dtype=dtype, requires_grad=True)
        value = torch.randn(3, 6, 2, dtype=dtype, requires_grad=True)
        sgd = optim.SGD((query, key, value))

        return query, key, value, sgd

    nnmha = nn.MultiheadAttention(2, 1, batch_first=True, dtype=dtype)
    # nn_sd = nnmha.state_dict()
    # mha = MultiheadAttention(2, 1, dtype=dtype)
    # sd = {
    #     "in_proj.weight": nn_sd["in_proj_weight"],
    #     "in_proj.bias": nn_sd["in_proj_bias"],
    #     "out_proj.weight": nn_sd["out_proj.weight"],
    #     "out_proj.bias": nn_sd["out_proj.bias"],
    # }
    # mha.load_state_dict(sd)

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
    # which will use the customed SDPA in `F.multi_head_attention_forward`
    # instead of calling `F.scaled_dot_product_attention`
    # Because `F.softmax` will return NaN for all NInf.
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

    # While `nn.MultiheadAttention` will return 0 in attention result and 0 in
    # `.grad` for the query with all key masked if `need_weights` is not set
    # for the calling and the implmentation of
    # `F.scaled_dot_product_attention`.
    query, key, value, sgd = _grad_data()
    nnattn, nnw = nnmha(
        query, key, value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=False,
    )
    nnattn.sum().backward()
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
# `F.scaled_dot_product_attention` will be called when `need_weights` is not
# set for acceleration and no-NaN in `.grad` after `.backward`, which is
# just like the `nn.MultiheadAttention`.
def test_MultiHeadAttention_backward_grad():
    dtype = torch.float64

    def _grad_data():
        query = torch.randn(3, 4, 2, dtype=dtype, requires_grad=True)
        key = torch.randn(3, 6, 2, dtype=dtype, requires_grad=True)
        value = torch.randn(3, 6, 2, dtype=dtype, requires_grad=True)
        sgd = optim.SGD((query, key, value))

        return query, key, value, sgd

    # nnmha = nn.MultiheadAttention(2, 1, batch_first=True, dtype=dtype)
    # nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(2, 1, dtype=dtype)
    # sd = {
    #     "in_proj.weight": nn_sd["in_proj_weight"],
    #     "in_proj.bias": nn_sd["in_proj_bias"],
    #     "out_proj.weight": nn_sd["out_proj.weight"],
    #     "out_proj.bias": nn_sd["out_proj.bias"],
    # }
    # mha.load_state_dict(sd)

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

    # `MultiheadAttention` will return NaN in attention result and NaN in
    # `.grad` for the query with all key masked if `need_weights` is set,
    # which will use the customed SDPA in `multi_head_attention_forward`
    # instead of calling `F.scaled_dot_product_attention`
    # Because `F.softmax` will return NaN for all NInf.
    query, key, value, sgd = _grad_data()
    nnattn, nnw = mha(
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

    # While `MultiheadAttention` will return 0 in attention result and 0 in
    # `.grad` for the query with all key masked if `need_weights` is not set
    # for the calling and the implmentation of
    # `F.scaled_dot_product_attention`.
    query, key, value, sgd = _grad_data()
    nnattn, nnw = mha(
        query, key, value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=False,
    )
    nnattn.sum().backward()
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
