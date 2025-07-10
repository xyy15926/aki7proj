#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_attention.py
#   Author: xyy15926
#   Created: 2025-06-17 15:58:08
#   Updated: 2025-07-10 09:30:31
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark
import numpy as np
import torch
from torch import nn
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

    # Attention mask SDPA.
    outp, ws = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask.logical_not())
    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask)
    assert torch.all(torch.isclose(outp, foutp, rtol=1e-3))

    # Causal attention mask SDPA.
    outp, ws = scaled_dot_product_attention(query, key, value, is_causal=True)
    foutp = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    assert torch.all(torch.isclose(outp, foutp, rtol=1e-3))

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
    assert torch.all(torch.isclose(outp, foutp, rtol=1e-3))


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

    # Construct `attn_mask` for `F.scaled_dot_product_attention` manually.
    F_attn_mask = attn_mask.logical_or(
        torch.ones(4, 6, dtype=torch.int).tril(diagonal=0)
        .logical_not()
    )
    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=F_attn_mask,
        key_padding_mask=key_padding_mask.logical_not())
    # `is_causal` and `attn_mask` are merged.
    attn, attn_ws = mha(
        query, key, value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask.logical_not(),
        is_causal=True)
    assert torch.all(torch.isclose(torch.nan_to_num(nnattn, 0.0), attn, rtol=1e-3))

    # Forward with causal attention-mask.
    # 1. `is_causal` in `nn.MultiheadAttention` is just a hint and
    # 2. `nn.MultiheadAttention`(and `F.multi_head_attention_foward`) may
    #   ignore `is_causal` if `attn_mask` is set in non-flash in some cases.
    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=False,
        is_causal=False)
    nnattn_, nnw_ = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=False,
        is_causal=True)
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
