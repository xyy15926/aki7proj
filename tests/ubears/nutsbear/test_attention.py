#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_attention.py
#   Author: xyy15926
#   Created: 2025-06-17 15:58:08
#   Updated: 2025-07-05 20:26:09
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
    TransformerEncoderLayer,
    TransformerEncoder,
)


# %%
def test_scaled_dot_product_attention():
    query = torch.randn(3, 4, 5, dtype=torch.float64)
    key = torch.randn(3, 6, 5, dtype=torch.float64)
    value = torch.randn(3, 6, 5, dtype=torch.float64)
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)

    outp, ws = scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask.logical_not())
    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask)
    assert torch.all(torch.isclose(outp, foutp, rtol=1e-4))

    outp, ws = scaled_dot_product_attention(query, key, value, is_causal=True)
    foutp = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    assert torch.all(torch.isclose(outp, foutp, rtol=1e-4))


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
    # 1. The size of last dimension must be 8n.
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

    nnattn, nnw = nnmha(query, key, value)
    attn, attn_ws = mha(query, key, value)
    assert torch.all(torch.isclose(nnattn, attn, rtol=1e-4))

    key_padding_mask = torch.tensor([[1, 1, 1, 1, 1, 0],
                                     [1, 1, 0, 0, 1, 1],
                                     [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    nnattn, nnw = nnmha(query, key, value, key_padding_mask=key_padding_mask)
    attn, attn_ws = mha(query, key, value, key_padding_mask=key_padding_mask)
    assert torch.all(torch.isclose(torch.nan_to_num(nnattn, 0.0), attn, rtol=1e-4))

    nnattn, nnw = nnmha(query, key, value,
                        key_padding_mask=key_padding_mask.logical_not())
    attn, attn_ws = mha(query, key, value,
                        key_padding_mask=key_padding_mask.logical_not())
    assert torch.all(torch.isclose(torch.nan_to_num(nnattn, 0.0), attn, rtol=1e-4))


# %%
def test_TransformerEncoderLayer():
    query = torch.randn(3, 4, 8, dtype=torch.float32)
    # key = torch.randn(3, 6, 8, dtype=torch.float32)
    # value = torch.randn(3, 6, 8, dtype=torch.float32)

    nntel = nn.TransformerEncoderLayer(8, 1, 16, 0, batch_first=True)
    nn_sd = nntel.state_dict()
    sd = {
        "self_attn.in_proj.weight"  : nn_sd["self_attn.in_proj_weight"],
        "self_attn.in_proj.bias"    : nn_sd["self_attn.in_proj_bias"],
        "self_attn.out_proj.weight" : nn_sd["self_attn.out_proj.weight"],
        "self_attn.out_proj.bias"   : nn_sd["self_attn.out_proj.bias"],
        "ffn_linear1.weight"        : nn_sd["linear1.weight"],
        "ffn_linear1.bias"          : nn_sd["linear1.bias"],
        "ffn_linear2.weight"        : nn_sd["linear2.weight"],
        "ffn_linear2.bias"          : nn_sd["linear2.bias"],
        "norm1.weight"              : nn_sd["norm1.weight"],
        "norm1.bias"                : nn_sd["norm1.bias"],
        "norm2.weight"              : nn_sd["norm2.weight"],
        "norm2.bias"                : nn_sd["norm2.bias"],
    }
    tel = TransformerEncoderLayer(8, 1, 16, 0)
    tel.load_state_dict(sd)

    nnret = nntel(query)
    ret = tel(query)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-4))

    key_padding_mask = torch.tensor([[1, 1, 1, 1],
                                     [1, 1, 0, 0],
                                     [1, 1, 1, 1]], dtype=torch.bool)
    nnret = nntel(query, src_key_padding_mask=key_padding_mask)
    ret = tel(query, src_key_padding_mask=key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-4))

    key_padding_mask = torch.tensor([[1, 1, 1, 1],
                                     [1, 1, 0, 0],
                                     [1, 1, 1, 1]], dtype=torch.bool)
    nnret = nntel(query, src_key_padding_mask=key_padding_mask.logical_not())
    ret = tel(query, src_key_padding_mask=key_padding_mask.logical_not())
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-4))


# %%
def test_TransformerEncoder():
    query = torch.randn(3, 4, 8, dtype=torch.float32)
    # key = torch.randn(3, 6, 8, dtype=torch.float32)
    # value = torch.randn(3, 6, 8, dtype=torch.float32)

    nntel = nn.TransformerEncoderLayer(8, 2, 16, 0, batch_first=True)
    nnte = nn.TransformerEncoder(nntel, 2)
    nn_sd = nntel.state_dict()
    sd = {
        "self_attn.in_proj.weight"  : nn_sd["self_attn.in_proj_weight"],
        "self_attn.in_proj.bias"    : nn_sd["self_attn.in_proj_bias"],
        "self_attn.out_proj.weight" : nn_sd["self_attn.out_proj.weight"],
        "self_attn.out_proj.bias"   : nn_sd["self_attn.out_proj.bias"],
        "ffn_linear1.weight"        : nn_sd["linear1.weight"],
        "ffn_linear1.bias"          : nn_sd["linear1.bias"],
        "ffn_linear2.weight"        : nn_sd["linear2.weight"],
        "ffn_linear2.bias"          : nn_sd["linear2.bias"],
        "norm1.weight"              : nn_sd["norm1.weight"],
        "norm1.bias"                : nn_sd["norm1.bias"],
        "norm2.weight"              : nn_sd["norm2.weight"],
        "norm2.bias"                : nn_sd["norm2.bias"],
    }

    tel = TransformerEncoderLayer(8, 2, 16, 0)
    tel.load_state_dict(sd)
    te = TransformerEncoder(tel, 2)

    nnret = nnte(query)
    ret = te(query)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-4))

    key_padding_mask = torch.tensor([[1, 1, 1, 1],
                                     [1, 1, 0, 0],
                                     [1, 1, 1, 1]], dtype=torch.bool)
    nnret = nnte(query, src_key_padding_mask=key_padding_mask)
    ret = te(query, src_key_padding_mask=key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-4))

    key_padding_mask = torch.tensor([[1, 1, 1, 1],
                                     [1, 1, 0, 0],
                                     [1, 1, 1, 1]], dtype=torch.bool)
    nnret = nnte(query, src_key_padding_mask=key_padding_mask.logical_not())
    ret = te(query, src_key_padding_mask=key_padding_mask.logical_not())
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-4))
