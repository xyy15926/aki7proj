#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_transformer.py
#   Author: xyy15926
#   Created: 2025-07-10 09:29:34
#   Updated: 2025-07-22 22:54:49
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
    from ubears.nutsbear import attention, transformer
    reload(attention)
    reload(transformer)

from ubears.nutsbear.transformer import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
    Transformer,
)


# %%
def test_TransformerEncoderLayer():
    bsz, slen, tlen, mlen = 3, 4, 6, 4
    src = torch.randn(bsz, slen, 8, dtype=torch.float32)
    tgt = torch.randn(bsz, tlen, 8, dtype=torch.float32)
    mem = torch.randn(bsz, mlen, 8, dtype=torch.float32)

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

    # Default forward.
    nnret = nntel(src)
    ret = tel(src)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with key-padding-mask.
    src_key_padding_mask = torch.randint(0, 2, (bsz, slen)).to(torch.bool)
    nnret = nntel(src, src_key_padding_mask=src_key_padding_mask)
    ret = tel(src, src_key_padding_mask=src_key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with attention-mask only.
    src_mask = torch.randint(0, 2, (slen, slen)).to(torch.bool)
    nnret = nntel(src, src_mask=src_mask)
    ret = tel(src, src_mask=src_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with attention-mask.
    src_mask = torch.randint(0, 2, (slen, slen)).to(torch.bool)
    nnret = nntel(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    ret = tel(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with causal attention-mask.
    # `is_causal` in `nn.TransformerEncoderLayer` is just a hint and
    # `attn_mask`(`src_mask`) must be set if `is_causal` is set.
    nn_src_mask = nn.Transformer.generate_square_subsequent_mask(slen)
    nnret = nntel(src, src_mask=nn_src_mask, is_causal=False)
    nnret2 = nntel(src, src_mask=nn_src_mask, is_causal=True)
    ret = tel(src, src_mask=nn_src_mask, is_causal=True)
    assert torch.all(torch.isclose(nnret, nnret2, rtol=1e-3))
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))


# %%
def test_TransformerEncoderLayer_with_SimpleMHA():
    bsz, slen, tlen, mlen = 3, 4, 6, 4
    src = torch.randn(bsz, slen, 8, dtype=torch.float32)
    tgt = torch.randn(bsz, tlen, 8, dtype=torch.float32)
    mem = torch.randn(bsz, mlen, 8, dtype=torch.float32)

    tel = TransformerEncoderLayer(8, 1, 16, 0, attn_style="qwk")

    # Default forward.
    ret = tel(src)
    assert ret.size() == (bsz, slen, 8)

    # Forward with key-padding-mask.
    src_key_padding_mask = torch.randint(0, 2, (bsz, slen)).to(torch.bool)
    ret = tel(src, src_key_padding_mask=src_key_padding_mask)
    assert ret.size() == (bsz, slen, 8)

    # Forward with attention-mask only.
    src_mask = torch.randint(0, 2, (slen, slen)).to(torch.bool)
    ret = tel(src, src_mask=src_mask)
    assert ret.size() == (bsz, slen, 8)

    # Forward with attention-mask.
    src_mask = torch.randint(0, 2, (slen, slen)).to(torch.bool)
    ret = tel(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    assert ret.size() == (bsz, slen, 8)

    # Forward with causal attention-mask.
    # `is_causal` in `nn.TransformerEncoderLayer` is just a hint and
    # `attn_mask`(`src_mask`) must be set if `is_causal` is set.
    nn_src_mask = nn.Transformer.generate_square_subsequent_mask(slen)
    ret = tel(src, src_mask=nn_src_mask, is_causal=True)
    ret_ = tel(src, src_mask=nn_src_mask, is_causal=False)
    assert torch.all(torch.isclose(ret, ret_, rtol=1e-3))


# %%
def test_TransformerEncoder():
    bsz, slen, tlen, mlen = 3, 4, 6, 4
    src = torch.randn(bsz, slen, 8, dtype=torch.float32)
    tgt = torch.randn(bsz, tlen, 8, dtype=torch.float32)
    mem = torch.randn(bsz, mlen, 8, dtype=torch.float32)

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

    # Default forward.
    nnret = nnte(src)
    ret = te(src)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with key-padding-mask.
    src_key_padding_mask = torch.randint(0, 2, (bsz, slen)).to(torch.bool)
    nnret = nnte(src, src_key_padding_mask=src_key_padding_mask)
    ret = te(src, src_key_padding_mask=src_key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with attention-mask only.
    src_mask = torch.randint(0, 2, (4, 4)).to(torch.bool)
    nnret = nnte(src, mask=src_mask)
    ret = te(src, src_mask=src_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with attention-mask.
    src_mask = torch.randint(0, 2, (4, 4)).to(torch.bool)
    nnret = nnte(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    ret = te(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with causal attention-mask.
    # `is_causal` in `nn.TransformerEncoderLayer` is just a hint and
    # `attn_mask`(`src_mask`) must be set if `is_causal` is set.
    nn_src_mask = nn.Transformer.generate_square_subsequent_mask(4)
    nnret = nnte(src, mask=nn_src_mask, is_causal=False)
    nnret2 = nnte(src, mask=nn_src_mask, is_causal=True)
    ret = te(src, is_causal=True)
    ret2 = te(src, src_mask=nn_src_mask, is_causal=True)
    assert torch.all(torch.isclose(nnret, nnret2, rtol=1e-3))
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))
    assert torch.all(torch.isclose(ret, ret2, rtol=1e-3))


# %%
def test_TransformerEncoder_with_SimpleMHA():
    bsz, slen, tlen, mlen = 3, 4, 6, 4
    src = torch.randn(bsz, slen, 8, dtype=torch.float32)
    tgt = torch.randn(bsz, tlen, 8, dtype=torch.float32)
    mem = torch.randn(bsz, mlen, 8, dtype=torch.float32)

    tel = TransformerEncoderLayer(8, 1, 16, 0, attn_style="qwk")
    te = TransformerEncoder(tel, 2)

    # Default forward.
    ret = te(src)
    assert ret.size() == (bsz, slen, 8)

    # Forward with key-padding-mask.
    src_key_padding_mask = torch.randint(0, 2, (bsz, slen)).to(torch.bool)
    ret = te(src, src_key_padding_mask=src_key_padding_mask)
    assert ret.size() == (bsz, slen, 8)

    # Forward with attention-mask only.
    src_mask = torch.randint(0, 2, (slen, slen)).to(torch.bool)
    ret = te(src, src_mask=src_mask)
    assert ret.size() == (bsz, slen, 8)

    # Forward with attention-mask.
    src_mask = torch.randint(0, 2, (slen, slen)).to(torch.bool)
    ret = te(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    assert ret.size() == (bsz, slen, 8)

    # Forward with causal attention-mask.
    # `is_causal` in `nn.TransformerEncoderLayer` is just a hint and
    # `attn_mask`(`src_mask`) must be set if `is_causal` is set.
    nn_src_mask = nn.Transformer.generate_square_subsequent_mask(slen)
    ret = te(src, src_mask=nn_src_mask, is_causal=True)
    ret_ = te(src, src_mask=nn_src_mask, is_causal=False)
    assert torch.all(torch.isclose(ret, ret_, rtol=1e-3))

# %%
def test_TransformerDecoderLayer():
    bsz, slen, tlen, mlen = 3, 4, 6, 4
    src = torch.randn(bsz, slen, 8, dtype=torch.float32)
    tgt = torch.randn(bsz, tlen, 8, dtype=torch.float32)
    mem = torch.randn(bsz, mlen, 8, dtype=torch.float32)

    nntdl = nn.TransformerDecoderLayer(8, 2, 16, 0.0, batch_first=True)
    tdl = TransformerDecoderLayer(8, 2, 16, 0.0)
    nn_sd = nntdl.state_dict()
    sd = {
        "self_attn.in_proj.weight"  : nn_sd["self_attn.in_proj_weight"],
        "self_attn.in_proj.bias"    : nn_sd["self_attn.in_proj_bias"],
        "self_attn.out_proj.weight" : nn_sd["self_attn.out_proj.weight"],
        "self_attn.out_proj.bias"   : nn_sd["self_attn.out_proj.bias"],
        "cross_attn.in_proj.weight" : nn_sd["multihead_attn.in_proj_weight"],
        "cross_attn.in_proj.bias"   : nn_sd["multihead_attn.in_proj_bias"],
        "cross_attn.out_proj.weight": nn_sd["multihead_attn.out_proj.weight"],
        "cross_attn.out_proj.bias"  : nn_sd["multihead_attn.out_proj.bias"],
        "ffn_linear1.weight"        : nn_sd["linear1.weight"],
        "ffn_linear1.bias"          : nn_sd["linear1.bias"],
        "ffn_linear2.weight"        : nn_sd["linear2.weight"],
        "ffn_linear2.bias"          : nn_sd["linear2.bias"],
        "norm1.weight"              : nn_sd["norm1.weight"],
        "norm1.bias"                : nn_sd["norm1.bias"],
        "norm2.weight"              : nn_sd["norm2.weight"],
        "norm2.bias"                : nn_sd["norm2.bias"],
        "norm3.weight"              : nn_sd["norm3.weight"],
        "norm3.bias"                : nn_sd["norm3.bias"],
    }
    tdl.load_state_dict(sd)

    # Default forward.
    nnret = nntdl(tgt, mem, tgt_is_causal=False)
    ret = tdl(tgt, mem, tgt_is_causal=False)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with key-padding-mask.
    memory_key_padding_mask = torch.randint(0, 2, (bsz, mlen)).to(torch.bool)
    nnret = nntdl(tgt, mem, memory_key_padding_mask=memory_key_padding_mask)
    ret = tdl(tgt, mem, memory_key_padding_mask=memory_key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    tgt_key_padding_mask = torch.randint(0, 2, (bsz, tlen)).to(torch.bool)
    nnret = nntdl(
        tgt, mem,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask)
    ret = tdl(
        tgt, mem,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with self-attention-mask only.
    tgt_mask = torch.randint(0, 2, (tlen, tlen)).to(torch.bool)
    nnret = nntdl(
        tgt, mem,
        tgt_mask=tgt_mask,
        tgt_is_causal=False,
    )
    ret = tdl(
        tgt, mem,
        tgt_mask=tgt_mask,
        tgt_is_causal=False
    )
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with self-attention-mask and key-padding-mask.
    tgt_mask = torch.randint(0, 2, (tlen, tlen)).to(torch.bool)
    nnret = nntdl(
        tgt, mem,
        tgt_mask=tgt_mask,
        tgt_is_causal=False,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    ret = tdl(
        tgt, mem,
        tgt_mask=tgt_mask,
        tgt_is_causal=False,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with causal mask.
    nn_tgt_mask = nn.Transformer.generate_square_subsequent_mask(tlen)
    nnret = nntdl(tgt, mem, tgt_is_causal=True, tgt_mask=nn_tgt_mask)
    ret = tdl(tgt, mem, tgt_is_causal=True)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))


# %%
def test_TransformerDecoderLayer_with_SimpleMHA():
    bsz, slen, tlen, mlen = 3, 4, 6, 4
    src = torch.randn(bsz, slen, 8, dtype=torch.float32)
    tgt = torch.randn(bsz, tlen, 8, dtype=torch.float32)
    mem = torch.randn(bsz, mlen, 8, dtype=torch.float32)

    tdl = TransformerDecoderLayer(8, 2, 16, 0.0, attn_style="qwk")

    # Default forward.
    ret = tdl(tgt, mem, tgt_is_causal=False)
    assert ret.size() == (bsz, tlen, 8)

    # Forward with key-padding-mask.
    memory_key_padding_mask = torch.randint(0, 2, (bsz, mlen)).to(torch.bool)
    ret = tdl(tgt, mem, memory_key_padding_mask=memory_key_padding_mask)
    assert ret.size() == (bsz, tlen, 8)

    tgt_key_padding_mask = torch.randint(0, 2, (bsz, tlen)).to(torch.bool)
    ret = tdl(
        tgt, mem,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask
    )
    assert ret.size() == (bsz, tlen, 8)

    # Forward with self-attention-mask only.
    tgt_mask = torch.randint(0, 2, (tlen, tlen)).to(torch.bool)
    ret = tdl(
        tgt, mem,
        tgt_mask=tgt_mask,
        tgt_is_causal=False
    )
    assert ret.size() == (bsz, tlen, 8)

    # Forward with self-attention-mask and key-padding-mask.
    tgt_mask = torch.randint(0, 2, (tlen, tlen)).to(torch.bool)
    ret = tdl(
        tgt, mem,
        tgt_mask=tgt_mask,
        tgt_is_causal=False,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    assert ret.size() == (bsz, tlen, 8)

    # Forward with causal mask.
    nn_tgt_mask = nn.Transformer.generate_square_subsequent_mask(tlen)
    ret = tdl(tgt, mem, tgt_is_causal=True)
    ret2 = tdl(tgt, mem, tgt_is_causal=False, tgt_mask=nn_tgt_mask)
    assert torch.all(torch.isclose(ret2, ret, rtol=1e-3))


# %%
def test_TransformerDecoder():
    bsz, slen, tlen, mlen = 3, 4, 6, 4
    src = torch.randn(bsz, slen, 8, dtype=torch.float32)
    tgt = torch.randn(bsz, tlen, 8, dtype=torch.float32)
    mem = torch.randn(bsz, mlen, 8, dtype=torch.float32)

    nntdl = nn.TransformerDecoderLayer(8, 2, 16, 0.0, batch_first=True)
    tdl = TransformerDecoderLayer(8, 2, 16, 0.0)
    nn_sd = nntdl.state_dict()
    sd = {
        "self_attn.in_proj.weight"  : nn_sd["self_attn.in_proj_weight"],
        "self_attn.in_proj.bias"    : nn_sd["self_attn.in_proj_bias"],
        "self_attn.out_proj.weight" : nn_sd["self_attn.out_proj.weight"],
        "self_attn.out_proj.bias"   : nn_sd["self_attn.out_proj.bias"],
        "cross_attn.in_proj.weight" : nn_sd["multihead_attn.in_proj_weight"],
        "cross_attn.in_proj.bias"   : nn_sd["multihead_attn.in_proj_bias"],
        "cross_attn.out_proj.weight": nn_sd["multihead_attn.out_proj.weight"],
        "cross_attn.out_proj.bias"  : nn_sd["multihead_attn.out_proj.bias"],
        "ffn_linear1.weight"        : nn_sd["linear1.weight"],
        "ffn_linear1.bias"          : nn_sd["linear1.bias"],
        "ffn_linear2.weight"        : nn_sd["linear2.weight"],
        "ffn_linear2.bias"          : nn_sd["linear2.bias"],
        "norm1.weight"              : nn_sd["norm1.weight"],
        "norm1.bias"                : nn_sd["norm1.bias"],
        "norm2.weight"              : nn_sd["norm2.weight"],
        "norm2.bias"                : nn_sd["norm2.bias"],
        "norm3.weight"              : nn_sd["norm3.weight"],
        "norm3.bias"                : nn_sd["norm3.bias"],
    }
    tdl.load_state_dict(sd)
    nntd = nn.TransformerDecoder(nntdl, 6)
    td = TransformerDecoder(tdl, 6)

    # Default forward.
    nnret = nntd(tgt, mem, tgt_is_causal=False)
    ret = td(tgt, mem, tgt_is_causal=False)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with key-padding-mask.
    memory_key_padding_mask = torch.randint(0, 2, (bsz, mlen)).to(torch.bool)
    nnret = nntd(tgt, mem, memory_key_padding_mask=memory_key_padding_mask)
    ret = td(tgt, mem, memory_key_padding_mask=memory_key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with self-attention-mask and key-padding-mask.
    tgt_mask = torch.randint(0, 2, (tlen, tlen)).to(torch.bool)
    nnret = nntd(
        tgt, mem,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=memory_key_padding_mask)
    ret = td(
        tgt, mem,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=memory_key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))

    # Forward with causal mask.
    nn_tgt_mask = nn.Transformer.generate_square_subsequent_mask(tlen)
    nnret = nntd(tgt, mem, tgt_is_causal=True, tgt_mask=nn_tgt_mask)
    ret = td(tgt, mem, tgt_is_causal=True)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-3))


# %%
def test_TransformerDecoder_with_SimpleMHA():
    bsz, slen, tlen, mlen = 3, 4, 6, 4
    src = torch.randn(bsz, slen, 8, dtype=torch.float32)
    tgt = torch.randn(bsz, tlen, 8, dtype=torch.float32)
    mem = torch.randn(bsz, mlen, 8, dtype=torch.float32)

    tdl = TransformerDecoderLayer(8, 2, 16, 0.0, attn_style="qwk")
    td = TransformerDecoder(tdl, 2)

    # Default forward.
    ret = td(tgt, mem, tgt_is_causal=False)
    assert ret.size() == (bsz, tlen, 8)

    # Forward with key-padding-mask.
    memory_key_padding_mask = torch.randint(0, 2, (bsz, mlen)).to(torch.bool)
    ret = td(tgt, mem, memory_key_padding_mask=memory_key_padding_mask)
    assert ret.size() == (bsz, tlen, 8)

    tgt_key_padding_mask = torch.randint(0, 2, (bsz, tlen)).to(torch.bool)
    ret = td(
        tgt, mem,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask
    )
    assert ret.size() == (bsz, tlen, 8)

    # Forward with self-attention-mask only.
    tgt_mask = torch.randint(0, 2, (tlen, tlen)).to(torch.bool)
    ret = td(
        tgt, mem,
        tgt_mask=tgt_mask,
        tgt_is_causal=False
    )
    assert ret.size() == (bsz, tlen, 8)

    # Forward with self-attention-mask and key-padding-mask.
    tgt_mask = torch.randint(0, 2, (tlen, tlen)).to(torch.bool)
    ret = td(
        tgt, mem,
        tgt_mask=tgt_mask,
        tgt_is_causal=False,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    assert ret.size() == (bsz, tlen, 8)

    # Forward with causal mask.
    nn_tgt_mask = nn.Transformer.generate_square_subsequent_mask(tlen)
    ret = td(tgt, mem, tgt_is_causal=True)
    ret2 = td(tgt, mem, tgt_is_causal=False, tgt_mask=nn_tgt_mask)
    assert torch.all(torch.isclose(ret2, ret, rtol=1e-3))


# %%
def test_Transformer():
    bsz, slen, tlen, mlen = 3, 4, 6, 4
    src = torch.randn(bsz, slen, 8, dtype=torch.float32)
    tgt = torch.randn(bsz, tlen, 8, dtype=torch.float32)
    mem = torch.randn(bsz, mlen, 8, dtype=torch.float32)

    nntf = nn.Transformer(8, 2, 6, 6, 16, 0.0, batch_first=True)
    tf = Transformer(8, 2, 16, 0.0, 6, 6)
    nn_sd = nntf.state_dict()
    enc_sd_mapping = {
        "self_attn.in_proj.weight"  : "self_attn.in_proj_weight",
        "self_attn.in_proj.bias"    : "self_attn.in_proj_bias",
        "self_attn.out_proj.weight" : "self_attn.out_proj.weight",
        "self_attn.out_proj.bias"   : "self_attn.out_proj.bias",
        "ffn_linear1.weight"        : "linear1.weight",
        "ffn_linear1.bias"          : "linear1.bias",
        "ffn_linear2.weight"        : "linear2.weight",
        "ffn_linear2.bias"          : "linear2.bias",
        "norm1.weight"              : "norm1.weight",
        "norm1.bias"                : "norm1.bias",
        "norm2.weight"              : "norm2.weight",
        "norm2.bias"                : "norm2.bias",
    }
    dec_sd_mapping = {
        "self_attn.in_proj.weight"  : "self_attn.in_proj_weight",
        "self_attn.in_proj.bias"    : "self_attn.in_proj_bias",
        "self_attn.out_proj.weight" : "self_attn.out_proj.weight",
        "self_attn.out_proj.bias"   : "self_attn.out_proj.bias",
        "cross_attn.in_proj.weight" : "multihead_attn.in_proj_weight",
        "cross_attn.in_proj.bias"   : "multihead_attn.in_proj_bias",
        "cross_attn.out_proj.weight": "multihead_attn.out_proj.weight",
        "cross_attn.out_proj.bias"  : "multihead_attn.out_proj.bias",
        "ffn_linear1.weight"        : "linear1.weight",
        "ffn_linear1.bias"          : "linear1.bias",
        "ffn_linear2.weight"        : "linear2.weight",
        "ffn_linear2.bias"          : "linear2.bias",
        "norm1.weight"              : "norm1.weight",
        "norm1.bias"                : "norm1.bias",
        "norm2.weight"              : "norm2.weight",
        "norm2.bias"                : "norm2.bias",
        "norm3.weight"              : "norm3.weight",
        "norm3.bias"                : "norm3.bias",
    }
    sd = {}
    for n in range(6):
        for ek, ev in enc_sd_mapping.items():
            sd[f"encoder.layers.{n}.{ek}"] = nn_sd[f"encoder.layers.{n}.{ev}"]
        for dk, dv in dec_sd_mapping.items():
            sd[f"decoder.layers.{n}.{dk}"] = nn_sd[f"decoder.layers.{n}.{dv}"]
    tf.load_state_dict(sd)

    # Default forward.
    nnret = nntf(src, tgt)
    ret = nntf(src, tgt)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-2))

    # Forward with key-padding-mask.
    src_key_padding_mask = torch.randint(0, 2, (bsz, slen)).to(torch.bool)
    tgt_key_padding_mask = torch.randint(0, 2, (bsz, tlen)).to(torch.bool)
    nnret = nntf(
        src, tgt,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask)
    ret = tf(
        src, tgt,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-2))

    # Forward with self-attention-mask and key-padding-mask.
    src_mask = torch.randint(0, 2, (slen, slen)).to(torch.bool)
    tgt_mask = torch.randint(0, 2, (tlen, tlen)).to(torch.bool)
    nnret = nntf(
        src, tgt,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask)
    ret = tf(
        src, tgt,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-2))

    # Forward with causal mask.
    nn_tgt_mask = nn.Transformer.generate_square_subsequent_mask(tlen)
    nnret = nntf(
        src, tgt,
        tgt_mask=nn_tgt_mask,
        tgt_is_causal=True,
    )
    ret = tf(src, tgt, tgt_is_causal=True)
    assert torch.all(torch.isclose(nnret, ret, rtol=1e-2))
