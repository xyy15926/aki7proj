#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_attention.py
#   Author: xyy15926
#   Created: 2025-06-17 15:58:08
#   Updated: 2025-11-17 23:06:03
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark
import torch
from torch import nn, optim
from torch.nn import functional as F

if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import attention
    reload(attention)

from ubears.nutsbear.attention import (
    ssoftmax,
    scaled_dot_product_attention,
    MultiheadAttention,
    SimpleMHA,
)
torch.autograd.set_detect_anomaly(False)


# %%
def all_close(
    lt = torch.Tensor,
    rt = torch.Tensor,
    lnan_to_zero = False,
    rnan_to_zero = False,
    equal_nan = True,
    rtol = 1e-5,
    atol = 1e-3,
):
    if lnan_to_zero and torch.is_tensor(lt):
        lt = torch.nan_to_num(lt, 0.0)
    if rnan_to_zero and torch.is_tensor(rt):
        rt = torch.nan_to_num(rt, 0.0)
    if torch.is_tensor(rt):
        return torch.allclose(
            lt,
            rt,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan
        )
    else:
        return torch.allclose(
            lt,
            torch.tensor([rt]),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan
        )


# %%
def test_ssoftmax():
    # Check common result.
    ori = torch.rand(5, 3, 4)
    for dim in (0, 1, 2, -1):
        ret1 = F.softmax(ori, dim)
        ret2 = ssoftmax(ori, dim)
        assert torch.all(torch.isclose(ret1, ret2))

    # Check all-NInf input.
    addup = torch.tensor([[float("-inf"), 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
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
    ori = torch.rand(5, 3, 4, requires_grad=True)
    inp = ori + addup
    loss = F.softmax(inp, 0).sum()
    loss.backward()
    assert torch.all(torch.isnan(ori.grad[:, 0, 0]))

    ori = torch.rand(5, 3, 4, requires_grad=True)
    inp = ori + addup
    loss = ssoftmax(inp, 0).sum()
    loss.backward()
    assert all_close(ret2[:, 0, 0], 0.0)


# %%
def test_scaled_dot_product_attention():
    query = torch.randn(3, 4, 5, dtype=torch.float64)
    key = torch.randn(3, 6, 5, dtype=torch.float64)
    value = torch.randn(3, 6, 5, dtype=torch.float64)
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    attn_mask[0] = 0
    attn_mask_3D = torch.randint(0, 2, (3, 4, 6)).to(torch.bool)

    # Attention mask SDPA.
    outp, ws = scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask.logical_not()
    )
    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask
    )
    assert all_close(outp, foutp, 0, 1)

    outp, ws = scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask.logical_not(),
        safe_softmax=False,
    )
    assert all_close(outp, foutp, 1, 1)

    # 3D-Attention mask SDPA.
    outp, ws = scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask_3D.logical_not()
    )
    foutp = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask_3D
    )
    assert all_close(outp, foutp, 0, 1)

    outp, ws = scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask_3D.logical_not(),
        safe_softmax=False,
    )
    assert all_close(outp, foutp, 1, 1)

    # Causal attention mask SDPA.
    outp, ws = scaled_dot_product_attention(query, key, value, is_causal=True)
    foutp = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    assert all_close(outp, foutp, 0, 1)

    outp, ws = scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        is_causal=True
    )
    # Construct `attn_mask` for `F.scaled_dot_product_attention` manually.
    F_attn_mask = attn_mask + 1 - torch.ones(4, 6, dtype=torch.int).tril(diagonal=0)
    foutp = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=F_attn_mask.logical_not()
    )
    assert all_close(outp, foutp, 0, 1)


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

    # `NaN` will be return for all-NInf query and lead to NaN in `.grad`
    # in backward.
    output, ws = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask.logical_not(),
        safe_softmax=False,
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

    # So does the `scaled_dot_product_attention` with safe-softmax.
    query = torch.randn(3, 4, 2, dtype=torch.float64, requires_grad=True)
    key = torch.randn(3, 6, 2, dtype=torch.float64, requires_grad=True)
    value = torch.randn(3, 6, 2, dtype=torch.float64, requires_grad=True)
    sgd = optim.SGD((query, key, value))

    outp, ws = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask.logical_not(),
        safe_softmax=True
    )
    os = outp.sum()
    os.backward()
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
def test_MultiHeadAttention():
    query = torch.randn(3, 4, 8, dtype=torch.float32)
    key = torch.randn(3, 6, 8, dtype=torch.float32)
    value = torch.randn(3, 6, 8, dtype=torch.float32)

    nnmha = nn.MultiheadAttention(8, 2, batch_first=True)
    nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(8, 2)
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
    assert all_close(nnattn, attn)

    # Forward with key-padding-mask.
    key_padding_mask = torch.randint(0, 2, (3, 6)).to(torch.bool)
    key_padding_mask[0, :] = True
    nnattn, nnw = nnmha(query, key, value, key_padding_mask=key_padding_mask)
    attn, attn_ws = mha(query, key, value, key_padding_mask=key_padding_mask)
    assert torch.all(torch.isnan(nnattn[0]))
    assert not torch.any(torch.isnan(attn))
    assert all_close(nnattn, attn, 1, 0)

    nnattn, nnw = nnmha(
        query, key, value,
        key_padding_mask=key_padding_mask.logical_not(),
        need_weights=False,
    )
    attn, attn_ws = mha(
        query, key, value,
        key_padding_mask=key_padding_mask.logical_not()
    )
    assert all_close(nnattn, attn, 1, 0)

    # Forward with attention-mask only.
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    attn_mask[0, :] = True
    nnattn, nnattn_w = nnmha(
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
    assert torch.all(torch.isnan(nnattn_w[:, 0]))
    assert not torch.any(torch.isnan(attn_ws_w))
    assert all_close(nnattn_w, attn_ws_w, 1, 0)
    assert all_close(nnattn, attn, 1, 1)
    assert all_close(attn, attn_w, 1, 0)

    # Forward with attention-mask(or mixed mask).
    attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )
    attn, attn_ws = mha(
        query, key, value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
    )
    assert all_close(nnattn, attn, 1, 0)

    nnattn, nnw = nnmha(
        query, key, value,
        attn_mask=attn_mask.logical_not(),
        key_padding_mask=key_padding_mask.logical_not(),
        need_weights=False,
    )
    attn, attn_ws = mha(
        query, key, value,
        attn_mask=attn_mask.logical_not(),
        key_padding_mask=key_padding_mask.logical_not()
    )
    assert all_close(nnattn, attn, 1, 0)


# %%
def test_MultiHeadAttention_qkv_diffsz():
    query = torch.randn(3, 4, 8, dtype=torch.float32)
    key = torch.randn(3, 6, 16, dtype=torch.float32)
    value = torch.randn(3, 6, 16, dtype=torch.float32)

    nnmha = nn.MultiheadAttention(
        8, 1,
        bias=True,
        kdim=16,
        vdim=16,
        batch_first=True
    )
    nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(
        8, 1,
        ksz=16,
        vsz=16,
    )
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
    assert all_close(nnattn, attn, 1, 0)


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

        assert all_close(nc_ret, cn_ret)
        assert all_close(nc_ret, cc_ret)
        assert all_close(nc_fret, cn_fret)
        assert all_close(nc_fret, cc_fret)
        assert all_close(nc_ret, nc_fret, 0, 1)

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
def test_MultiHeadAttention_is_causal():
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
        key_padding_mask=key_padding_mask.logical_not(),
        need_weights=False,
    )
    # `is_causal` and `attn_mask` are merged.
    attn, attn_ws = mha(
        query, key, value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask.logical_not(),
        is_causal=True
    )
    assert all_close(nnattn, attn, 1, 0)


# %%
def test_SimpleMHA():
    bsz, slen, tlen, mlen = 3, 4, 6, 5
    hn, esz = 1, 8
    query = torch.randn(bsz, slen, esz, dtype=torch.float32)
    key = torch.randn(bsz, mlen, esz, dtype=torch.float32)
    value = torch.randn(bsz, mlen, esz, dtype=torch.float32)

    mha = SimpleMHA(esz, hn)
    nnmha = nn.MultiheadAttention(esz, hn, batch_first=True)
    single_w = mha.state_dict()["attn_proj.weight"]
    nn_sd = nnmha.state_dict()
    nn_sd["in_proj_weight"][:8, :] = single_w
    nn_sd["in_proj_weight"][8:-8, :] = single_w
    torch.eye(8, out=nn_sd["in_proj_weight"][-8:, :])
    torch.eye(8, out=nn_sd["out_proj.weight"])
    nnmha.load_state_dict(nn_sd)

    nnattn, nnattn_ws = nnmha(query, key, value, need_weights=True)
    attn, attn_ws = mha(query, key, value)
    assert attn.size() == (bsz, slen, esz)
    assert attn_ws.size() == (bsz, slen, mlen)
    assert torch.all(torch.isclose(nnattn, attn, rtol=1e-3, equal_nan=True))
    assert torch.all(torch.isclose(nnattn_ws, attn_ws, rtol=1e-3, equal_nan=True))

    # Forward with key-padding-mask.
    key_padding_mask = torch.randint(0, 2, (bsz, mlen)).to(torch.bool)
    nnattn, nnattn_ws = nnmha(
        query, key, value,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    attn, attn_ws = mha(
        query, key, value,
        key_padding_mask=key_padding_mask
    )
    assert attn.size() == (bsz, slen, esz)
    assert attn_ws.size() == (bsz, slen, mlen)
    assert torch.all(torch.isclose(
        torch.nan_to_num(nnattn, 0.0),
        attn,
        rtol=1e-3,
        equal_nan=True
    ))
    assert torch.all(torch.isclose(
        torch.nan_to_num(nnattn_ws, 0.0),
        attn_ws,
        rtol=1e-3,
        equal_nan=True
    ))

    # Forward with attention-mask only.
    attn_mask = torch.randint(0, 2, (slen, mlen)).to(torch.bool)
    nnattn, nnattn_ws = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        need_weights=True,
    )
    attn, attn_ws = mha(
        query, key, value,
        attn_mask=attn_mask,
    )
    assert attn.size() == (bsz, slen, esz)
    assert attn_ws.size() == (bsz, slen, mlen)
    assert torch.all(torch.isclose(
        torch.nan_to_num(nnattn, 0.0),
        attn,
        rtol=1e-3,
        equal_nan=True
    ))
    assert torch.all(torch.isclose(
        torch.nan_to_num(nnattn_ws, 0.0),
        attn_ws,
        rtol=1e-3,
        equal_nan=True
    ))

    # Forward with attention-mask(or mixed mask).
    nnattn, nnattn_ws = nnmha(
        query, key, value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    attn, attn_ws = mha(
        query, key, value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask
    )
    assert attn.size() == (bsz, slen, esz)
    assert attn_ws.size() == (bsz, slen, mlen)
    assert torch.all(torch.isclose(
        torch.nan_to_num(nnattn, 0.0),
        attn,
        rtol=1e-3,
        equal_nan=True
    ))
    assert torch.all(torch.isclose(
        torch.nan_to_num(nnattn_ws, 0.0),
        attn_ws,
        rtol=1e-3,
        equal_nan=True
    ))


# %%
def test_SimpleMHA_qkv_diffsz():
    bsz, slen, tlen, mlen = 3, 4, 6, 5
    hn, qksz, vsz = 2, 8, 16
    query = torch.randn(bsz, slen, qksz, dtype=torch.float32)
    key = torch.randn(bsz, mlen, qksz, dtype=torch.float32)
    value = torch.randn(bsz, mlen, vsz, dtype=torch.float32)

    mha = SimpleMHA(8, 1, vsz=vsz)
    attn, attn_ws = mha(query, key, value)
    assert attn.size() == (bsz, slen, vsz)
    assert attn_ws.size() == (bsz, slen, mlen)

    mha = SimpleMHA(8, 1, vsz=vsz, out_proj=True)
    attn, attn_ws = mha(query, key, value)
    assert attn.size() == (bsz, slen, qksz)
    assert attn_ws.size() == (bsz, slen, mlen)
