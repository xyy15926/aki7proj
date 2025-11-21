#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_attention.py
#   Author: xyy15926
#   Created: 2025-06-17 15:58:08
#   Updated: 2025-11-21 12:08:58
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
    ssoftmax,
    scaled_dot_product_attention,
    MultiheadAttention,
    SimpleMHA,
)
torch.autograd.set_detect_anomaly(False)


# %%
if fkwargs_32_dml:
    torch_fkwargs_params = [fkwargs_64_cpu, fkwargs_32_dml]
else:
    torch_fkwargs_params = [fkwargs_64_cpu, ]
@pytest.fixture(params=[fkwargs_64_cpu, fkwargs_32_dml])
def torch_fkwargs(request):
    return request.param
# torch_fkwargs = fkwargs_32_dml
# torch_fkwargs = fkwargs_64_cpu


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


# %%
@mark.filterwarnings("ignore: .*is not currently supported on the DML backend*")
def test_scaled_dot_product_attention(torch_fkwargs):
    dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]
    query = torch.randn(3, 4, 5, **torch_fkwargs)
    key = torch.randn(3, 6, 5, **torch_fkwargs)
    value = torch.randn(3, 6, 5, **torch_fkwargs)
    attn_mask = torch.randint(0, 2, (4, 6), device=device).to(torch.bool)
    attn_mask[0] = 0
    attn_mask_3D = torch.randint(0, 2, (3, 4, 6), device=device).to(torch.bool)

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
    F_attn_mask = (
        attn_mask + 1
        - torch.ones(4, 6, dtype=torch.int, device=device).tril(diagonal=0)
    )
    foutp = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=F_attn_mask.logical_not()
    )
    assert all_close(outp, foutp, 0, 1)


# %%
def test_scaled_dot_product_attention_backward_grad(torch_fkwargs):
    dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]
    query = torch.randn(3, 4, 2, requires_grad=True, **torch_fkwargs)
    key = torch.randn(3, 6, 2, requires_grad=True, **torch_fkwargs)
    value = torch.randn(3, 6, 2, requires_grad=True, **torch_fkwargs)
    sgd = optim.SGD((query, key, value))
    # attn_mask = torch.randint(0, 2, (4, 6)).to(torch.bool)
    # attn_mask[0, :] = False
    # attn_mask[:, 0] = False
    attn_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
    ]).to(dtype=torch.bool, device=device)

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
    query = torch.randn(3, 4, 2, requires_grad=True, **torch_fkwargs)
    key = torch.randn(3, 6, 2, requires_grad=True, **torch_fkwargs)
    value = torch.randn(3, 6, 2, requires_grad=True, **torch_fkwargs)
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
def test_MultiHeadAttention(torch_fkwargs):
    dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]
    query = torch.randn(3, 4, 8, **torch_fkwargs)
    key = torch.randn(3, 6, 8, **torch_fkwargs)
    value = torch.randn(3, 6, 8, **torch_fkwargs)

    nnmha = nn.MultiheadAttention(8, 2, batch_first=True, **torch_fkwargs)
    nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(8, 2, **torch_fkwargs)
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

    # Forward with `is_causal` only.
    # `is_causal` in `nn.MultiheadAttention` is just a hint and
    # `attn_mask`(`src_mask`) must be set if `is_causal` is set.
    causal_mask = MultiheadAttention.merge_masks(
        None, None, 1, query, key
    ).squeeze().to(query.dtype)
    nnattn, nnw = nnmha(query, key, value, attn_mask=causal_mask, is_causal=True)
    attn, attn_ws = mha(query, key, value, is_causal=True)
    assert all_close(nnattn, attn)

    # Forward with key-padding-mask.
    key_padding_mask = torch.randint(0, 2, (3, 6)).to(dtype=torch.bool, device=device)
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
    attn_mask = torch.randint(0, 2, (4, 6)).to(dtype=torch.bool, device=device)
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
    attn_mask = torch.randint(0, 2, (4, 6)).to(dtype=torch.bool, device=device)
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
def test_MultiHeadAttention_qkv_diffsz(torch_fkwargs):
    dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]
    query = torch.randn(3, 4, 8, **torch_fkwargs)
    key = torch.randn(3, 6, 16, **torch_fkwargs)
    value = torch.randn(3, 6, 16, **torch_fkwargs)

    nnmha = nn.MultiheadAttention(
        8, 1,
        bias=True,
        kdim=16,
        vdim=16,
        batch_first=True,
        **torch_fkwargs,
    )
    nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(8, 1, ksz=16, vsz=16, **torch_fkwargs)
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
def test_MHA_merge_masks(torch_fkwargs):
    dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]

    # 4D-QKV and mask will be used here so that
    # `F.scaled_dot_product_attention` won't raise RuntimeError.
    query = torch.randn(3, 1, 4, 2, requires_grad=True, **torch_fkwargs)
    key = torch.randn(3, 1, 6, 2, requires_grad=True, **torch_fkwargs)
    value = torch.randn(3, 1, 6, 2, requires_grad=True, **torch_fkwargs)

    key_padding_mask = torch.randint(0, 2, (3, 6), device=device).to(torch.bool)
    key_padding_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
    ], device=device).to(torch.bool).logical_not()
    attn_mask = torch.randint(0, 2, (4, 6), device=device).to(torch.bool)
    attn_mask = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1],
    ], device=device).to(torch.bool).logical_not()

    # Causality from `is_causal` or `attn_mask` lead to the same result.
    def check_SDPA_and_causal(query, key, value, non_causal_mask, causal_mask):
        cn_ret, c_ws = scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_mask,
            is_causal=False,
        )
        nc_ret, nc_ws = scaled_dot_product_attention(
            query, key, value,
            attn_mask=non_causal_mask,
            is_causal=True,
        )
        cc_ret, c_ws = scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_mask,
            is_causal=True,
        )
        cn_fret = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_mask,
            is_causal=False,
        )

        assert all_close(cn_ret, nc_ret)
        assert all_close(nc_ret, cc_ret)
        assert all_close(cc_ret, cn_fret, 0, 1)

        if device == torch.device("cpu"):
            nc_fret = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=non_causal_mask,
                is_causal=True,
            )
            cc_fret = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=causal_mask,
                is_causal=True,
            )
            assert all_close(cn_fret, nc_fret)
            assert all_close(nc_fret, cc_fret)

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
def test_MultiHeadAttention_is_causal(torch_fkwargs):
    dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]

    query = torch.randn(3, 4, 8, **torch_fkwargs)
    key = torch.randn(3, 6, 8, **torch_fkwargs)
    value = torch.randn(3, 6, 8, **torch_fkwargs)

    nnmha = nn.MultiheadAttention(8, 1, batch_first=True, **torch_fkwargs)
    nn_sd = nnmha.state_dict()
    mha = MultiheadAttention(8, 1, **torch_fkwargs)
    sd = {
        "in_proj.weight": nn_sd["in_proj_weight"],
        "in_proj.bias": nn_sd["in_proj_bias"],
        "out_proj.weight": nn_sd["out_proj.weight"],
        "out_proj.bias": nn_sd["out_proj.bias"],
    }
    mha.load_state_dict(sd)
    attn_mask = torch.randint(0, 2, (4, 6), device=device).to(torch.bool)
    key_padding_mask = torch.randint(0, 2, (3, 6), device=device).to(torch.bool)

    # Construct `attn_mask` for `F.scaled_dot_product_attention` manually.
    F_attn_mask = attn_mask.logical_or(
        torch.ones(4, 6, dtype=torch.int, device=device).tril(diagonal=0)
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
def test_SimpleMHA(torch_fkwargs):
    dtype, device = torch_fkwargs["dtype"], torch_fkwargs["device"]

    bsz, slen, tlen, mlen = 3, 4, 6, 5
    hn, esz = 1, 8
    query = torch.randn(bsz, slen, esz, **torch_fkwargs)
    key = torch.randn(bsz, mlen, esz, **torch_fkwargs)
    value = torch.randn(bsz, mlen, esz, **torch_fkwargs)

    mha = SimpleMHA(esz, hn, **torch_fkwargs)
    nnmha = nn.MultiheadAttention(esz, hn, batch_first=True, **torch_fkwargs)
    single_w = mha.state_dict()["attn_proj.weight"]
    nn_sd = nnmha.state_dict()
    # Attention: only slice on first demension is allowed, even single `:` for
    # the second demension is not allowed for the tensor assignment in GPU.
    nn_sd["in_proj_weight"][:8] = single_w
    nn_sd["in_proj_weight"][8:-8] = single_w
    torch.eye(8, out=nn_sd["in_proj_weight"][-8:], **torch_fkwargs)
    torch.eye(8, out=nn_sd["out_proj.weight"], **torch_fkwargs)
    nnmha.load_state_dict(nn_sd)

    nnattn, nnattn_ws = nnmha(query, key, value, need_weights=True)
    attn, attn_ws = mha(query, key, value)
    assert attn.size() == (bsz, slen, esz)
    assert attn_ws.size() == (bsz, slen, mlen)
    assert all_close(nnattn, attn, 1, 0)
    assert all_close(nnattn_ws, attn_ws, 1, 0)

    # Forward with key-padding-mask.
    key_padding_mask = torch.randint(0, 2, (bsz, mlen), device=device).to(torch.bool)
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
    assert all_close(nnattn, attn, 1, 0)
    assert all_close(nnattn_ws, attn_ws, 1, 0)

    # Forward with attention-mask only.
    attn_mask = torch.randint(0, 2, (slen, mlen), device=device).to(torch.bool)
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
    assert all_close(nnattn, attn, 1, 0)
    assert all_close(nnattn_ws, attn_ws, 1, 0)

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
    assert all_close(nnattn, attn, 1, 0)
    assert all_close(nnattn_ws, attn_ws, 1, 0)


# %%
def test_SimpleMHA_qkv_diffsz(torch_fkwargs):
    bsz, slen, tlen, mlen = 3, 4, 6, 5
    hn, qksz, vsz = 2, 8, 16
    query = torch.randn(bsz, slen, qksz, **torch_fkwargs)
    key = torch.randn(bsz, mlen, qksz, **torch_fkwargs)
    value = torch.randn(bsz, mlen, vsz, **torch_fkwargs)

    mha = SimpleMHA(8, 1, vsz=vsz, **torch_fkwargs)
    attn, attn_ws = mha(query, key, value)
    assert attn.size() == (bsz, slen, vsz)
    assert attn_ws.size() == (bsz, slen, mlen)

    mha = SimpleMHA(8, 1, vsz=vsz, out_proj=True, **torch_fkwargs)
    attn, attn_ws = mha(query, key, value)
    assert attn.size() == (bsz, slen, qksz)
    assert attn_ws.size() == (bsz, slen, mlen)
