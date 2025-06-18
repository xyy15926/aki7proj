#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: attention.py
#   Author: xyy15926
#   Created: 2025-06-17 12:01:06
#   Updated: 2025-06-18 15:26:39
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class MultiheadAttention(nn.Module):
    """Multi-head attention.

    1. NestedTensor compatiable.
    2. Inner projection output-size for query, key and value are presumed to
      the same, though the output-size of query/key could be different from
      the value.
    3. Inner projection will be packed together if the size of inputs of
      query, key and value are the same.

    Ref:
    --------------------------
    Accelerating transformer with NestedTensor:
    - https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    - https://pytorch.ac.cn/tutorials/intermediate/transformer_building_blocks.html

    Attrs:
    --------------------------
    heads_n: Int.
      The number of the multi-heads.
    hsz: Int.
      The size of one attention head.
    dropout_p: Float.
      The probability of the dropout.
    bias: Bool.
      If to enable bias in projection.
    q_proj: nn.Linear
      Linear projection for query input.
    k_proj: nn.Linear
      Linear projection for key input.
    v_proj: nn.Linear
      Linear projection for value input.
    out_proj: nn.Linear
      Linear projection for concated attention ouptut of multi-heads.
    """
    def __init__(
        self,
        qsz: int,
        heads_n: int,
        ksz: int = None,
        vsz: int = None,
        tsz: int = None,
        dropout_p: float = 0.0,
        bias: bool = True,
        device: str = None,
        dtype: str = None,
    ):
        ksz = qsz if ksz is None else ksz
        vsz = qsz if vsz is None else vsz
        tsz = qsz if tsz is None else tsz
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_n = heads_n
        assert tsz % heads_n == 0, "Embedding dim is not divisible by nheads."
        self.hsz = tsz // heads_n
        self.dropout_p = dropout_p
        self._qkv_same_embed_dim = qsz == ksz and ksz == vsz
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(
                qsz, tsz * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(qsz, tsz, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(ksz, tsz, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(vsz, tsz, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(tsz, qsz, bias=bias, **factory_kwargs)
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Multi-head attention forward step.

        1. Packed linear projection and chunk will be performed instead of
          3 linear projection for query, key and value input seperately if
          query, key and value are the exact one tensor.
          (NestedTensor excluded.)

        Ref:
        --------------------------
        torch.nested._internal.sdpa.py:
        - https://github.com/pytorch/pytorch/blob/main/torch/nested/_internal/sdpa.py
        - `nested._internal.sdpa._can_use_flash_spda_jagged`

        Params:
        --------------------------
        query: Query input Tensor
        key: Key input Tensor.
        value: Value input Tensor.
        attn_mask:
        is_causal:

        Shape:
        --------------------------
        query: (bsz, seq_len, qsz)
        key: (bsz, seq_len, ksz)
        value: (bsz, seq_len, vsz)
        attn_mask: (seq_len, seq_len)
        RETURN: (bsz, seq_len, qsz)

        Return:
        --------------------------
        Attention outpout.
        """
        # 1. Apply input projection.
        if self._qkv_same_embed_dim:
            # `NJT` will be uncontiguous after chunk and the exact C-style
            # contiguous isn't compatiable with `F.SPDA`, namely simple
            # `NJT.contiguous` doesn't work.
            if query is key and key is value and not query.is_nested:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                qw, kw, vw = torch.chunk(self.packed_proj.weight, 3, dim=0)
                if self.bias:
                    qb, kb, vb = torch.chunk(self.packed_proj.bias, 3, dim=0)
                else:
                    qb, kb, vb = None, None, None
                query = F.linear(query, qw, qb)
                key = F.linear(key, kw, kb)
                value = F.linear(value, vw, vb)
        else:
            query = self.q_proj(query)
            key = self.q_proj(key)
            value = self.q_proj(value)

        # 2. Split heads for SDPA.
        # (bsz, seq_len, heads_n * hsz)
        # => (bsz, seq_len, heads_n, hsz)
        # => (bsz, heads_n, seq_len, hsz)
        query = query.unflatten(-1, [self.heads_n, self.hsz]).transpose(1, 2)
        key = key.unflatten(-1, [self.heads_n, self.hsz]).transpose(1, 2)
        value = value.unflatten(-1, [self.heads_n, self.hsz]).transpose(1, 2)

        # 3. SPDA.
        dropout_p = self.dropout_p if self.training else 0.0
        attn_val = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal)
        # (bsz, heads_n, seq_len, hsz)
        # => (bsz, seq_len, heads_n, hsz)
        # => (bsz, seq_len, heads_n * hsz)
        attn_val = attn_val.transpose(1, 2).flatten(-2)

        return self.out_proj(attn_val)


# %%
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor:
    """

    Ref:
    ---------------------------
    F.scaled_dot_product_attention:
    - https://pytorch.ac.cn/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    Transformer Code Analysis:
    - https://ifwind.github.io/2021/08/19/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%8810%EF%BC%89Transformer%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90

    Params:
    --------------------------
    query: Query input Tensor
    key: Key input Tensor.
    value: Value input Tensor.
    attn_mask: Padding mask with **False** representing as the masked.
    is_causal: If to enforce causality, namely tokens can only attend to
      previous tokens.

    Shape:
    --------------------------
    query: (..., q_seq_len, qksz)
    key: (..., kv_seq_len, qksz)
    value: (..., kv_seq_len, vsz)
    attn_mask: (..., q_seq_len, kv_seq_len)
    RETURN: (..., seq_len, vsz)
    ...(mostly): (bsz, heads_n)

    Return:
    --------------------------
    Scaled-dot-product.
    """
    *____, qslen, qksz = query.size()
    *____, kvslen, vsz = value.size()

    # Perform mask.
    attn_bias = torch.zeros(
        qslen, kvslen, dtype=query.dtype, device=query.device)
    # Causal mask.
    if is_causal:
        assert attn_mask is None, "Causal mask shouldn't be with padding mask."
        tmp_mask = torch.ones(qslen, kvslen, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(tmp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    # Padding mask.
    if attn_mask is not None:
        # Convert bool mask to float mask with ninf for softmax.
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    # (bsz, qslen, qksz) * (bsz, qksz, kvslen)
    # => (bsz, qslen, kvslen)
    scaled_dot = (query / np.sqrt(qksz)) @ key.transpose(-2, -1)
    scaled_dot += attn_bias
    attn_weight = F.softmax(scaled_dot, dim=-1)
    if dropout_p:
        attn_weight = F.dropout(scaled_dot, p=dropout_p)

    # (bsz, qslen, kvslen) * (bsz, kvslen, vsz)
    # => (bsz, qslen, vsz)
    output = attn_weight @ value

    # Fill `nan` with 0 to keep with `F.SDPA` for straight-masked-line
    #  in `attn_mask`.
    torch.nan_to_num_(output, 0.0)
    return output
