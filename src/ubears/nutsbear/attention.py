#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: attention.py
#   Author: xyy15926
#   Created: 2025-06-17 12:01:06
#   Updated: 2025-07-05 20:18:30
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple
import logging

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from IPython.core.debugger import set_trace

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot product attention.

    1. Doesn't support NestedTensor.
    2. True represents to mask the correspondant position, which is
      controversary to `F.scaled_dot_product_attention`, so to keep up with
      `nn.MultiheadAttention` behavior.
    3. NaN will be filled with 0, for masked positions mostly, so to keep up
      with `F.scaled_dot_product_attention`.

    Ref:
    ---------------------------
    - F.scaled_dot_product_attention:
      - https://pytorch.ac.cn/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    - Transformer Code Analysis:
      - https://ifwind.github.io/2021/08/19/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%8810%EF%BC%89Transformer%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90

    Params:
    --------------------------
    query: Query input Tensor
    key: Key input Tensor.
    value: Value input Tensor.
    attn_mask: Mask for relations between query and key sequences.
    is_causal: If to enforce causality, namely tokens can only attend to
      previous tokens.

    Shape:
    --------------------------
    query: (..., q_seq_len, qksz)
    key: (..., kv_seq_len, qksz)
    value: (..., kv_seq_len, vsz)
    attn_mask: (..., q_seq_len, kv_seq_len)
    ...(mostly): (bsz, heads_n)

    Return Shape:
    --------------------------
    attn_weight: (..., q_seq_len, kv_seq_len)
    output: (..., seq_len, vsz)

    Return:
    --------------------------
    attn_weight: Weights for values.
    output: Scaled-dot-product result.
    """
    *_____, qslen, qksz = query.size()
    *_____, kvslen, vsz = value.size()

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
    # set_trace()
    if attn_mask is not None:
        if attn_mask.dim() == 3:
            attn_bias.unsqueeze_(0)
        # Convert bool mask to float mask with ninf for softmax.
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    # Fit mask for query and key.
    if attn_bias.dim() == 3:
        for _____ in range(query.dim() - 3):
            attn_bias.unsqueeze_(1)

    # (bsz,..., qslen, qksz) * (bsz,..., qksz, kvslen)
    # => (bsz,..., qslen, kvslen)
    scaled_dot = (query / np.sqrt(qksz)) @ key.transpose(-2, -1)
    scaled_dot = scaled_dot + attn_bias
    attn_weight = F.softmax(scaled_dot, dim=-1)
    if dropout_p:
        attn_weight = F.dropout(scaled_dot, p=dropout_p)

    # (bsz, qslen, kvslen) * (bsz, kvslen, vsz)
    # => (bsz, qslen, vsz)
    output = attn_weight @ value

    # Fill `nan` with 0 to keep with `F.SDPA` for straight-masked-line
    #  in `attn_mask`.
    torch.nan_to_num_(output, 0.0)
    return output, attn_weight


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
    -  Accelerating transformer with NestedTensor:
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
        """MultiheadAttention initialization.

        Params:
        ----------------------------
        qsz: The (embedding)size of the query.
        heads_n: The number the heads.
        ksz: The (embedding)size of the key.
        vsz: The (embedding)size of the value.
        tsz: The sum of the (hidden)sizes of output of the multi-heads.
        dropout_p: The probability of the dropout.
        bias: If to use bias in the projection for Q, K, V.
        device:
        dtype:

        Return:
        ----------------------------
        None
        """
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
        # Pack inner projection up.
        if self._qkv_same_embed_dim:
            self.in_proj = nn.Linear(
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
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Multi-head attention forward step.

        1. Packed linear projection and chunk will be performed instead of
          3 linear projection for query, key and value input seperately if
          query, key and value are the exact one tensor.
          (NestedTensor excluded.)
        2. NaN will be filled with 0, for masked positions mostly so to keep up
          with `F.scaled_dot_product_attention`, while the
          `nn.MultiheadAttention` will keep NaN untouched.

        Ref:
        --------------------------
        - torch.nn.functional.multi_head_attention_forward
          - https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L6083
          - https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L6492
        - torch.nested._internal.sdpa.py:
          - https://github.com/pytorch/pytorch/blob/main/torch/nested/_internal/sdpa.py
          - `nested._internal.sdpa._can_use_flash_spda_jagged`

        Params:
        --------------------------
        query: Query input Tensor
        key: Key input Tensor.
        value: Value input Tensor.
        key_padding_mask: Padding mask for key sequence.
        attn_mask: Mask for the attending relations between query and key.
        is_causal: If to enforce causality, namely tokens can only attend to
          previous tokens.

        Shape:
        --------------------------
        query: (bsz, query_seq_len, qsz)
        key: (bsz, kv_seq_len, ksz)
        value: (bsz, kv_seq_len, vsz)
        key_padding_mask: (bsz, kv_seq_len)
        attn_mask: (query_seq_len, kv_seq_len)
        RETURN: (bsz, query_seq_len, qsz)

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
                result = self.in_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                qw, kw, vw = torch.chunk(self.in_proj.weight, 3, dim=0)
                if self.bias:
                    qb, kb, vb = torch.chunk(self.in_proj.bias, 3, dim=0)
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

        # 3. SDPA.
        dropout_p = self.dropout_p if self.training else 0.0
        # `attn_mask` could be passed to `scaled_dot_product_attention` directly.
        if key_padding_mask is not None:
            bias_mask = self.merge_masks(key_padding_mask, attn_mask)
        else:
            bias_mask = attn_mask
        # set_trace()
        attn_val, attn_ws = scaled_dot_product_attention(
            query, key, value,
            attn_mask=bias_mask,
            dropout_p=dropout_p,
            is_causal=is_causal)
        # (bsz, heads_n, seq_len, hsz)
        # => (bsz, seq_len, heads_n, hsz)
        # => (bsz, seq_len, heads_n * hsz)
        attn_val = attn_val.transpose(1, 2).flatten(-2)

        return self.out_proj(attn_val), attn_ws

    @classmethod
    def merge_masks(
        cls,
        key_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Combine padding mask and attenion mask.

        1. `query_padding_mask` isn't added, which is the same with the
          `nn.MultiheadAttention`, since the query's mask will be handled by
          users lately somehow.
        2. True represents to mask the correspondant position, which is
          controversary to `F.scaled_dot_product_attention`, so to keep up with
          `nn.MultiheadAttention` behavior.

        Ref:
        --------------------------
        - torch.nn.modules.activation.MultiheadAttention.merge_masks:
          - https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py#L1406

        Params:
        --------------------------
        key_padding_mask: Mask for padding in key sequence.
        attn_mask: Mask to mark the position that shouldn't be attented.

        Shape:
        --------------------------
        key_padding_mask: (batch_size, key_seq_len)
        attn_mask: (query_seq_len, key_seq_len)

        Return:
        --------------------------
        Tensor[float]
        """
        bsz, kslen = key_padding_mask.size()
        if attn_mask is not None:
            qslen, kslen_ = attn_mask.size()
            assert kslen == kslen_, (
                "Key padding mask must be compatiable with attending mask.")
        else:
            qslen = 1

        bias_mask = torch.zeros(bsz, qslen, kslen)

        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                bias_mask.masked_fill_(
                    key_padding_mask.view(bsz, 1, kslen),
                    float("-inf"))
            else:
                bias_mask += key_padding_mask.view(bsz, 1, kslen)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                bias_mask.masked_fill_(
                    attn_mask.view(1, qslen, kslen),
                    float("-inf"))
            else:
                bias_mask += attn_mask.view(1, qslen, kslen)

        return bias_mask


# %%
class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer.

    Attrs:
    --------------------------
    self_attn: MHA module.
    sa_dropout: Dropout module after MHA.
    ffn_linear1: FFN layer 1.
    ffn_linear2: FFN layer 2.
    ffn_activation: Activation after FFN layer1.
    ffn_dropout1: Dropout module after FFN layer1.
    ffn_dropout2: Dropout module after FFN layer2.
    norm1: Layer norm module after MHA.
    norm2: Layer norm module after FFN.

    Ref:
    --------------------------
    - PyTorch Transformer:
      - https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/transformer.py
    """
    def __init__(
        self,
        embed_sz: int,
        heads_n: int,
        ffn_sz: int,
        dropout_p: float = 0.0,
    ):
        """Encoder Layer Initialization.

        1. The size of query, key, value and embeding in MHA will be the same.
        2. Two linear layers will perform `embed_sz * 1 * 1` 1D-Conv.

        Params:
        --------------------------
        embed_sz: The size of embedding for the MHA.
        heads_n: The number of heads in the MHA.
        ffn_sz: The size of hidden layer in the FFN.
        dropout_p: The probability of the dropout.

        Return:
        --------------------------
        None
        """
        super().__init__()
        self.self_attn = MultiheadAttention(embed_sz, heads_n, dropout_p=dropout_p)
        self.sa_dropout = nn.Dropout(dropout_p)

        self.ffn_linear1 = nn.Linear(embed_sz, ffn_sz, bias=True)
        self.ffn_linear2 = nn.Linear(ffn_sz, embed_sz, bias=True)
        self.ffn_activation = nn.ReLU()
        self.ffn_dropout1 = nn.Dropout(dropout_p)
        self.ffn_dropout2 = nn.Dropout(dropout_p)

        self.norm1 = nn.LayerNorm(embed_sz)
        self.norm2 = nn.LayerNorm(embed_sz)

    def _sa_block(
        self,
        inp: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Self-Attention block."""
        attn_val, attn_ws = self.self_attn(
            inp, inp, inp,
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        return self.sa_dropout(attn_val)

    def _ffn_block(
        self,
        inp: torch.Tensor,
    ) -> torch.Tensor:
        """Feed-Forward Network block."""
        outp = self.ffn_dropout1(self.ffn_activation(self.ffn_linear1(inp)))
        outp = self.ffn_dropout2(self.ffn_linear2(outp))
        return outp

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Network forward.

        Apply Add & Normlization after Self-Attention and FeedForward Network.
        """
        src = self.norm1(
            src
            + self._sa_block(
                src,
                src_key_padding_mask=src_key_padding_mask,
                attn_mask=attn_mask,
                is_causal=is_causal
            )
        )
        src = self.norm2(src + self._ffn_block(src))
        return src


# %%
class TransformerEncoder(nn.Module):
    """
    """
    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        layers_n: int = 6,
        dropout_p: float = 0.0,
    ):
        """Init Transformer Encoder.

        Params:
        --------------------------
        embed_sz: Size of the embeding, query, key and value.
        heads_n: Number of the multi-heads.
        ffn_sz: The size of the hidden dimension of the FFN.
        dropout_p: Dropout probability.
        is_causal: If the attention should be causal.
        """
        super().__init__()
        # `copy.deepcopy` may be more time-efficient than sequent init.
        self.enc_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(layers_n)])
        self.layers_n = layers_n

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Transformer Encoder part forward."""
        outp = src
        # Merge the `key_padding_mask` and `attn_mask` only once.
        if src_key_padding_mask is not None:
            bias_mask = MultiheadAttention.merge_masks(src_key_padding_mask, attn_mask)
        else:
            bias_mask = attn_mask
        for mod in self.enc_layers:
            outp = mod(outp, attn_mask=bias_mask, is_causal=is_causal)
        return outp
