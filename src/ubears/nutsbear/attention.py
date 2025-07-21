#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: attention.py
#   Author: xyy15926
#   Created: 2025-06-17 12:01:06
#   Updated: 2025-07-19 17:05:07
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
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
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
class RotaryPE(nn.Module):
    """Rotary Position Embedding.

    Ref:
    -------------------------------
    - RePE: https://www.zhihu.com/tardis/bd/art/647109286

    Attrs:
    -------------------------------
    embed_sz: Embeding size.
      Only tensor with same embedding size(last dim size) should be encoded.
    cur_len: Rotary cache length.
    cos_cache: Cosine part of the rotary cache.
    sin_cache: Sine part of the rotary cache.

    Shape:
    -------------------------------
    cos_cache: [cur_len, embed_sz]
    sin_cache: [cur_len, embed_sz]
    """

    def __init__(
        self,
        embed_sz: int,
        bot: float = 1e4,
        device: str = None,
        dtype: str = None,
    ):
        """Init Module.
        """
        super().__init__()
        self.embed_sz = embed_sz
        self._bot = bot
        self.cur_len = 0
        self.cos_cache = torch.tensor([])
        self.sin_cache = torch.tensor([])

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None):
        """
        Params:
        -------------------------------
        x: Tensor be updated with position encoding.
        position: Position of the input tensor.
          0-slen will be used as default.

        Shape:
        -------------------------------
        x: [..., seq_len, embed_sz]
        position: [..., seq_len]
        RETURN: [..., seq_len, embed_sz]

        Return:
        -------------------------------
        Tensor after rotary embedding.
        """
        factory_kwargs = {"device": x.device, "dtype": x.dtype}
        *_____, slen, sz = x.size()
        assert sz == self.embed_sz, "Embedding size must be the same."
        if pos is not None:
            assert pos.size(-1) == slen, "All positions must be provided."
            max_pos = torch.max(pos)
            if max_pos >= self.cur_len:
                self.rotary_cache(max_pos + 1, **factory_kwargs)
            cos_ = self.cos_cache[pos]
            sin_ = self.sin_cache[pos]
        else:
            if slen >= self.cur_len:
                self.rotary_cache(slen, **factory_kwargs)
            cos_ = self.cos_cache[:slen]
            sin_ = self.sin_cache[:slen]

        cross_x = x.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
        ret = x * cos_ + cross_x * sin_

        return ret

    def rotary_cache(
        self,
        max_pos: int,
        device: str = None,
        dtype: str = None,
    ):
        """Calculate rotary cache.

        Params:
        -----------------------------------
        max_pos: The rotary cache to be calculated.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        # Set the rotary parameter.
        esz = self.embed_sz
        theta = -torch.arange(0, esz, 2, **factory_kwargs) / esz
        theta = (self._bot ** theta).unsqueeze(1).expand(-1, 2).flatten(-2)
        mtheta = torch.arange(max_pos).unsqueeze(1) * theta.unsqueeze(0)
        self.cos_cache = torch.cos(mtheta)
        sin_cache = torch.sin(mtheta)
        sin_cache[:, ::2] *= -1
        self.sin_cache = sin_cache


# %%
# TODO: `F.softmax` will return NaN for all NInf and lead to NaN in `.grad`.
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
    **** Different from `F.scaled_dot_product_attention`:
    2. True represents to mask the correspondant position, which is
      controversary to `F.scaled_dot_product_attention`, so to keep up with
      `nn.MultiheadAttention` behavior.
    3. `is_causal` and `attn_mask` could be set together, with warning log as
      reminder, and merge will be done.
    4. NaN won't filled with 0 after the softmax.
    **** The same with `F.scaled_dot_product_attention`:
    4. NaN will be filled with 0, for masked positions mostly, so to keep up
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
    attn_mask: 2D or 3D mask to prevent attention to certain position between
      query and key sequences.
      2D-mask: Mask that is same for all batches, which may be the mask
        representint the relations between the query and the key, namely
        "attention-mask".
      3D-mask: Mask that fit for batches seperately, which should the
        merged mask of "attention-mask" and "padding-mask",
        or just "padding-mask" if the `.size(1) == 1`.
    is_causal: If to enforce causality, namely tokens can only attend to
      previous tokens.

    Shape:
    --------------------------
    query: (..., q_seq_len, qksz)
    key: (..., kv_seq_len, qksz)
    value: (..., kv_seq_len, vsz)
    attn_mask: ([bsz,] q_seq_len, kv_seq_len)
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

    # Init mask with ninf.
    if (attn_mask is not None
            and attn_mask.dtype != torch.bool
            and not is_causal):
        bias_mask = attn_mask
    else:
        bias_mask = torch.zeros(
            qslen, kvslen,
            dtype=query.dtype,
            device=query.device
        )
        # Causal mask.
        if is_causal:
            if attn_mask is not None:
                logger.warning(
                    "Explicit attn_mask and is_causal are be set simultaneously."
                )
            tril_mask = torch.ones(qslen, kvslen, dtype=torch.bool).tril(diagonal=0)
            bias_mask.masked_fill_(tril_mask.logical_not(), float("-inf"))
            bias_mask.to(query.dtype)
        # Padding mask or attention mask.
        # set_trace()
        if attn_mask is not None:
            # Broadcast to fit for 3D merged or padding mask.
            if attn_mask.dim() == 3:
                tgt_shape = torch.broadcast_shapes(
                    (1, *(bias_mask.size())),
                    attn_mask.size(),
                )
                bias_mask = torch.broadcast_to(bias_mask, tgt_shape).clone()
            # Convert bool mask to float mask with ninf for softmax.
            if attn_mask.dtype == torch.bool:
                bias_mask.masked_fill_(attn_mask, float("-inf"))
            else:
                bias_mask = attn_mask + bias_mask

    # Fit mask for query and key.
    if bias_mask.dim() == 3:
        for _____ in range(query.dim() - 3):
            bias_mask.unsqueeze_(1)

    # (bsz,..., qslen, qksz) * (bsz,..., qksz, kvslen)
    # => (bsz,..., qslen, kvslen)
    q_scaled = query * np.sqrt(1.0 / qksz)
    scaled_dot = q_scaled @ key.transpose(-2, -1)
    scaled_dot += bias_mask
    attn_weight = F.softmax(scaled_dot, dim=-1)
    # attn_weight = torch.nan_to_num(attn_weight, 0.0)
    if dropout_p:
        attn_weight = F.dropout(scaled_dot, p=dropout_p)

    # (bsz, qslen, kvslen) * (bsz, kvslen, vsz)
    # => (bsz, qslen, vsz)
    output = attn_weight @ value

    # Fill `nan` with 0 to keep with `F.SDPA` for straight-masked-line
    #  in `attn_mask`.
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

        # Init parameters.
        self._reset_parameter()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> torch.Tensor:
        """Multi-head attention forward step.

        1. Packed linear projection and chunk will be performed instead of
          3 linear projection for query, key and value input seperately if
          query, key and value are the exact one tensor.
          (NestedTensor excluded.)
        **** Different from `nn.MultiheadAttention`:
        2. NaN will be filled with 0, for masked positions mostly so to keep up
          with `F.scaled_dot_product_attention`,
          while the `nn.MultiheadAttention` will keep NaN untouched.
        3. `is_causal` and `attn_mask` are independent here and merge will be
          done if both provided,
          while `is_causal` in `nn.MultiheadAttention` is just a hint for 
          accelaration and doesn't make any differences in
          `nn.MultiheadAttention`, or `F.multi_head_attention_foward`
          preciesly, if `attn_mask` is set in same cases, `need_weights` set
          for example.

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
        need_weights: If return valid attention weights.

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
            key = self.k_proj(key)
            value = self.v_proj(value)

        # 2. Split heads for SDPA.
        # (bsz, seq_len, heads_n * hsz)
        # => (bsz, seq_len, heads_n, hsz)
        # => (bsz, heads_n, seq_len, hsz)
        query = query.unflatten(-1, [self.heads_n, self.hsz]).transpose(1, 2)
        key = key.unflatten(-1, [self.heads_n, self.hsz]).transpose(1, 2)
        value = value.unflatten(-1, [self.heads_n, self.hsz]).transpose(1, 2)

        # 3. SDPA.
        dropout_p = self.dropout_p if self.training else 0.0
        # Merged `bias_mask` could be passed to `scaled_dot_product_attention`
        # directly.
        # if key_padding_mask is not None or is_causal:
        #     bias_mask = self.merge_masks(
        #         key_padding_mask,
        #         attn_mask,
        #         is_causal=is_causal,
        #         query=query,
        #     )
        # else:
        #     if attn_mask is None:
        #         bias_mask = None
        #     elif attn_mask.dtype == torch.bool:
        #         bias_mask = (torch.zeros_like(attn_mask, dtype=query.dtype)
        #                      .masked_fill(attn_mask, float("-inf")))
        #     else:
        #         bias_mask = attn_mask
        # `merge_mask` will preprocess all masks properly.
        bias_mask = self.merge_masks(
            key_padding_mask,
            attn_mask,
            is_causal=is_causal,
            query=query,
            key=key,
        )
        if need_weights:
            attn_val, attn_ws = scaled_dot_product_attention(
                query, key, value,
                attn_mask=bias_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            # Attention: `F.scaled_dot_product_attention`'s `attn_mask` mask logic is opposite to the all others.
            # Match the attention mask for SPDA of (bsz, heads_n, qslen, kvslen).
            # from (bsz, qslen, kvslen)
            # or (qslen, kvslen)
            if bias_mask is not None:
                # set_trace()
                if bias_mask.dim() == 3:
                    bias_mask = bias_mask.unsqueeze(1)
                elif bias_mask.dim() == 2:
                    bias_mask = bias_mask.unsqueeze(0).unsqueeze(0)
            attn_val = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=bias_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
            attn_ws = None
        # (bsz, heads_n, seq_len, hsz)
        # => (bsz, seq_len, heads_n, hsz)
        # => (bsz, seq_len, heads_n * hsz)
        attn_val = attn_val.transpose(1, 2).flatten(-2)

        return self.out_proj(attn_val), attn_ws

    def _reset_parameter(self):
        """Init parameters with xavier_uniform."""
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj.weight)
            if self.bias:
                constant_(self.in_proj.bias, 0.0)
        else:
            xavier_uniform_(self.q_proj.weight)
            xavier_uniform_(self.k_proj.weight)
            xavier_uniform_(self.v_proj.weight)
            if self.bias:
                constant_(self.q_proj.bias, 0.0)
                constant_(self.k_proj.bias, 0.0)
                constant_(self.v_proj.bias, 0.0)

        xavier_uniform_(self.out_proj.weight)
        if self.bias:
            constant_(self.out_proj.bias, 0.0)

    @classmethod
    def merge_masks(
        cls,
        key_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
        query: torch.Tensor = None,
        key: torch.Tensor = None,
        dtype: str = None,
        device: str = None,
    ) -> torch.Tensor | None:
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
        is_causal: Generate causal mask for Q,K.
        query: Query tensor.
        key: Key tensor.
        dtype:
        device:

        Shape:
        --------------------------
        key_padding_mask: (batch_size, key_seq_len)
        attn_mask: ([batch_size,]query_seq_len, key_seq_len)
        query: (..., query_seq_len, embed_sz)
        key: (..., key_seq_len, embed_sz)
        RETURN:(batch_size, query_seq_len, key_seq_len)

        Return:
        --------------------------
        Tensor[ninf, 0.0] or None
        """
        # Return None directly if no mask need to be generated.
        if (key_padding_mask is None
                and attn_mask is None
                and not is_causal):
            return None
        # Query length is required to generate causal mask.
        assert not (is_causal and query is None and attn_mask is None), (
            "Query or attn_mask is required to generate causal attention mask."
        )
        assert key_padding_mask is None or key_padding_mask.dim() == 2, (
            "Only 2D key padding mask with shape(bsz, kvslen) is allowed."
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        # Get mask sizes: bsz, qslen, kslen
        if key_padding_mask is not None:
            bsz = key_padding_mask.size(0)
        elif attn_mask is not None and attn_mask.dim() == 3:
            bsz = attn_mask.size(0)
        else:
            bsz = 1

        if attn_mask is not None:
            qslen = attn_mask.size(-2)
        elif is_causal and query is not None:
            qslen = query.size(-2)
        else:
            qslen = 1

        if key_padding_mask is not None:
            kslen = key_padding_mask.size(-1)
        elif attn_mask is not None:
            kslen = attn_mask.size(-1)
        elif is_causal and key is not None:
            kslen = key.size(-2)
        else:
            kslen = qslen

        # Return attention mask directly.
        if (attn_mask is not None
                and attn_mask.dtype != torch.bool
                and key_padding_mask is None
                and not is_causal):
            if attn_mask.dim() == 3:
                return attn_mask.to(**factory_kwargs)
            else:
                return attn_mask.unsqueeze(0).to(**factory_kwargs)

        # Init bias mask.
        bias_mask = torch.zeros(bsz, qslen, kslen, **factory_kwargs)
        if is_causal:
            # set_trace()
            tril_mask = torch.ones(qslen, kslen, dtype=torch.bool).tril(diagonal=0)
            bias_mask.masked_fill_(tril_mask.logical_not(), float("-inf"))

        # Merge key_padding_mask and attention mask.
        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                bias_mask.masked_fill_(
                    key_padding_mask.view(bsz, 1, kslen),
                    float("-inf")
                )
            else:
                bias_mask += key_padding_mask.broadcast_to(bsz, qslen, kslen)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                bias_mask.masked_fill_(
                    attn_mask.view(1, qslen, kslen),
                    float("-inf")
                )
            else:
                bias_mask += attn_mask.broadcast_to(bsz, qslen, kslen)

        return bias_mask
