#!/usr/bin/env python3
# ---------------------------------------------------------
#    Name: accencoder.py
#   Author: xyy15926
#   Created: 2025-07-15 10:51:09
#   Updated: 2025-07-23 11:34:11
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple
from collections.abc import Mapping, Sequence
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_uniform_
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
# from IPython.core.debugger import set_trace

if __name__ == "__main__":
    from importlib import reload
    from ubears.nutsbear import transformer, attention
    reload(attention)
    reload(transformer)

from ubears.nutsbear.attention import RotaryPE
from ubears.nutsbear.transformer import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
)

MON_N = 60
MON_CATS_MAPPING = torch.tensor(
    [0,] * 12 + [1,] * 12 + [2,] * 12 + [3,] * 12 + [4,] * 12
)
MON_CATS_N = len(np.unique(MON_CATS_MAPPING))
REPAY_STATUS_N = 13
ACC_CATS_N = 30
AMT_CATS_N = 7


# %%
# 1. 整体 Attention Encoder
# 2. 单个账户 Attention Decoder
# 3. AccMark 做引导 token
class AccRecsTransformer(nn.Module):
    """Transformer for account records in PBOC.

    1. Repayment records will used as the basic input sequence:
      1.2. For encoder, all the repayment records of all accounts will be
        flattened concated together as the source sequence.
      1.2. For decoder, the target sequence will be the repayment records
        of the per-account. And the output of the encoder will expanded to fit
        for the number of the records of each account.

    2. input = repst * mongap * ovdlv + accat


    Attrs:
    --------------------------
    repst_embed: Embedding layer for repay status.
    accat_embed: Embeding layer for account mark.
    mongap_w: The weights of the gap of the months.
    embed_sz: Embedding size for the transformer.
    out_proj: Projection to predict the repay status.
    """
    def __init__(
        self,
        embed_sz: int,
        heads_n: int,
        ffn_sz: int,
        enc_layers_n: int = 2,
        dec_layers_n: int = 1,
        dropout_p: float = 0.0,
        attn_style: str = "SDPA",
        device: str = None,
        dtype: str = None,
    ):
        """Init module.

        Params:
        ---------------------------
        embed_sz: Embedding size of encoder and decoder.
        heads_n: The number of heads
        ffn_sz: Hidden size of the feed-forward network.
        enc_layers_n: The number of the encoder layers.
        dec_layers_n: The number of the decoder layers.
        dropout_p: The probability of dropout.
        """
        super().__init__()
        self.embed_sz = embed_sz
        self.heads_n = heads_n
        self.rpe = RotaryPE(embed_sz)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Init some embedings.
        self.repst_embed = nn.Embedding(
            REPAY_STATUS_N + ACC_CATS_N,
            embed_sz,
            **factory_kwargs
        )
        # self.accat_embed = nn.Embedding(
        #     ACC_CATS_N,
        #     embed_sz,
        #     **factory_kwargs
        # )
        # Init some additional weights and bias.
        # self._inti_rep_addup(**factory_kwargs)
        self.amtlv_embed = nn.Embedding(AMT_CATS_N, 1, **factory_kwargs)

        # Init the transformer encoder and decoder.
        enc_layer = TransformerEncoderLayer(
            embed_sz,
            heads_n,
            ffn_sz,
            dropout_p=dropout_p,
            attn_style=attn_style,
        )
        self.enc = TransformerEncoder(enc_layer, enc_layers_n)
        dec_layer = TransformerDecoderLayer(
            embed_sz,
            heads_n,
            ffn_sz,
            dropout_p=dropout_p,
            attn_style=attn_style,
        )
        self.dec = TransformerDecoder(dec_layer, dec_layers_n)

    def _repst_embed(
        self,
        repay_status: torch.Tensor,
        acc_mark: torch.Tensor,
        ovda_level: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        embsz = self.embed_sz
        accat = acc_mark[..., :1]
        acc_mongap = acc_mark[..., 1]
        acc_amtlv = acc_mark[..., 2]
        bsz, slen, mon_n = repay_status.size()

        # Concat account mark with repay status.
        # Shift `accat` right by adding `REPAY_STATUS_N` up as
        # `repay_status` and `accat` share one Embedding.
        repst = torch.concat([accat + REPAY_STATUS_N, repay_status], dim=-1)

        # (bsz, slen, mon_n+1) => (bsz, slen, mon_n+1, embsz)
        # Scale the embedding up before add position encoding.
        repst = self.repst_embed(repst) * np.sqrt(embsz)
        accat_ = repst[..., 0, :].clone()

        # (bsz, slen) => (bsz, slen, 1)
        acc_amtlv = self.amtlv_embed(acc_amtlv)
        repst[..., 0, :] *= acc_amtlv
        # (bsz, slen, mon_n) => (bsz, slen, mon_n, 1)
        repst_ovdlv = self.amtlv_embed(ovda_level)
        repst[..., 1:, :] *= repst_ovdlv

        # Add the account mark embeding for correspondant repay status.
        repst[..., 1:, :] += repst[..., :1, :] / np.sqrt(embsz)

        # Rotary position encoding.
        repst[..., :1, :] = self.rpe(
            repst[..., :1, :].transpose(1, 2), acc_mongap
        ).transpose(1, 2)
        repst[..., 1:, :] = self.rpe(repst[..., 1:, :])

        # And repadding padding mask correspondantly.
        pmask = torch.concat(
            [
                torch.zeros(bsz, slen, 1, dtype=torch.bool),
                padding_mask,
            ],
            dim=-1,
        )

        return repst, pmask, accat_

    def forward(
        self,
        repay_status: torch.Tensor,
        ovda_level: torch.Tensor,
        acc_mark: torch.Tensor,
        padding_mask: torch.Tensor,
    ):
        """Module forward.

        Params:
        ---------------------------
        repay_status: Repay status.
        ovda_level: Overdue amount level.
        padding_mask: Padding mask.
        acc_mark: Account category

        Shape:
        ---------------------------
        repay_status: (bsz, acc_n, mon_n)
        ovda_level: (bsz, acc_n, mon_n)
        acc_mark: (bsz, acc_n, <CAT, MON_GAP, ...>)
        padding_mask: (bsz, acc_n, mon_n)
        RETURN: (bsz, acc_n, mon_n)

        Return:
        ---------------------------
        Account repay status prediction.
        """
        bsz, slen, mon_n = repay_status.size()

        # # (bsz, slen, mon_n) => (bsz, slen, mon_n, embsz)
        # repst = self.repst_embed(repay_status)
        # # (bsz, slen) => (bsz, slen, embsz) => (bsz, slen, 1, embsz)
        # accat = self.accat_embed(acc_mark).unsqueeze(-2)

        # if self.rep_addup_style == 1 or self.rep_addup_style == 2:
        #     inp = self._repst_01D(repst, ovda_level, accat)
        # else:
        #     inp = self._repst_2D(repst, ovda_level, accat)

        # Encoder and decoder.
        # (bsz, slen * MON_N, embed_sz)
        # enced = self.enc(
        #     inp.flatten(1, 2),
        #     src_key_padding_mask=padding_mask.flatten(1, 2)
        # )
        # deced = self.dec(
        #     inp.flatten(0, 1),
        #     enced.expand(slen, -1, -1),
        #     tgt_mask=nn.Transformer.generate_square_subsequent_mask(MON_N),
        #     tgt_key_padding_mask=padding_mask.flatten(0, 1),
        #     memory_key_padding_mask=padding_mask.flatten(1, 2).expand(slen, -1),
        #     tgt_is_causal=True,
        # )

        embed, pmask, accat_ = self._repst_embed(
            repay_status,
            acc_mark,
            ovda_level,
            padding_mask
        )

        # Extract only valid elements with padding mask first so that:
        # 1. No source key padding mask.
        # 2. No memory key padding mask.
        # As no valid.
        # The `pmask[..., 1:]` will mask the last element so that the
        # last element won't be attented in decoder.
        enced = self.enc(embed[pmask.logical_not()].unsqueeze(0))
        # set_trace()
        embed[..., 0, :] = accat_
        deced = self.dec(
            embed.flatten(0, 1)[..., :-1,:],
            enced.expand(slen, -1, -1),
            tgt_key_padding_mask=pmask[...,1:].flatten(0, 1),
            tgt_is_causal=True,
        )

        # Use the `repst_embed.weight` as the output projection.
        sft = deced @ self.repst_embed.weight[:REPAY_STATUS_N].transpose(1, 0)
        sft = F.softmax(sft, dim=-1)

        return sft
