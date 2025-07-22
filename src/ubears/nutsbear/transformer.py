#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: transformer.py
#   Author: xyy15926
#   Created: 2025-07-10 09:25:01
#   Updated: 2025-07-22 22:50:36
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
# from torch.nn import MultiheadAttention
from ubears.nutsbear.attention import MultiheadAttention, SimpleMHA
# from IPython.core.debugger import set_trace

# If `need_weights` is set, customed SDPA will be used instead of
# `F.scaled_dot_product_attention`, which may lead to NaNs after the Softmax
# in customed SDPA for query with all NInf(masked) and lead to NaN in `.grad`
# in `.backward()` process.
NEED_ATTN_WEIGHTS = False


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class Transformer(nn.Module):
    """Transformer.

    Attrs:
    --------------------------
    encoder: Transformer encoder.
    decoder: Transformer decoder.
    """
    def __init__(
        self,
        embed_sz: int,
        heads_n: int,
        ffn_sz: int,
        dropout_p: float = 0.0,
        enc_layers_n: int = 6,
        dec_layers_n: int = 6,
        encoder: "TransformerEncoder" = None,
        decoder: "TransformerEncoder" = None,
    ):
        """Init Transformer.

        Params:
        --------------------------
        embed_sz: The size of embedding for the MHA in encoder and decoder.
        heads_n: The number of heads in the MHA in encoder and decoder.
        ffn_sz: The size of hidden layer in the FFN in encoder and decoder.
        dropout_p: The probability of the dropout.
        enc_layers_n: The number of encoder layers in encoder.
        dec_layers_n: The number of decoder layers in decoder.
        encoder: The customed encoder.
          This will be set as the encoder in transformer directly if provided
          instead of initing a encoder with previous parameters.
        decoder: The customed decoder.
          This will be set as the decoder in transformer directly if provided
          instead of initing a decoder with previous parameters.
        """
        super().__init__()
        if encoder is None:
            encoder_layer = TransformerEncoderLayer(
                embed_sz, heads_n, ffn_sz, dropout_p)
            self.encoder = TransformerEncoder(encoder_layer, enc_layers_n)
        else:
            self.encoder = encoder
        if decoder is None:
            decoder_layer = TransformerDecoderLayer(
                embed_sz, heads_n, ffn_sz, dropout_p)
            self.decoder = TransformerDecoder(decoder_layer, dec_layers_n)
        else:
            self.decoder = decoder

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        src_is_causal: bool = False,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        """Transformer forward.

        Params:
        ---------------------------
        src: Source sequence.
        tgt: Target sequence for the decoder as the input.
        memory: Memory sequence for attention, which may be the output of
          the transformer encoder or something else.
        src_mask: Attention mask for source sequence.
        tgt_mask: Attention mask for target sequence.
        memory_mask: Attention mask for memory sequence.
        src_key_padding_mask: Padding mask for source sequence.
        tgt_key_padding_mask: Padding mask for target sequence.
        memory_key_padding_mask: Padding mask for memory sequence.
        src_is_causal: If to enforce causality, namely tokens can only attend
          previous tokens.
        tgt_is_causal: Ditto
        memory_is_causal: Ditto.

        Shape:
        ---------------------------
        src: (batch_size, src_seq_len, embed_size)
        tgt: (batch_size, tgt_seq_len, embed_size)
        src_mask: (src_seq_len, src_seq_len)
        tgt_mask: (tgt_seq_len, tgt_seq_len)
        memory_mask: (src_seq_len, tgt_seq_len)
        src_key_padding_mask: (batch_size, src_seq_len)
        tgt_key_padding_mask: (batch_size, tgt_seq_len)
        memory_key_padding_mask: (batch_size, src_seq_len)
        RETURN: (batch_size, tgt_seq_len, embed_size)

        Return:
        ---------------------------
        Output of the decoder.
        """
        memory = self.encoder(
            src, src_mask, src_key_padding_mask, src_is_causal)
        outp = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            tgt_is_causal,
            memory_is_causal)
        return outp


# %%
class TransformerEncoder(nn.Module):
    """Transformer Encoder.

    Attrs:
    --------------------------
    layers: ModuleList of TransformerEncoderLayer.
    layers_n: Number of TransformerEncoderLayers in the encoder.

    Ref:
    --------------------------
    - PyTorch Transformer:
      - https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/transformer.py
    """
    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        layers_n: int = 6,
    ):
        """Init Transformer Encoder.

        Params:
        --------------------------
        encoder_layer: TransformerEncoderLayer instances for stacking.
        layers_n: Number of TransformerEncoderLayers in the encoder.
        """
        super().__init__()
        # `copy.deepcopy` may be more time-efficient than sequent init.
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(layers_n)])
        self.layers_n = layers_n

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Transformer Encoder part forward.

        Params:
        -------------------------
        src: Source sequence.
        src_mask: Attention mask for source sequence.
        src_key_padding_mask: Padding mask for source sequence.
        is_causal: If to enforce causality, namely tokens can only attend to
          previous tokens.

        Shape:
        ---------------------------
        src: (batch_size, src_seq_len, embed_size)
        src_mask: (src_seq_len, src_seq_len)
        src_key_padding_mask: (batch_size, src_seq_len)
        RETURN: (batch_size, src_seq_len, embed_size)

        Return:
        ---------------------------
        Encoder result with the same shape of source sequence.
        """
        outp = src
        # Merge the `key_padding_mask` and `attn_mask` only once.
        if src_key_padding_mask is not None or is_causal:
            bias_mask = MultiheadAttention.merge_masks(
                src_key_padding_mask,
                src_mask,
                is_causal=is_causal,
                query=src,
                key=src,
            )
        else:
            bias_mask = src_mask
        # Skip padding mask and causality since maskes are all merged.
        for mod in self.layers:
            outp = mod(
                outp,
                src_mask=bias_mask,
                src_key_padding_mask=None,
                is_causal=False
            )
        return outp


# %%
class TransformerDecoder(nn.Module):
    """Transformer Decoder.

    Attrs:
    --------------------------
    layers: ModuleList of TransformerDecoderLayer.
    layers_n: Number of TransformerDecoderLayers in the decoder.

    Ref:
    --------------------------
    - PyTorch Transformer:
      - https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/transformer.py
    """
    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        layers_n: int = 6,
    ):
        """Init Transformer Decoder.

        Params:
        --------------------------
        decoder_layer: TransformerDecoderLayer instances for stacking.
        layers_n: Number of TransformerDecoderLayers in the decoder.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for i in range(layers_n)]
        )
        self.layers_n = layers_n

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        """Transformer Encoder part forward.

        Params:
        -------------------------
        tgt: Target sequence for the decoder as the input.
        memory: Memory sequence for attention, which may be the output of
          the transformer encoder or something else.
        tgt_mask: Attention mask for target sequence.
        memory_mask: Attention mask for memory sequence.
        tgt_key_padding_mask: Padding mask for target sequence.
        memory_key_padding_mask: Padding mask for memory sequence.
        tgt_is_causal: If to enforce causality, namely tokens can only attend
          to previous tokens.
        memory_is_causal: Ditto.

        Shape:
        ---------------------------
        tgt: (batch_size, tgt_seq_len, embed_size)
        memory: (batch_size, mem_seq_len, embed_size)
        tgt_mask: (tgt_seq_len, tgt_seq_len)
        memory_mask: (mem_seq_len, tgt_seq_len)
        tgt_key_padding_mask: (batch_size, tgt_seq_len)
        memory_key_padding_mask: (batch_size, mem_seq_len)
        RETURN: (batch_size, tgt_seq_len, embed_size)

        Return:
        ---------------------------
        Decoder result with the same shape of the target sequence.
        """
        outp = tgt
        # Merge the `key_padding_mask` and `attn_mask` only once.
        if tgt_key_padding_mask is not None or tgt_is_causal:
            self_bias_mask = MultiheadAttention.merge_masks(
                tgt_key_padding_mask,
                tgt_mask,
                is_causal=tgt_is_causal,
                query=tgt,
                key=tgt,
            )
        else:
            self_bias_mask = tgt_mask
        if memory_key_padding_mask is not None or memory_is_causal:
            cross_bias_mask = MultiheadAttention.merge_masks(
                memory_key_padding_mask,
                memory_mask,
                is_causal=memory_is_causal,
                query=tgt,
                key=memory,
            )
        else:
            cross_bias_mask = memory_mask
        # DecoderLayer forward sequently.
        # Skip padding mask and causality since maskes are all merged.
        for mod in self.layers:
            outp = mod(
                outp,
                memory,
                tgt_mask=self_bias_mask,
                memory_mask=cross_bias_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_is_causal=False,
                memory_is_causal=False,
            )

        return outp


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
        attn_style: str = "SDPA",
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
        attn_style: The attention style(module).
          SDPA: MultiheadAttention by default.
          SingleW: SimpleMHA.
        """
        super().__init__()
        if attn_style == "SDPA":
            self.self_attn = MultiheadAttention(
                embed_sz,
                heads_n,
                dropout_p=dropout_p,
            )
        elif attn_style == "SingleW":
            self.self_attn = SimpleMHA(
                embed_sz,
                heads_n,
                dropout_p=dropout_p,
            )
        else:
            raise ValueError("Invalid MHA style.")
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
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Self-Attention block."""
        # Reset `need_weights` to use `F.scaled_dot_product_attention`
        # instead the customed one
        # 1. Acceleration (maybe).
        # 2. No NaN from `F.softmax` when all-NInf.
        attn_val, attn_ws = self.self_attn(
            inp, inp, inp,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            is_causal=is_causal,
            need_weights=NEED_ATTN_WEIGHTS,
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
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Network forward.

        Apply Add & Normlization after Self-Attention and FeedForward Network.

        Params:
        -------------------------
        src: Source sequence.
        src_mask: Attention mask for source sequence.
        src_key_padding_mask: Padding mask for source sequence.
        is_causal: If to enforce causality, namely tokens can only attend to
          previous tokens.

        Shape:
        ---------------------------
        src: (batch_size, src_seq_len, embed_size)
        src_mask: (src_seq_len, src_seq_len)
        src_key_padding_mask: (batch_size, src_seq_len)
        RETURN: (batch_size, src_seq_len, embed_size)

        Return:
        ---------------------------
        Encoder result with the same shape of source sequence.
        """
        src = self.norm1(
            src
            + self._sa_block(
                src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal
            )
        )
        src = self.norm2(src + self._ffn_block(src))
        return src


# %%
class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer.

    Attrs:
    --------------------------
    self_attn: MHA module for self-attention of target.
    sa_dropout: Dropout module after `self_attn`.
    cross_attn: MHA module for cross-attention between target and memery.
    ca_dropout: Dropout module after `cross_attn`.
    ffn_linear1: FFN layer 1.
    ffn_linear2: FFN layer 2.
    ffn_activation: Activation after FFN layer1.
    ffn_dropout1: Dropout module after FFN layer1.
    ffn_dropout2: Dropout module after FFN layer2.
    norm1: Layer norm module after `self_attn`.
    norm2: Layer norm module after `cross_attn`.
    norm3: Layer norm module after FFN.

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
        attn_style: str = "SDPA",
    ):
        """Decoder Layer Initialization.

        1. The size of query, key, value and embeding in MHA will be the same.
        2. Two linear layers will perform `embed_sz * 1 * 1` 1D-Conv.

        Params:
        --------------------------
        embed_sz: The size of embedding for the MHA.
        heads_n: The number of heads in the MHA.
        ffn_sz: The size of hidden layer in the FFN.
        dropout_p: The probability of the dropout.
        attn_style: The attention style(module).
          SDPA: MultiheadAttention by default.
          SingleW: SimpleMHA.
        """
        super().__init__()
        if attn_style == "SDPA":
            self.self_attn = MultiheadAttention(
                embed_sz,
                heads_n,
                dropout_p=dropout_p,
            )
            self.cross_attn = MultiheadAttention(
                embed_sz,
                heads_n,
                dropout_p=dropout_p,
            )
        elif attn_style == "SingleW":
            self.self_attn = SimpleMHA(
                embed_sz,
                heads_n,
                dropout_p=dropout_p,
            )
            self.cross_attn = SimpleMHA(
                embed_sz,
                heads_n,
                dropout_p=dropout_p,
            )
        else:
            raise ValueError("Invalid MHA style.")
        self.sa_dropout = nn.Dropout(dropout_p)
        self.ca_dropout = nn.Dropout(dropout_p)

        self.ffn_linear1 = nn.Linear(embed_sz, ffn_sz, bias=True)
        self.ffn_linear2 = nn.Linear(ffn_sz, embed_sz, bias=True)
        self.ffn_activation = nn.ReLU()
        self.ffn_dropout1 = nn.Dropout(dropout_p)
        self.ffn_dropout2 = nn.Dropout(dropout_p)

        self.norm1 = nn.LayerNorm(embed_sz)
        self.norm2 = nn.LayerNorm(embed_sz)
        self.norm3 = nn.LayerNorm(embed_sz)

    def _sa_block(
        self,
        inp: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Self-Attention block for target sequence."""
        # Reset `need_weights` to use `F.scaled_dot_product_attention`
        # instead the customed one
        # 1. Acceleration (maybe).
        # 2. No NaN from `F.softmax` when all-NInf.
        attn_val, attn_ws = self.self_attn(
            inp, inp, inp,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            is_causal=is_causal,
            need_weights=NEED_ATTN_WEIGHTS,
        )
        return self.sa_dropout(attn_val)

    def _ca_block(
        self,
        tgt: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Cross-Attention block for target and memory."""
        # Reset `need_weights` to use `F.scaled_dot_product_attention`
        # instead the customed one
        # 1. Acceleration (maybe).
        # 2. No NaN from `F.softmax` when all-NInf.
        attn_val, attn_ws = self.cross_attn(
            tgt, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=NEED_ATTN_WEIGHTS,
        )
        return self.ca_dropout(attn_val)

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
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        """Network forward.

        Apply Add & Normlization after Self-Attention, Cross-Attention and
        FeedForward Network.

        Params:
        -------------------------
        tgt: Target sequence for the decoder as the input.
        memory: Memory sequence for attention, which may be the output of
          the transformer encoder or something else.
        tgt_mask: Attention mask for target sequence.
        memory_mask: Attention mask for memory sequence.
        tgt_key_padding_mask: Padding mask for target sequence.
        memory_key_padding_mask: Padding mask for memory sequence.
        tgt_is_causal: If to enforce causality, namely tokens can only attend
          to previous tokens.
        memory_is_causal: Ditto.

        Shape:
        ---------------------------
        tgt: (batch_size, tgt_seq_len, embed_size)
        tgt_mask: (tgt_seq_len, tgt_seq_len)
        memory_mask: (mem_seq_len, tgt_seq_len)
        tgt_key_padding_mask: (batch_size, tgt_seq_len)
        memory_key_padding_mask: (batch_size, mem_seq_len)
        RETURN: (batch_size, tgt_seq_len, embed_size)

        Return:
        ---------------------------
        Decoder result with the same shape of the target sequence.
        """
        tgt = self.norm1(
            tgt
            + self._sa_block(
                tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=tgt_is_causal,
            )
        )
        tgt = self.norm2(
            tgt
            + self._ca_block(
                tgt,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                is_causal=memory_is_causal,
            )
        )
        tgt = self.norm3(tgt + self._ffn_block(tgt))
        return tgt
