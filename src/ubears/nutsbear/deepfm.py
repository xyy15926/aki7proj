#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: deepfm.py
#   Author: xyy15926
#   Created: 2024-06-27 11:17:08
#   Updated: 2025-07-08 14:11:30
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import List, Any, Tuple
from collections.abc import Sequence
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

DEVICE = ("cuda"
          if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu")


# %%
class DeepFM(nn.Module):
    """DeepFM.

    Ref:
    ---------------------------
    - DeepFM and CTR: https://fancyerii.github.io/2019/12/19/deepfm/
    - DeepFM by PyTorch: https://zhuanlan.zhihu.com/p/332786045

    Attrs:
    ---------------------------
    fea_num: Number of features.
    fea_sizes: Sizes of the features, namley the length of the tensor to
      represent.
    emb_size: Size of the embedding tensor of the FM and DNN input.
    hidden_sizes: Sizes of the layers of the DNN part.
    fm_dropout_probs: The probability of the dropout layers after the embeding
      in FM.
    deep_dropout_probs: The probability of the dropout layers in DNN.
    deep_bn: If to add BN layer in DNN.
    writer: SummaryWriter to log.
    """
    def __init__(
        self,
        fea_catns: List[int],
        emb_sz:int = 8,
        hidden_szs:List[int] = [32, 32],
        dropout_p: float = 0.0
    ):
        super().__init__()
        self.emb_sz = emb_sz
        self.dropout_p = dropout_p
        self._init_fm_block(fea_catns)
        self._init_dnn_block(fea_catns, emb_sz * len(fea_catns), hidden_szs)

    def _init_fm_block(self, fea_catns: list[int]):
        """Init FM block.

        1. `fmo1_embeds` actes as the weights of the sparse features directly
          added up linearly in FM.
        """
        self.fmo1_embeds = nn.ModuleList(
            [nn.Embedding(catn, 1) for catn in fea_catns])
        self.fmo2_embeds = nn.ModuleList(
            [nn.Embedding(catn, self.emb_sz) for catn in fea_catns])
        self.fmo1_dropout = nn.Dropout(self.dropout_p)
        self.fmo2_dropout = nn.Dropout(self.dropout_p)

    def _init_dnn_block(
        self,
        fea_catns: list[int],
        in_sz: int,
        hidden_szs: list[int],
    ):
        """Init DNN block."""
        dnn_layers = []
        for hsz in hidden_szs:
            dnn_layers.append(nn.Linear(in_sz, hsz))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.BatchNorm1d(hsz))
            dnn_layers.append(nn.Dropout(self.dropout_p))
            in_sz = hsz
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, inp_idx: torch.Tensor, inp_val: torch.Tensor):
        """DeepFM forward.

        Params:
        --------------------------
        inp_idx: The index of the category in the corespondant field.
          Namely the position of the 1 in the one-hot encoding for the
          categorical feature.
          And the index should always be 0 for the numeric features.
        inp_val: The value of the category of the corespondant field.
          The value should be 1 for the categorical feature.

        Shape:
        --------------------------
        inp_idx: (batch_size, fea_num)
        inp_val: (batch_size, fea_num)
        RETURN: (batch_size,)

        Return:
        --------------------------
        Sigmoid probability.
        """
        # Embed.
        o1emb, o2emb = self._embed(inp_idx, inp_val)

        # FM and DNN forward.
        fmo1_out, fmo2_out = self._fm_forward(o1emb, o2emb)
        dnn_out = self.dnn(o2emb.flatten(1))

        # Sigmoid output.
        outp = F.sigmoid(fmo1_out.sum(dim=-1) + fmo2_out.sum(dim=-1)
                         + dnn_out.sum(dim=-1))

        return outp

    def _embed(self, inp_idx: torch.Tensor, inp_val: torch.Tensor):
        """Embedding block.

        The numerical feature will be the `embedding * value` after embedding.
        1. For order-1, the embeddings act as the weights for both the
          categorical and numerics features.
        2. For order-2, the embedding is the hidden vector for interactions
          between the features.

        Shape:
        --------------------------
        inp_idx: (batch_size, fea_num)
        inp_val: (batch_size, fea_num)
        o1emb: (batch_size,)
        o2emb: (batch_size, embed_size)

        """
        bsz, fea_num = inp_idx.size()
        # [(bsz, 1),...]
        o1emb_arr = [
            emb(inp_idx[:, i]) * torch.unsqueeze(inp_val[:, i], -1)
            for i, emb in enumerate(self.fmo1_embeds)
        ]
        # [(bsz, 1, embed_sz),...]
        o2emb_arr = [
            emb(inp_idx[:, i]).unsqueeze(1) * inp_val[:, i].view(-1, 1, 1)
            for i, emb in enumerate(self.fmo2_embeds)
        ]
        # (bsz, fea_num)
        o1emb = torch.cat(o1emb_arr, dim=1).view(bsz, -1)
        # (bsz, fea_num, emb_sz)
        o2emb = torch.cat(o2emb_arr, dim=1)

        return o1emb, o2emb

    def _fm_forward(self, o1emb: torch.Tensor, o2emb: torch.Tensor):
        """FM block forward."""
        o1out = self.fmo1_dropout(o1emb)
        o2out = self.fmo2_dropout(o2emb)
        o2out = 0.5 * (torch.sum(o2out, dim=1) ** 2 - torch.sum(o2out ** 2, dim=1))
        return o1out, o2out
