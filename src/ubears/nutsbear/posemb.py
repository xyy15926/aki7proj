#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: posemb.py
#   Author: xyy15926
#   Created: 2025-09-12 19:51:13
#   Updated: 2025-09-12 20:15:50
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
class SinusoidalPECache:
    """Sinusoidal Positional Embedding.
    Ref:
    -------------------------------
    - SinusoidalPE:
      - https://zhuanlan.zhihu.com/p/359500899
      - https://www.cnblogs.com/yanshw/p/16740972.html

    Attrs:
    -------------------------------
    pe_cache: PE cache.
    base: The base of exponential for theta in PE.

    Shape:
    -------------------------------
    pe_cache: [..., cur_len, embed_sz]
    """
    def __init__(
        self,
        base: float = 1e4,
    ):
        self.base = base
        self.pe_cache = None

    def get_pe(
        self,
        pos: int | torch.Tensor,
        esz: int,
        device: str = None,
        dtype: str = None,
    ) -> torch.Tensor:
        """Get the SinPE.

        Params:
        -----------------------------------
        pos: Positions.
          NOTE: The last embedding(`pos`) is not included for `pos` of integer
            dtype.
        esz: Position embedding size.

        Shape:
        -----------------------------------
        pos: int | [slen]
        RETURN: [slen, esz]
        """
        if not isinstance(pos, int):
            max_pos = torch.max(pos) + 1
        else:
            max_pos = pos

        # Update PE cache if necessary.
        if self.pe_cache is None:
            self.pe_cache = SinusoidalPECache.sinpe(
                max_pos,
                esz,
                self.base,
                device=device,
                dtype=dtype,
            )
        else:
            clen, csz = self.pe_cache.size()
            if max_pos > clen or esz > csz:
                self.pe_cache = SinusoidalPECache.sinpe(
                    max(max_pos, clen),
                    max(esz, csz),
                    self.base,
                    device=device,
                    dtype=dtype,
                )

        if isinstance(pos, int):
            return self.pe_cache[:pos, :esz]
        else:
            return self.pe_cache[pos, :esz]

    @staticmethod
    def sinpe(
        slen: int,
        emb_sz: int,
        base: float = 1e4,
        device: str = None,
        dtype: str = None,
    ) -> torch.Tensor:
        """Compute the SinPE.

        Params:
        ----------------------------------
        slen: The length of the SinPE.
        emb_sz: The embedding size of the SinPE.
        base: The base of exponential for theta in PE.

        Shape:
        ----------------------------------
        RETURN: [slen, emb_sz]

        Return:
        ----------------------------------
        SinPE addup.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        pe_cache = torch.zeros(slen, emb_sz, **factory_kwargs)
        theta = -torch.arange(0, emb_sz, 2, **factory_kwargs) / emb_sz
        # theta = -torch.arange(0, esz // 2, **factory_kwargs) / esz
        theta = torch.pow(base, theta)
        pos = torch.arange(0, slen).unsqueeze(1).float()
        pe_cache[:, 0::2] = torch.sin(pos * theta)
        pe_cache[:, 1::2] = torch.cos(pos * theta)
        return pe_cache


_sinpe_cache = SinusoidalPECache()


# %%
class SinPE(nn.Module):
    """Sinusoidal Positional Embedding.

    Use SinusoidalPECache as cache.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor = None,
    ) -> torch.Tensor:
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
        Tensor after SinPE embedding.
        """
        # In case that x of integer dtype passed in.
        if torch.is_floating_point(x):
            dtype = x.dtype
        else:
            dtype = torch.float
        factory_kwargs = {"device": x.device, "dtype": dtype}
        *_____, slen, sz = x.size()
        if pos is not None:
            assert pos.size(-1) == slen, "All positions must be provided."
        else:
            pos = slen
        return x + _sinpe_cache.get_pe(pos, sz, **factory_kwargs)


# %%
class RotaryPE(nn.Module):
    """Rotary Position Embedding.

    Ref:
    -------------------------------
    - RoPE:
      - https://www.zhihu.com/tardis/bd/art/647109286
      - https://arxiv.org/pdf/2104.09864
      - https://zhuanlan.zhihu.com/p/359502624

    Attrs:
    -------------------------------
    embed_sz: Embeding size.
      Only tensor with same embedding size(last dim size) should be encoded.
    cos_cache: Cosine part of the rotary cache.
    sin_cache: Sine part of the rotary cache.
    base: The base of exponential for theta in PE.

    Shape:
    -------------------------------
    cos_cache: [..., cur_len, embed_sz]
    sin_cache: [..., cur_len, embed_sz]
    """

    def __init__(
        self,
        embed_sz: int,
        base: float = 1e4,
    ):
        """Init Module.

        Params:
        -------------------------------
        embed_sz: The size of the last dimension of the tensor to be embeded.
        base: The base of exponential for theta in PE.
        """
        super().__init__()
        self.embed_sz = embed_sz
        self.base = base
        self.cos_cache = nn.Buffer(persistent=False)
        self.sin_cache = nn.Buffer(persistent=False)

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
        # In case that x of integer dtype passed in.
        if torch.is_floating_point(x):
            dtype = x.dtype
        else:
            dtype = torch.float
        factory_kwargs = {"device": x.device, "dtype": dtype}
        *_____, slen, sz = x.size()
        assert sz == self.embed_sz, "Embedding size must be the same."
        if pos is not None:
            assert pos.size(-1) == slen, "All positions must be provided."
            max_pos = torch.max(pos)
            if max_pos >= self.cos_cache.size(0):
                self.rotary_cache(max_pos + 1, **factory_kwargs)
            cos_ = self.cos_cache[pos]
            sin_ = self.sin_cache[pos]
        else:
            if slen >= self.cos_cache.size(0):
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
        theta = (self.base ** theta).unsqueeze(1).expand(-1, 2).flatten(-2)
        mtheta = torch.arange(max_pos).unsqueeze(1) * theta.unsqueeze(0)
        self.cos_cache = torch.cos(mtheta)
        sin_cache = torch.sin(mtheta)
        sin_cache[:, ::2] *= -1
        self.sin_cache = sin_cache
