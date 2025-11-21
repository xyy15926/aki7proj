#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: unet.py
#   Author: xyy15926
#   Created: 2025-09-09 18:39:56
#   Updated: 2025-11-21 22:38:40
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

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class DoubleConv(nn.Module):
    """Double Convolution.

    Sequential double Conv2d with BN and ReLU.
    """
    def __init__(
        self,
        cin:int = 4,
        cout:int = 8,
        device: str = None,
        dtype: str = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Pad to keep the shape unchanged so to make it easy to concat
        # the channels in UP-decoder process for skip-connection.
        self.double_conv = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, **factory_kwargs),
            nn.BatchNorm2d(cout, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, **factory_kwargs),
            nn.BatchNorm2d(cout, **factory_kwargs),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        inp:torch.Tensor
    ) -> torch.Tensor:
        """DoubleConv forward.
        """
        return self.double_conv(inp)


# %%
class UNetDown(nn.Module):
    """Down scaling.

    MaxPool2d + DoubleConv acts as the encoder to extract the global
      information.
    """
    def __init__(
        self,
        cin: int,
        cout: int,
        device: str = None,
        dtype: str = None,
    ):
        """UNet down-scaling initiation.

        Params:
        ---------------------------
        cin: Number of channels of the input tensor.
        cout: Number of channels of the output tensor.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.maxpool_dconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(cin, cout, **factory_kwargs)
        )

    def forward(
        self,
        inp:torch.Tensor
    ) -> torch.Tensor:
        """UNet down scaling forward.

        Shape:
        ---------------------------
        inp: [bsz, channel_n, height, width]
        """
        return self.maxpool_dconv(inp)


class UNetUp(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        bilinear: bool = False,
        device: str = None,
        dtype: str = None,
    ):
        """UNet up-scaling initiation.

        Params:
        ---------------------------
        cin: Number of channels of the input tensor for upsampling.
          The number of channels of the skipped tensor will be calculated
          automatically for convolution after upsampling.
        cout: Number of channels of the output tensor.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # `nn.Upsample` may be faster than `nn.ConvTranspose2d` for non-parameter.
        # but less powerful.
        if bilinear:
            self.up_samp = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
        else:
            self.up_samp = nn.ConvTranspose2d(
                cin,
                cin,
                kernel_size=2,
                stride=2,
                **factory_kwargs,
            )
        self.dconv = DoubleConv(cin + cin // 2, cout, **factory_kwargs)

    def forward(
        self,
        inp: torch.Tesnor,
        skip: torch.Tensor,
    ):
        """UNet up scaling forward.

        Shape:
        ---------------------------
        inp: [bsz, channel_n, height, width]
        skip: [bsz, channel_n, height, width]
        """
        inp = self.up_samp(inp)
        inp = torch.concat([skip, inp], dim=1)
        return self.dconv(inp)


# %%
class UNet(nn.Module):
    """UNet.

    Ref:
    ---------------------------
    - https://cuijiahua.com/blog/2019/12/dl-15.html
    - https://zhuanlan.zhihu.com/p/97488817
    """
    def __init__(
        self,
        cin: int,
        cout: int,
        class_n: int = None,
        bilinear: bool = False,
        device: str = None,
        dtype: str = None,
    ):
        """UNet initialization.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.cin = cin
        self.cout = cout
        self.class_n = class_n
        self.in_conv = DoubleConv(cin, 4, **factory_kwargs)
        # The in-channel and out-channel must be set carefully.
        self.dnc1 = UNetDown(4, 8, **factory_kwargs)
        self.dnc2 = UNetDown(8, 16, **factory_kwargs)
        self.upc2 = UNetUp(16, 8, bilinear, **factory_kwargs)
        self.upc1 = UNetUp(8, 4, bilinear, **factory_kwargs)
        self.out_conv = nn.Conv2d(4, cout, kernel_size=1, **factory_kwargs)

    def forward(
        self,
        inp: torch.Tensor
    ):
        """UNet forward.

        Shape:
        ---------------------------
        inp: [bsz, channel_n, height, width]
        """
        inp = self.in_conv(inp)
        dn1 = self.dnc1(inp)
        dn2 = self.dnc2(dn1)
        up2 = self.upc2(dn2, dn1)
        up1 = self.upc1(up2, inp)
        oup = self.out_conv(up1)

        return F.sigmoid(oup)
