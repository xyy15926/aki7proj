#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: diffusion.py
#   Author: xyy15926
#   Created: 2025-09-04 09:02:38
#   Updated: 2025-09-14 23:06:31
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
from tqdm import tqdm
# from IPython.core.debugger import set_trace


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class GaussianDiffusion:
    """Denoise Diffustion Probability Model.

    Ref:
    -----------------------------
    - https://github.com/Allenem/DDPM
    - https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py
    - https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/simple_diffusion.py

    Attr:
    -----------------------------
    steps_n: The number of steps of diffusion.
    beta: Diffusion rates of each steps.
    alpha: 1 - beta
    alpha_bar: \\prod alpha_i
    """
    def __init__(
        self,
        steps_n: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        """Gaussian Diffusion initialization.

        Params:
        -----------------------------
        steps_n: The number of steps of diffusion.
        beta_start: The minimum diffusion rate.
        beta_end: The maximum diffusion rate.
        """
        super().__init__()
        self.steps_n = steps_n
        self.beta = nn.Buffer(torch.linspace(beta_start, beta_end, steps_n))
        self.alpha = nn.Buffer(1. - self.beta)
        self.alpha_bar = nn.Buffer(torch.cumprod(self.alpha, dim=0))

    def addnoise(
        self,
        inp: torch.Tensor,
        tsteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add gaussian noise with diffusion process.

        Params:
        -----------------------------
        inp: Input tensor.
        tsteps: The number of diffusion steps to add noise.

        Shape:
        -----------------------------
        inp: [bsz, channel_n, ...]
        tsteps: [bsz,]
        RETURN_noised: [bsz, channel_n, ...]
        RETURN_zt: [bsz, channel_n, ...]

        Return:
        -----------------------------
        noised: Noised input tensor.
        zt: Guassian noise.
        """
        zt = torch.randn_like(inp)
        ba = self.alpha_bar[tsteps].view([-1,] + [1,] * (inp.dim() - 1))
        noised = torch.sqrt(ba) * inp + torch.sqrt(1. - ba) * zt
        return noised, zt

    @torch.no_grad()
    def denoise(
        self,
        model: nn.Module,
        inp: torch.Tensor | tuple | list,
        tstart: int = None,
        labels: torch.Tensor = None,
    ):
        """Remove gaussian noise with backward diffusion process.

        Params:
        -----------------------------
        model: Model to predict noise from input tesnor.
        inp: Noised tensor or tensor size for random gaussion noise.
        tstart: The start step of backward diffusion.
        labels: Labels of the input tensor.d

        Shape:
        -----------------------------
        inp: [bsz, channel_n, ...]
        tstart: [bsz,]
        labels: [bsz,]
        RETURN: [bsz, channel_n, ...]

        Return:
        -----------------------------
        Tensor after noise removed from input tensor.
        """
        if isinstance(inp, (tuple, list)):
            inp_sz = inp
            inp = torch.randn(inp)
        else:
            inp_sz = inp.size()
        bsz = inp_sz[0]

        model.eval()
        tstart = self.steps_n if tstart is None else tstart
        for stn in tqdm(range(tstart - 1, -1, -1)):
            t = torch.ones(bsz, dtype=torch.int) * stn
            beta = self.beta[t].view([-1,] + [1,] * (len(inp_sz) - 1))
            alpha = self.alpha[t].view([-1,] + [1,] * (len(inp_sz) - 1))
            ba = self.alpha_bar[t].view([-1,] + [1,] * (len(inp_sz) - 1))

            # Predict and remove noise.
            pred_zt = model(inp, t, labels)
            if stn > 0:
                noise = torch.randn(inp_sz)
            else:
                noise = torch.zeros(inp_sz)
            inp = (
                1 / torch.sqrt(alpha)
                * (inp - beta / torch.sqrt(1. - ba) * pred_zt)
                + noise * torch.sqrt(beta)
            )
        model.train()
        # inp = (inp.clamp(-1, 1) + 1) / 2 * 255
        # return inp.to(torch.int8)
        return inp
