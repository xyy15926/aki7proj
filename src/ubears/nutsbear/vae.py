#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: vae.py
#   Author: xyy15926
#   Created: 2025-08-28 22:45:24
#   Updated: 2025-09-03 23:25:24
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
class VAEBase(nn.Module):
    """Basic Variational AutoEncoder.

    Attrs:
    --------------------------
    encoder: Encoder part of the VAE.
      Encode the input to the hidden vector for the mu and sigma.
    decoder: Decoder part of the VAE.
      Recover the sample from the distribution determined by the mu and sigma
      to the quasi-source input.
    fc_mu: Linear layer to generate mean value of the latent distribution from
      the encoder output.
    fc_logsig: Linear layer to generate log of the variance of the latent
      distribution from the encoder output.
    """
    def __init__(
        self,
        lat_sz: int,
        enc_out_sz: int,
        encoder: nn.Module,
        decoder: nn.Module,
        device: str = None,
        dtype: str = None,
    ):
        """VAE initiation.

        Params:
        -------------------------------
        lat_sz: Size of the latent.
        enc_out_sz: Size of the output of the encoder.
        encoder: Encoder part for the VAE.
        decoder: Decoder part for the VAE.
        device:
        dtype:
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Encoder.
        self.encoder = encoder
        self.fc_mu = nn.Linear(enc_out_sz, lat_sz, **factory_kwargs)
        self.fc_logsig = nn.Linear(enc_out_sz, lat_sz, **factory_kwargs)

        # Decoder.
        self.decoder = decoder

    def encode(
        self,
        inp: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        """Encode.

        Params:
        -------------------------------
        inp: Input Tensor.

        Shape:
        -------------------------------
        inp: [bsz, ...]
        mu: [bsz, lat_sz]
        logsig: [bsz, lat_sz]

        Return:
        -------------------------------
        mu: Mean value of the latent distribution.
        logsigd: Log of the variance of the latent distribution.
        """
        enc_out = self.encoder(inp)
        mu = self.fc_mu(enc_out)
        logsig = self.fc_logsig(enc_out)
        return mu, logsig

    def decode(
        self,
        inp: torch.Tensor,
    ) -> torch.Tensor:
        """Decode.

        Params:
        -------------------------------
        inp: Input Tensor.

        Shape:
        -------------------------------
        inp: [bsz, ...]
        dec_out: [bsz, ...]

        Return:
        -------------------------------
        Decode result to recover the source.
        """
        dec_out = self.decoder(inp)
        return dec_out

    def reparameterize(
        self,
        mu: torch.Tensor,
        logsig: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization.

        1. Draw sample from standard gaussion distribution.
        2. Apply transformation with mean and variance value so to get the
          sample subjected to given latent distribution.

        Params:
        -------------------------------
        mu: Mean value of the latent distribution.
        logsig: Log of the variance of the latent distribution.

        Shape:
        -------------------------------
        mu: [bsz, lat_sz]
        logsig: [bsz, lat_sz]

        Return:
        -------------------------------
        Sample subject to given latent distrition.
        """
        sig = torch.exp(0.5 * logsig)
        eps = torch.randn_like(sig)
        return mu + eps * sig

    def forward(
        self,
        inp: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """VAE forward.

        1. Encode to get the mean and variance value of the latent distrition.
        2. Reparameterize to get the sample subjected to latent distribution.
        3. Decode to recover.

        Params:
        -------------------------------
        inp: Input tensor.

        Shape:
        -------------------------------
        inp: [bsz, ...]
        mu: [bsz, lat_sz]
        logsig: [bsz, lat_sz]

        Return:
        -------------------------------
        recon: Recovery result.
        mu: Mean value of the latent distribution.
        logsig: Log of the variance of the latent distribution.
        """
        mu, logsig = self.encode(inp)
        z = self.reparameterize(mu, logsig)
        recon = self.decode(z)

        return recon, mu, logsig

    @staticmethod
    def vae_loss(
        recon: torch.Tensor,
        src: torch.Tensor,
        mu: torch.Tesnor,
        logsig: torch.Tensor,
        mu_: torch.Tensor = 0,
    ) -> torch.float64:
        """Compute the loss for VAE.

        loss = ce(recon, src) + kl(latent_dist, norm(mu, 1))

        Params:
        -------------------------------
        recon: Recovery result.
        src: Source input.
        mu: Mean value of the latent distribution.
        logsig: Log of the variance of the latent distribution.
        mu_: Expect mean value of the latent distribution.
        """
        ce = F.binary_cross_entropy(recon, src, reduction="sum")
        kl = 0.5 * torch.sum((mu - mu_).pow(2) + logsig.exp() - logsig - 1)
        return ce + kl
