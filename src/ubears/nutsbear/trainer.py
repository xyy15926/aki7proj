#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: trainer.py
#   Author: xyy15926
#   Created: 2025-07-08 09:02:04
#   Updated: 2025-07-08 20:15:09
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import List, Any, Tuple, Self
from collections.abc import Sequence
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.slp import finer
    reload(finer)

from ubears.flagbear.slp.finer import get_tmp_path, tmp_file

DEVICE = ("cuda"
          if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu")

logging.basicConfig(
    # format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    format="%(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class Trainer:
    """Module Trainer.

    Attrs:
    -------------------------
    mod: Module instance.
    loss_fn: Loss function.
    optimizer: Optimizer.
    mod_name: Name of current module instance.
      1. Determines the path to save the module.
      2. Determines `tb_writer` to record the training process.
    tb_writer: The path for TensorBoard.SummaryWriter to record the training
      process.
    epoch_idx: The index of current training epoch.
    batch_idx: The index of batch of current training epochs.
    """
    def __init__(
        self,
        mod: nn.Module,
        loss_fn: "nn.Function",
        optimizer: optim.Optimizer = None,
        mod_name: str = None,
    ):
        """Init trainer.

        Params:
        -------------------------
        mod: Module instance.
        loss_fn: Loss function.
        optimizer: Optimizer.
        mod_name: Name of current module instance.
        TensorBoard.SummaryWriter to record the training process.
        """
        self.mod = mod
        self.loss_fn = loss_fn
        self.optimizer = (optim.Adam(mod.parameters())
                          if optimizer is None else optimizer)
        if mod_name is not None:
            self.mod_name = f"{mod.__class__.__name__}_{mod_name}"
            absp = get_tmp_path() / self.mod_name
            self.tb_writer = SummaryWriter(absp)
        else:
            self.mod_name = None
            self.tb_writer = None
        self.epoch_idx = 0
        self.batch_idx = 0
        self.batch_n = None

    def log(self, **kwargs):
        """Log the training process.

        Params:
        -------------------------
        scalar: {NAME: VALUE} to be added by
          `TensorBoard.SummaryWriter.add_scalar`.
        """
        # Log loss with native logger.
        if "Loss" in kwargs:
            loss = kwargs["Loss"]
            if self.batch_n is None:
                logger.info(
                    f"Loss: {loss:>07f}\t"
                    f"[Epoch: {self.epoch_idx:>2d} - "
                    f"Batch: {self.batch_idx:>05d}]"
                )
            else:
                logger.info(
                    f"Loss: {loss:>07f}\t"
                    f"[Epoch: {self.epoch_idx:>2d} - "
                    f"Batch: {self.batch_idx:>05d}/"
                    f"{self.batch_n:>05d}]"
                )

        # Log all infomations with `TensorBoard.SummaryWriter`.
        if self.tb_writer is None:
            return
        if self.batch_n is None:
            all_bidx = self.batch_idx
        else:
            all_bidx = self.epoch_idx * self.batch_n + self.batch_idx
        with self.tb_writer as writer:
            for key, val in kwargs.items():
                if isinstance(val, float):
                    writer.add_scalar(key, val, all_bidx)

    def fit(self, dloader: DataLoader, epoch_n: int = 1, log_itvl: int = 10):
        """Fit the module.

        Params:
        -------------------------
        dloader: Train DataLoader.
        epoch_n: The number of epochs to train.
        log_itvl: The number of batch between the record-point.
        """
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        try:
            self.batch_n = len(dloader)
        except TypeError:
            self.batch_n = None

        for ce in range(self.epoch_idx, self.epoch_idx + epoch_n):
            itvl_loss = 0
            logger.info(f"Epoch: {ce}")
            for bidx, (*feats, label) in enumerate(dloader):
                # Forward and backward.
                ret = self.mod(*feats)
                loss = loss_fn(ret, label)
                loss.backward()

                # Optimize.
                optimizer.step()
                optimizer.zero_grad()

                # Log for every `log_itvl` batchs.
                itvl_loss += loss.item()
                if (bidx + 1) % log_itvl == 0:
                    loss = itvl_loss / log_itvl
                    self.log(Loss=loss)
                    itvl_loss = 0
                self.batch_idx += 1
            # Set the number the of batchs in the first epoch as the `batch_n`
            # if `batch_n` is not set properly earlier.
            if self.batch_n is None:
                self.batch_n = self.batch_idx
            self.batch_idx = 0
            self.epoch_idx += 1

    # TODO: save optim.
    def save(self):
        """Save module."""
        if self.mod_name is None:
            mod_name = self.mod.__class__.__name__
        else:
            mod_name = self.mod_name
        spath = tmp_file(f"{mod_name}_E{self.epoch_idx:>04d}")
        torch.save(self.mod.state_dict(), spath)
        logger.info(f"Save {mod_name} after training {self.epoch_idx} "
                    f"epochs at {spath}.")

    @staticmethod
    def load(
        mod: "nn.Module",
        mod_name: str = None,
        *args,
        **kwargs,
    ) -> nn.Module:
        """Load module."""
        if mod_name is None:
            mod_name = mod.__class__.__name__
        else:
            mod_name = f"{mod.__class__.__name__}_{mod_name}"
        epoch_ptn = r"E\d{4}"
        spath = tmp_file(rf"{mod_name}_{epoch_ptn}", None, incr=0)
        sdict = torch.load(spath)
        mod.load_state_dict(sdict)
        return mod
