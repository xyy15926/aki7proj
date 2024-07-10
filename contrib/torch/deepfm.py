#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: deepfm.py
#   Author: xyy15926
#   Created: 2024-06-27 11:17:08
#   Updated: 2024-07-10 10:40:30
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import List, Any, Tuple
from collections.abc import Sequence
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    from importlib import reload
    from suitbear import finer
    reload(finer)

from suitbear.finer import get_tmp_path

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
class ZipDataset(Dataset):
    def __init__(self, *datas: List[Sequence]):
        self.datas = datas
        self.length = len(datas[0])

    def __len__(self):
        return len(self.datas[0])

    def __getitem__(self, idx: int):
        return [ele[idx] for ele in self.datas]


# %%
class DeepFM(nn.Module):
    """DeepFM.

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
    def __init__(self, fea_sizes: List[int],
                 emb_size:int = 4,
                 hidden_sizes:List[int] = [32, 32],
                 fm_dropout_probs:List[float] = None,
                 deep_dropout_probs:List[float] = None,
                 deep_bn:bool = False,
                 writer:SummaryWriter = None):
        super().__init__()
        self.fea_num = len(fea_sizes)
        self.fea_sizes = fea_sizes
        self.emb_size = emb_size
        self.hidden_sizes = hidden_sizes
        self.dtype = torch.long
        self.device = DEVICE
        self.bias = nn.Parameter(torch.randn(1))
        self.loss_fn = nn.functional.binary_cross_entropy_with_logits

        if writer is not None:
            self.writer = SummaryWriter(writer)
        else:
            self.writer = None

        # Init FM part.
        # `fm_fst_embeddings` actes as the weights of the sparse features
        # directly added up linearly in FM.
        self.fm_fst_embeddings = nn.ModuleList(
            [nn.Embedding(fea_size, 1) for fea_size in self.fea_sizes])
        self.fm_sec_embeddings = nn.ModuleList(
            [nn.Embedding(fea_size, self.emb_size)
             for fea_size in self.fea_sizes])

        # Init FM dropout.
        if fm_dropout_probs is None:
            self.fm_dropouts = None
        else:
            self.fm_dropouts = nn.ModuleList(
                [nn.Dropout(prob) for prob in fm_dropout_probs])

        # Init Deep part with dropout.
        dnn_sizes = ([self.fea_num * self.emb_size]
                     + self.hidden_sizes)
        if deep_dropout_probs is None:
            deep_dropout_probs = [None] * len(dnn_sizes)

        layer_seqs = []
        if deep_dropout_probs[0] is not None:
            layer_seqs.append(nn.Dropout(deep_dropout_probs[0]))
        for in_sz, out_sz, prob in zip(dnn_sizes[:-1], dnn_sizes[1:],
                                       deep_dropout_probs[1:]):
            if deep_bn:
                layer_seqs.append(nn.BatchNorm1d(in_sz))
            layer_seqs.append(nn.Linear(in_sz, out_sz))
            layer_seqs.append(nn.ReLU())
            if prob is not None:
                layer_seqs.append(nn.Dropout(prob))
        self.dnn = nn.Sequential(*layer_seqs)

    def forward(self, Xi:torch.Tensor, Xv:torch.Tensor):
        """Forward pass.

        Params:
        --------------------------------
        Xi: Tensor[batch_size, fea_num, 1]
          Tensor recording the index of fields.
        Xv: Tesnor[batch_size, fea_num, 1]
          Tensor recording the value of fields, which will be all-straight ones
          for one-hot field.
        """
        batch_size = Xi.shape[0]
        fst_emb_arr = [torch.sum(emb(Xi[:, i, :]) * torch.unsqueeze(Xv[:, i, :], -1), 1)
                       for i, emb in enumerate(self.fm_fst_embeddings)]
        sec_emb_arr = [torch.sum(emb(Xi[:, i, :]) * torch.unsqueeze(Xv[:, i, :], -1), 1)
                       for i, emb in enumerate(self.fm_sec_embeddings)]
        fst_emb = torch.cat(fst_emb_arr, dim=1).reshape(batch_size, -1)
        sec_emb = torch.cat(sec_emb_arr, dim=1).reshape(batch_size, -1)

        # Apply FM dropout.
        if self.fm_dropouts is not None:
            fst_emb = self.fm_dropouts[0](fst_emb)
            sec_emb = self.fm_dropouts[1](sec_emb)

        fm_sec_out = 0.5 * (sum(sec_emb_arr) ** 2
                            - sum([embed ** 2 for embed in sec_emb_arr]))
        deep_out = self.dnn(sec_emb)

        total_sum = (fst_emb.sum(1) + fm_sec_out.sum(1) + self.bias
                     + torch.sum(deep_out, 1))

        return total_sum

    def fit(self, train_Xi:np.ndarray,
            train_Xv:np.ndarray,
            train_y:np.ndarray,
            batch_size:int = 64,
            epoch:int = 0):
        """Fit.

        Params:
        --------------------------------
        train_Xi: NDA[N, fea_num, 1] full of index of each field.
        train_Xv: NDA[N, fea_num, 1] full of value of each field.
        train_y: NDA[N] of the label.
        batch_size:
        epoch:
        """
        train_set = DataLoader(ZipDataset(train_Xi, train_Xv, train_y),
                               batch_size=64)
        train_size = len(train_Xi)
        self.train()
        optimizer = optim.Adam(self.parameters())

        logger.info(f"Epoch: {epoch}")
        total_loss = 0
        for batch, (Xi, Xv, y) in enumerate(train_set):
            pred = self(Xi, Xv)
            loss = self.loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if batch % 100 == 99:
                loss, current = total_loss / 100, (batch + 1) * len(Xi)
                logger.info(f"Loss: {loss:>7f} [{current: >5d} / "
                            f"{train_size:>5d}]")
                if self.writer is not None:
                    self.writer.add_scalar("train_loss", loss,
                                           epoch * len(train_set) + batch)
                total_loss = 0

    @torch.no_grad()
    def validate(self, test_Xi, test_Xv, test_y, epoch=0):
        """Validate.

        Params:
        --------------------------------
        test_Xi: NDA[N, fea_num, 1] full of index of each field.
        test_Xv: NDA[N, fea_num, 1] full of value of each field.
        test_y: NDA[N] of the label.
        epoch:
        """
        self.eval()
        pred = self(test_Xi, test_Xv)
        loss = self.loss_fn(pred, test_y)
        logger.info(f"Test Loss: {loss.item()}")
        self.writer.add_pr_curve("PR", test_y, nn.functional.sigmoid(pred),
                                 global_step=epoch)


# %%
def fit():
    root = get_tmp_path()
    fdir = root / "data" / "CTRPhony"
    embs = pd.read_csv(fdir / "category_emb.csv", header=None)
    fea_sizes = embs.groupby(0)[1].apply(lambda x: len(set(x)))

    # Read train data.
    train = pd.read_csv(fdir / "tiny_train_input.csv", header=None).values
    train_y = train[:, 0].astype(np.float32)
    train_Xi = train[:, 1:, np.newaxis]
    train_Xv = np.ones_like(train_Xi, dtype=np.float32)
    # train_set = DataLoader(list(zip(train_Xi, train_Xv, train_y)),
    #                        batch_size=64)
    # Xi, Xv, y = next(iter(train_set))
    # Xi = torch.unsqueeze(Xi, -1)
    # Xv = torch.unsqueeze(Xv, -1)
    # model_bn.forward(Xi, Xv)

    # Read test data.
    test = pd.read_csv(fdir / "tiny_test_input.csv", header=None).values
    test_y = torch.tensor(test[:, 0].astype(np.float32))
    test_Xi = torch.tensor(test[:, 1:, np.newaxis])
    test_Xv = torch.ones_like(test_Xi, dtype=torch.float32)

    # Train and write summary.
    writer = get_tmp_path() / "runs" / "deepfm_dropout"
    model = DeepFM(fea_sizes,
                   fm_dropout_probs=[0.5, 0.5],
                   deep_dropout_probs=[0.5, 0.5, 0.5],
                   writer=writer)
    for epoch in range(5, 10):
        model.fit(train_Xi, train_Xv, train_y, epoch=epoch)
        model.validate(test_Xi, test_Xv, test_y, epoch)
    torch.save(model.state_dict(), get_tmp_path() / "deepfm_dropout.pth")

    writer_bn = get_tmp_path() / "runs" / "deepfm_bn"
    model_bn = DeepFM(fea_sizes,
                      deep_bn=True,
                      writer=writer_bn)
    for epoch in range(10):
        model_bn.fit(train_Xi, train_Xv, train_y, epoch=epoch)
        model_bn.validate(test_Xi, test_Xv, test_y, epoch)
    torch.save(model_bn.state_dict(), get_tmp_path() / "deepfm_bn.pth")

    # model.load_state_dict(torch.load(get_tmp_path() / "deepfm.pth"))

