from typing import List, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn

from ssl_bm_modality.ssl_methods.projection_heads import NonLinearProjection


class VICReg(LightningModule):
    def __init__(
            self,
            encoder: Union[LightningModule, nn.Module],
            ssl_batch_size: int = 128,
            sim_coeff: int = 25,
            std_coeff: int = 25,
            cov_coeff: int = 1,
            optimizer_name: str = 'adam',
            lr: float = 0.005,
            projection_hidden: List = [256],
            projection_out: int = 128,
            **kwargs
    ):

        """ Implementation of VicReg adapted from https://github.com/facebookresearch/vicreg/

        Parameters
        ----------
        encoder :
            encoder to train
        ssl_batch_size : int, optional
            batch size for ssl pre-training, by default 128
        sim_coeff : float, optional
            , by default 25
        std_coeff : float, optional
            , by default 25
        cov_coeff : float, optional
            , by default 1
        optimizer_name : str, optional
            optimizer, by default 'adam'
        lr : float, optional
            learning rate, by default 0.005
        """
        super().__init__()

        self.encoder = encoder

        self.projection = NonLinearProjection(
            self.encoder.out_size,
            projection_out,
            projection_hidden
        )

        self.optimizer_name_ssl = optimizer_name
        self.lr = lr

        self.embedding_size = self.encoder.out_size
        self.ssl_batch_size = ssl_batch_size

        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

        self.save_hyperparameters(ignore=["encoder"])

    def on_fit_start(self):
        self.encoder.to(self.device)

    def _process_batch(self, batch):
        aug1, aug2 = batch[0].float(), batch[-1].float()
        return aug1, aug2

    def _compute_vicreg_loss(self, x, y, partition):
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.ssl_batch_size - 1)
        cov_y = (y.T @ y) / (self.ssl_batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.embedding_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.embedding_size)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        self.log(f"repr_{partition}_loss", repr_loss)
        self.log(f"std_{partition}_loss", std_loss)
        self.log(f"cov_{partition}_loss", cov_loss)
        self.log(f"{partition}_loss", loss)

        return loss

    def forward(self, x, y):
        x = self.projection(nn.Flatten()(self.encoder(x)))
        y = self.projection(nn.Flatten()(self.encoder(y)))
        return x, y

    def training_step(self, batch, batch_idx):
        aug, x = self._process_batch(batch)
        x, y = self(x, aug)
        return self._compute_vicreg_loss(x, y, 'train')

    def validation_step(self, batch, batch_idx):
        aug, x = self._process_batch(batch)
        x, y = self(x, aug)
        return self._compute_vicreg_loss(x, y, 'val')

    def test_step(self, batch, batch_idx):
        aug, x = self._process_batch(batch)
        x, y = self(x, aug)
        return self._compute_vicreg_loss(x, y, 'test')

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_name_ssl.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=7)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'train_loss'
                }
            }


def off_diagonal(x: torch.Tensor):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
