from typing import List, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn

from ssl_audio_modality.ssl_methods.projection_heads import NonLinearProjection


class SimCLR(LightningModule):
    def __init__(
            self,
            encoder: Union[LightningModule, nn.Module],
            ssl_batch_size: int = 128,
            temperature: float = 0.05,
            n_views: int = 2,
            optimizer_name: str = 'adam',
            lr: float = 0.001,
            projection_hidden: List = [256],
            projection_out: int = 128,
            **kwargs
    ):
        """ Implementation of SimCLR with Pytorch Lightning. Input encoder is supposed to be provided.

        Parameters
        ----------
        encoder : Union[LightningModule, nn.Module]
            encoder to train
        ssl_batch_size : int, optional
            batch size for ssl pre-training, by default 128
        temperature : float, optional
            temperature hyperparameter value, by default 0.05
        n_views : int, optional
            number of views, by default 2
        optimizer_name : str, optional
            optimizer, by default 'adam'
        lr : float, optional
            learning rate, by default 0.001
        projection_hidden : List
            hidden dimensions for projection MLP
        projection_out : int
            output dimension for projection MLP
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

        self.loss = NTXent(ssl_batch_size, n_views, temperature)

        self.save_hyperparameters(ignore=["encoder"])

    def _prepare_batch(self, batch):
        # expects that the augmented views are returned as the first and the last value in a tuple
        # other values may contain subjects information, label (if available), etc.

        aug1, aug2 = batch[0], batch[-1]
        batch = torch.cat([aug1, aug2], dim=0)
        batch = batch.float()

        return batch

    def forward(self, x):
        # works with one modality group only (one modality or stacked inputs from different modalities)
        x = self.encoder(x)
        x = nn.Flatten()(x)
        x = self.projection(x)
        return x

    def training_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch)
        out = self(batch)
        loss, pos, neg = self.loss(out)
        self.log('ssl_train_loss', loss)
        self.log("avg_positive_sim", pos)
        self.log("avg_neg_sim", neg)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch)
        out = self(batch)
        loss, _, _ = self.loss(out)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch)
        out = self(batch)
        loss, _, _ = self.loss(out)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_name_ssl.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'ssl_train_loss'
                }
            }


class NTXent(LightningModule):
    """ Implements NTXent loss in contrastive learning
        Implementation adapted from https://github.com/sthalles/SimCLR/blob/master/simclr.py
    """
    def __init__(self, batch_size, n_views=2, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits, labels, pos, neg = self.get_infoNCE_logits_labels(x, self.batch_size, self.n_views, self.temperature)
        return self.criterion(logits, labels), pos, neg

    def get_infoNCE_logits_labels(self, features, batch_size, n_views=2, temperature=0.1):
        # creates a vector with labels [0, 1, 2, 0, 1, 2]
        labels = torch.cat([torch.arange(int(features.shape[0]/2)) for i in range(n_views)], dim=0)

        # creates matrix where 1 is on the main diagonal and where indexes of the same intances match 
        # (e.g. [0, 4][1, 5] for batch_size=3 and n_views=2)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # computes similarity matrix by multiplication, shape: (batch_size * n_views, batch_size * n_views)
        similarity_matrix = get_cosine_sim_matrix(features)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)

        # mask out the main diagonal - output has one column less
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix_wo_diag = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix_wo_diag[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives
        negatives = similarity_matrix_wo_diag[~labels.bool()].view(similarity_matrix_wo_diag.shape[0], -1)

        # reshuffles values in each row so that positive similarity value for each row is in the first column
        logits = torch.cat([positives, negatives], dim=1)
        # labels is a zero vector because all positive logits are in the 0th column
        labels = torch.zeros(logits.shape[0])

        logits = logits / temperature

        return logits, labels.long().to(logits.device), positives.mean(), negatives.mean()


def get_cosine_sim_matrix(features):
    """ Computes pair-wise cosine similarity matrix for input features

    Parameters
    ----------
    features : torch.Tensor
        Input features

    Returns
    -------
    torch.Tensor
        Generated cosine similarity matrix
    """
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    return similarity_matrix
