from collections import OrderedDict
from typing import List, Optional

import torch.nn as nn
from pytorch_lightning import LightningModule


class NonLinearProjection(LightningModule):
    """ Projection head: MLP with non-linear (ReLU) transformations and batch normalization.
    """
    def __init__(
            self,
            in_size: int,
            out_size: int,
            hidden: List = [256, 128],
    ):
        super().__init__()
        self.name = 'NonLinearProjection'

        self.hidden_blocks = OrderedDict()
        self._all_dims = [in_size] + hidden

        for i in range(1, len(self._all_dims)):
            self.hidden_blocks.update({f'mlp_block{i}': nn.Sequential(OrderedDict([
                (f'linear{i}', nn.Linear(self._all_dims[i - 1], self._all_dims[i])),
                (f'relu{i}', nn.ReLU(inplace=True)),
                ((f'batch_norm{i}', nn.BatchNorm1d(self._all_dims[i])))
            ]))})

        self.hidden_blocks = nn.Sequential(self.hidden_blocks)

        self.output = nn.Linear(hidden[-1], out_size)

        self.save_hyperparameters()

    def forward(self, x):
        for hidden_block in self.hidden_blocks:
            x = hidden_block(x)
        x = self.output(x)
        return x
