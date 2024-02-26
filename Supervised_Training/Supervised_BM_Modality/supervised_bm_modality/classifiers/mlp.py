from collections import OrderedDict
from typing import List, Optional

import torch.nn as nn
from pytorch_lightning import LightningModule


class MLPClassifier(LightningModule):
    """ MLP for classification
    """
    def __init__(
            self,
            in_size: int,
            out_size: int,
            hidden: List = [256, 128],
            p_dropout: Optional[float] = None,
    ):
        super(MLPClassifier, self).__init__()
        self.name = 'MLP'

        self.hidden_blocks = OrderedDict()
        self._all_dims = [in_size] + hidden

        for i in range(1, len(self._all_dims)):
            self.hidden_blocks.update({f'mlp_block{i}': nn.Sequential(OrderedDict([
                (f'linear{i}', nn.Linear(self._all_dims[i - 1], self._all_dims[i])),
                (f'relu{i}', nn.ReLU(inplace=True)),
                (f'dropout{i}', nn.Dropout(p=p_dropout) if p_dropout is not None else nn.Identity())
            ]))})

        self.hidden_blocks = nn.Sequential(self.hidden_blocks)

        self.output = nn.Linear(hidden[-1], out_size)

        self.save_hyperparameters()

    def forward(self, x):
        for hidden_block in self.hidden_blocks:
            x = hidden_block(x)
        x = self.output(x)
        return x
