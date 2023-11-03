import torch.nn as nn
from pytorch_lightning import LightningModule


class LinearClassifier(LightningModule):
    """ Simple linear layer (linear probe)
    """
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.name = 'LinearClassifier'
        self.classifier = nn.Linear(in_size, out_size)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.classifier(x)
        return x
