import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    """ Simple linear layer (linear probe) 
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.name = 'LinearClassifier'
        self.classifier = nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.classifier(x)
        return x