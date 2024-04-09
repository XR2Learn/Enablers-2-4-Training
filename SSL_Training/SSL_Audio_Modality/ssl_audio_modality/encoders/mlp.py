import torch
from torch import nn
from pytorch_lightning import LightningModule


class LinearEncoder(LightningModule):
    def __init__(self, in_size, out_size, pretrained=None):
        super().__init__()
        self.out_size = out_size

        self.layer1 = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()
        self.len_seq = in_size

        if pretrained is not None:
            loaded_checkpoint = torch.load(pretrained.replace('\\', '/'))
            # Pytorch lightning checkpoints store more values, and state dict needs to be accessed
            # using "state_dict" key, whereas default pytorch checkpoints store state_dict only
            if "state_dict" in loaded_checkpoint:
                loaded_checkpoint = loaded_checkpoint["state_dict"]
            self.load_state_dict(loaded_checkpoint)
            print(f'succesfully loaded weights from {pretrained}')
        else:
            print("NO pretrained weights loaded")

        self.save_hyperparameters()

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        return x
