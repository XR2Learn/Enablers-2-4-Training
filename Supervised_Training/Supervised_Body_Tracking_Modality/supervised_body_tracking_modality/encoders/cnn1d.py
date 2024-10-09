from collections import OrderedDict
from typing import Optional, List


import torch.nn as nn
from pytorch_lightning import LightningModule
import torch


class CNN1D(LightningModule):
    def __init__(
            self,
            in_channels: int,
            len_seq: int,
            out_channels: List = [32, 64, 128],
            kernel_sizes: List = [7, 5, 3],
            stride: int = 1,
            padding: int = 1,
            pool_padding: int = 0,
            pool_size: int = 2,
            p_drop: float = 0.2,
            pretrained: Optional[str] = None,
            **kwargs,
    ):
        """
        1D-Convolutional Network with three layers.

        Args:
            in_channels: Number of channels in the input data.
            len_seq: Length of the input sequence.
            out_channels: List containing the number of channels in the convolutional layers.
            kernel_sizes: List containing the sizes of the convolutional kernels.
            stride: Size of the stride.
            padding: Unused, just to compute the out size.
            pool_padding: Padding for maxpooling.
            pool_size: Size of the maxpooling.
            p_drop: Dropout value.
            pretrained: Path to pretrained model.
        """
        super(CNN1D, self).__init__()
        assert len(out_channels) == len(kernel_sizes), "out_channels and kernel_size list lengths should match"

        self.len_seq = len_seq
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.name = 'cnn1d'
        self.num_layers = len(out_channels)
        self.convolutional_blocks = OrderedDict()

        self._all_channels = [self.in_channels] + self.out_channels

        for i in range(1, len(self._all_channels)):
            self.convolutional_blocks.update({f'conv_block{i}': nn.Sequential(OrderedDict([
                (
                    f'conv{i}',
                    nn.Conv1d(
                        in_channels=self._all_channels[i - 1],
                        out_channels=self._all_channels[i],
                        kernel_size=self.kernel_sizes[i - 1],
                        stride=stride,
                        padding=padding,
                    )
                ),
                (f'relu{i}', nn.ReLU(inplace=True)),
                (f'pool{i}', nn.MaxPool1d(kernel_size=pool_size, stride=None)),
                (f'dropout{i}', nn.Dropout(p=p_drop))
            ]))})

        self.convolutional_blocks = nn.Sequential(self.convolutional_blocks)

        self.out_size = self._compute_out_size(
            len_seq,
            padding,
            self.kernel_sizes,
            stride,
            self.num_layers,
            out_channels[-1],
            pool_size,
            pool_padding
        )

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

    @staticmethod
    def _compute_out_size(
        sample_length: int,
        padding: int,
        kernel_sizes: int,
        stride: int,
        num_layers: int,
        num_channels: int,
        pool_size: int,
        pool_padding: int,
    ):
        assert len(kernel_sizes) == num_layers, "Number of layers should be equal to the number of kernels"
        conv_out_size = sample_length
        for i in range(num_layers):
            conv_out_size = int((conv_out_size + 2 * padding - (kernel_sizes[i] - 1) - 1) / stride + 1)
            conv_out_size = int((conv_out_size + 2 * pool_padding - (pool_size - 1) - 1) / pool_size + 1)
        return int(num_channels * conv_out_size)

    def forward(self, x):
        x = self.convolutional_blocks(x)
        return x
