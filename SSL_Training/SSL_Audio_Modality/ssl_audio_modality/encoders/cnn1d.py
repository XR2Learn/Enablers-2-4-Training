import torch
import torch.nn as nn
from collections import OrderedDict

from pytorch_lightning import LightningModule


class CNN1D(LightningModule):
    def __init__(self, 
                in_channels, 
                len_seq, 
                out_channels=[32, 64, 128], 
                kernel_sizes=[7, 5, 3], 
                stride=1, 
                padding=1,
                pool_padding=0, 
                pool_size=2, 
                p_drop=0.2,
                pretrained = None,
                **kwargs):
        """
        1D-Convolutional Network with three layers.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input data.
        len_seq : int
            Length of the input sequence.
        out_channels : list of int
            List containing the number of channels in the convolutional layers.
        kernel_sizes : list of int
            List containing the sizes of the convolutional kernels.
        stride : int
            Size of the stride.
        padding : int
            Unused, just to compute the out size.
        pool_padding : int
            Padding for maxpooling.
        pool_size : int
            Size of the maxpooling.
        p_drop : float
            Dropout value.
        pretrained : str
            Path to pretrained model.
        """
        super(CNN1D, self).__init__()
        self.len_seq = len_seq
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.name = 'cnn1d'
        self.num_layers = len(out_channels)
        self.convolutional_blocks = OrderedDict()

        assert len(out_channels)==len(kernel_sizes), "out_channel and kernel_size list lengths should match"
        self.out_channels.insert(0,self.in_channels)

        for i in range(1,len(out_channels)):
            self.convolutional_blocks.update({f'conv_block{i}':nn.Sequential(OrderedDict([
                (f'conv{i}',nn.Conv1d(in_channels=out_channels[i-1], out_channels=out_channels[i], kernel_size=self.kernel_sizes[i-1], stride=stride, padding=padding)),
                (f'relu{i}', nn.ReLU(inplace=True)),
                (f'pool{i}', nn.MaxPool1d(kernel_size=pool_size, stride=None)),
                (f'dropout{i}',nn.Dropout(p=p_drop))
            ]))})
        
        self.convolutional_blocks = nn.Sequential(self.convolutional_blocks)

        self.out_size = self._compute_out_size(len_seq, padding, self.kernel_sizes, stride,len(self.out_channels)-1, out_channels[-1], pool_size, pool_padding)

        try:
            if pretrained is not None:
                self.load_state_dict(torch.load(pretrained.replace('\\','/')))
                print(f'succesfully loaded weights from {pretrained}')
            else:
                print("NO pretrained weights loaded")
        except:
            print(f'failed to loaded weights from {pretrained}, encoder initialised with random weights')

    @staticmethod
    def _compute_out_size(sample_length, padding, kernel_sizes, stride, num_layers, num_channels, pool_size, pool_padding):
        conv_out_size = sample_length
        for i in range(num_layers):
            conv_out_size = int((conv_out_size + 2 * padding - (kernel_sizes[i] - 1) - 1) / stride + 1)
            conv_out_size = int((conv_out_size + 2 * pool_padding - (pool_size - 1) - 1) / pool_size + 1)
        return int(num_channels * conv_out_size)

    def forward(self, x):
        """
        Forward pass of the CNN1D.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying CNN layers.
        """
        for conv_block in self.convolutional_blocks:
            x = conv_block(x)

        return x