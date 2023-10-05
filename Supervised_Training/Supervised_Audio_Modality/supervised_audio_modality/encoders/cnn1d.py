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
        1D-Convolutional Network with three layers
        Args:
            in_channels: number of channels in the input data
            len_seq: lenght of the input sequence
            out_channels: list containing the number of channels in the convolutional layers
            kernel_sizes: list containing the sizes of the convolutional kernels
            strid: size of the stride
            padding: unused, just to compute the out size
            pool_size: size of the maxpooling 
            p_drop: dropout value
            pretrained: path to pretrained model
        """
        super(CNN1D, self).__init__()
        self.len_seq = len_seq
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.name = 'cnn1d'
        self.num_layers = len(out_channels)
            
        self.conv_block1 = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=self.kernel_sizes[0], stride=stride, padding=padding)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool1d(kernel_size=pool_size, stride=None))
        ]))

        self.conv_block2 = nn.Sequential(OrderedDict([
            ('conv2',nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=self.kernel_sizes[1], stride=stride, padding=padding)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool1d(kernel_size=pool_size, stride=None))
        ]))

        self.conv_block3 = nn.Sequential(OrderedDict([
            ('conv3',nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=self.kernel_sizes[2], stride=stride, padding=padding)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool1d(kernel_size=pool_size, stride=None))
        ]))

        self.dropout = nn.Dropout(p=p_drop)

        self.out_size = self._compute_out_size(len_seq, padding, self.kernel_sizes, stride, 3, out_channels[-1], pool_size, pool_padding)

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
        x = self.conv_block1(x)
        x = self.dropout(x)

        x = self.conv_block2(x)
        x = self.dropout(x)

        x = self.conv_block3(x)
        x = self.dropout(x)
        return x


class CNN1D1L(LightningModule):
    def __init__(self, 
                in_channels, 
                len_seq, 
                out_channels=[32], 
                kernel_sizes=[7], 
                stride=1, 
                padding=1,
                pool_padding=0, 
                pool_size=2, 
                p_drop=0.2,
                **kwargs):
        """
        1D-Convolutional Network with three layers
        Args:
            in_channels: number of channels in the input data
            len_seq: lenght of the input sequence
            out_channels: the number of channels in the convolutional layer
            kernel_sizes: the sizes of the convolutional kernel
            strid: size of the stride
            padding: unused, just to compute the out size
            pool_size: size of the maxpooling 
            p_drop: dropout value
            pretrained: path to pretrained model
        """
        super(CNN1D1L, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        self.name = 'cnn1d'
        self.num_layers = len(out_channels)
            
        self.conv_block1 = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=self.kernel_sizes[0], stride=stride, padding=padding)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ('pool1', nn.MaxPool1d(kernel_size=pool_size, stride=None))
        ]))

        self.dropout = nn.Dropout(p=p_drop)

        self.out_size = self._compute_out_size(len_seq, padding, self.kernel_sizes, stride, self.num_layers, out_channels[-1], pool_size, pool_padding)

    @staticmethod
    def _compute_out_size(sample_length, padding, kernel_sizes, stride, num_layers, num_channels, pool_size, pool_padding):
        conv_out_size = sample_length
        for i in range(num_layers):
            conv_out_size = int((conv_out_size + 2 * padding - (kernel_sizes[i] - 1) - 1) / stride + 1)
            conv_out_size = int((conv_out_size + 2 * pool_padding - (pool_size - 1) - 1) / pool_size + 1)
        return int(num_channels * conv_out_size)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.dropout(x)
        return x