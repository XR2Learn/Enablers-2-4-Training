import torch
import torchaudio
import torch.nn as nn
from pytorch_lightning import LightningModule


class Wav2Vec2Wrapper(LightningModule):
    def __init__(self, w2v2_type='base', freeze=True):
        """
        Wraps Wav2Vec 2.0 model from torch audio into the Lightning Module.

        Parameters
        ----------
        w2v2_type : str, optional
            Type of Wav2Vec 2.0 model. Default is 'base'.
        freeze : bool, optional
            If True, freeze the parameters of the Wav2Vec 2.0 model. Default is True.
        """
        super().__init__()
        # TODO: add other configurations of wav2vec2.0 and integrate with Wav2VecCNN
        if w2v2_type == 'base':
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
        else:
            raise ValueError("wrong type of W2V2 model provided")
        self.w2v2 = bundle.get_model()
        if freeze:
            for param in self.w2v2.parameters():
                param.requires_grad = False  

    def forward(self, x, lengths=None):
        """
        Forward pass of the Wav2Vec2Wrapper.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        lengths : torch.Tensor, optional
            Tensor containing lengths of the input sequences. Default is None.

        Returns
        -------
        Tuple
            Tuple containing internal features and valid lengths.
        """
        internal_features, _ = self.w2v2.extract_features(x.squeeze(axis=1), lengths=lengths)
        output_contextual_encoder, valid_lengths = self.w2v2.forward(x.squeeze(axis=1), lengths=lengths)
        internal_features.append(output_contextual_encoder)
        return internal_features, valid_lengths


class Wav2Vec2CNN(LightningModule):
    def __init__(self, length_samples=10, sample_rate=16000, w2v2_type='base', freeze=True, out_channels=128, pretrained=None):
        """
        CNN applied on top of the Wav2Vec 2.0 features with weighted average applied to different transformer layers from Wav2Vec.
        Adapted from: https://arxiv.org/pdf/2104.03502.pdf

        Parameters
        ----------
        length_samples : int, optional
            Length of input samples. Default is 10.
        sample_rate : int, optional
            Sample rate of the input. Default is 16000.
        w2v2_type : str, optional
            Type of Wav2Vec 2.0 model. Default is 'base'.
        freeze : bool, optional
            If True, freeze the parameters of the Wav2Vec 2.0 model. Default is True.
        out_channels : int, optional
            Number of output channels. Default is 128.
        pretrained : str, optional
            Path to pretrained weights. Default is None.
        """
        super().__init__()

        self.wav2vec2 = Wav2Vec2Wrapper(w2v2_type=w2v2_type, freeze=freeze)

        # TODO: make the code more modular -- separate class for CNN
        # TODO: update CNN hyperparameters. We can make them arguments for better flexibility and matching with different types of Wav2Vec2 features: in_channels, weighted average blocks
        self.weighted_average = nn.Parameter(torch.ones(13))  # for learnable weights
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1)

        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.out_size = self.compute_out_size(length=length_samples, sample_rate=sample_rate)

        try:
            if pretrained is not None:
                self.load_state_dict(torch.load(pretrained.replace('\\', '/')))
                print(f'successfully loaded weights from {pretrained}')
            else:
                print("NO pretrained weights loaded")
        except:
            print(f'failed to load weights from {pretrained}, encoder initialized with random weights')

    def compute_out_size(self, length, sample_rate):
        """
        Compute the output size based on the input length and sample rate.

        Parameters
        ----------
        length : int
            Length of the input.
        sample_rate : int
            Sample rate of the input.

        Returns
        -------
        int
            Output size.
        """
        dummy_input = torch.rand((1, 1, length * sample_rate)).to(self.device)
        out = self.forward(dummy_input)

        return torch.numel(out)

    def forward(self, x, lengths=None):
        """
        Forward pass of the Wav2Vec2CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        lengths : torch.Tensor, optional
            Tensor containing lengths of the input sequences. Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor after applying CNN layers.
        """
        # pass data through Wav2Vec2
        w2v2_features, valid_lengths = self.wav2vec2(x, lengths=lengths)
        # process the features and apply weighted average
        embedding = torch.stack(w2v2_features, axis=1)
        embedding = embedding * self.weighted_average[None, :, None, None]
        embedding = torch.sum(embedding, 1) / torch.sum(self.weighted_average)
        # setting channels first
        embedding = torch.transpose(embedding, 1, 2)

        # zero-ing the invalid lengths
        mask = torch.ones(embedding.shape).to(self.device)
        if valid_lengths is not None:
            for batch, valid in enumerate(valid_lengths):
                mask[batch, :, valid:] = 0

        masked_embedding = embedding * mask

        # apply CNN layers
        outs = self.conv1(masked_embedding)
        outs = self.relu(outs)
        outs = self.dropout(outs)
        outs = self.conv2(outs)
        outs = self.relu(outs)
        outs = self.dropout(outs)

        mask = torch.ones(outs.shape).to(self.device)
        if valid_lengths is not None:
            for batch, valid in enumerate(valid_lengths):
                mask[batch, :, valid:] = 0

        # return global average
        return masked_mean(outs, mask, dim=2)



def masked_mean(tensor, mask, dim):
    #from: https://discuss.pytorch.org/t/equivalent-of-numpy-ma-array-to-mask-values-in-pytorch/53354/5
    masked = torch.mul(tensor, mask)  # Apply the mask using an element-wise multiply
    return masked.sum(dim=dim) / mask.sum(dim=dim)