from typing import List

import torch
import torchaudio
import torch.nn as nn
from pytorch_lightning import LightningModule

from .cnn1d import CNN1D


class Wav2Vec2Wrapper(LightningModule):
    """ Wraps Wav2Vec 2.0 model from torch audio into the Lightning Module
    """
    def __init__(
            self,
            w2v2_type: str = 'base',
            freeze: bool = True
    ):
        super().__init__()
        # TODO: add other configurations of wav2vec2.0 and integrate with Wav2VecCNN
        if w2v2_type == "base":
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
        elif w2v2_type == "large":
            bundle = torchaudio.pipelines.WAV2VEC2_LARGE
        else:
            raise ValueError("wrong type of W2V2 model provided")
        self.w2v2 = bundle.get_model()
        if freeze:
            for param in self.w2v2.parameters():
                param.requires_grad = False

    def forward(self, x, lengths=None):
        internal_features, valid_lengths = self.w2v2.extract_features(x.squeeze(axis=1), lengths=lengths)
        output_local_encoder = self.w2v2.encoder.feature_projection.forward(
            self.w2v2.feature_extractor(
                x.squeeze(axis=1), length=lengths
            )[0]
        )
        internal_features.append(output_local_encoder)
        return internal_features, valid_lengths


class Wav2Vec2CNN(LightningModule):
    """ CNN applied on top of the wav2vec 2.0 features and weighted average
        applied to different transformer layers from wav2vec.
        Adapted from: https://arxiv.org/pdf/2104.03502.pdf
    """
    def __init__(
            self,
            length_samples: int = 10,
            sample_rate: int = 16000,
            w2v2_type: str = 'base',
            freeze: bool = True,
            out_channels: List = [128, 128],
            kernel_sizes: List = [1, 1],
            pretrained=None,
    ):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Wrapper(w2v2_type=w2v2_type, freeze=freeze)

        if w2v2_type == "base":
            self.weighted_average = nn.Parameter(torch.ones(13))
            in_channels = 768
        elif w2v2_type == "large":
            self.weighted_average = nn.Parameter(torch.ones(25))
            in_channels = 1024

        self.cnn = CNN1D(
            in_channels=in_channels,
            len_seq=499,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            padding=0,
            stride=1,
            pool_size=1,
            pool_padding=0
        )

        self.out_size = self.compute_out_size(length=length_samples, sample_rate=sample_rate)

        self.save_hyperparameters()

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

    def compute_out_size(self, length, sample_rate):
        dummy_input = torch.rand((1, 1, length * sample_rate)).to(self.device)
        out = self.forward(dummy_input)

        return torch.numel(out)

    def forward(self, x, lengths=None):
        # pass data through wav2vec2
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

        # apply cnn layers
        outs = self.cnn(masked_embedding)

        mask = torch.ones(outs.shape).to(self.device)
        if valid_lengths is not None:
            for batch, valid in enumerate(valid_lengths):
                mask[batch, :, valid:] = 0

        # return global average
        return masked_mean(outs, mask, dim=2)


def masked_mean(tensor, mask, dim):
    """from: https://discuss.pytorch.org/t/equivalent-of-numpy-ma-array-to-mask-values-in-pytorch/53354/5
    """
    masked = torch.mul(tensor, mask)  # Apply the mask using an element-wise multiply
    return masked.sum(dim=dim) / mask.sum(dim=dim)
