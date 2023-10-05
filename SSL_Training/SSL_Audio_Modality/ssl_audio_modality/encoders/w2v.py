import torch
import torchaudio
import torch.nn as nn
from pytorch_lightning import LightningModule


class Wav2Vec2Wrapper(LightningModule):
    """ Wraps Wav2Vec 2.0 model from torch audio into the Lightning Module
    """    
    def __init__(self, w2v2_type='base', freeze=True):
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
        internal_features, _ = self.w2v2.extract_features(x.squeeze(axis=1),lengths=lengths)
        output_contextual_encoder, valid_lengths = self.w2v2.forward(x.squeeze(axis=1),lengths=lengths)
        internal_features.append(output_contextual_encoder)
        return internal_features, valid_lengths     


class Wav2Vec2CNN(LightningModule):
    """ CNN applied on top of the wav2vec 2.0 features and weighted average applied to different transformer layers from wav2vec.
        Adapted from: https://arxiv.org/pdf/2104.03502.pdf
    """    
    def __init__(self,length_samples = 10, sample_rate=16000, w2v2_type='base', freeze=True, out_channels=128,pretrained=None):
        super().__init__()
        
        self.wav2vec2 = Wav2Vec2Wrapper(w2v2_type=w2v2_type, freeze=freeze)
        
        # TODO: 
        # (1) make the code more modular -- separate class for CNN 
        # (2) update CNN hyperparameters. We can make them arguments for better flexibility and matching with different types of wav2vec2 features: in_channels, weighted average blocks
        self.weighted_average = nn.Parameter(torch.ones(13)) # for learnable weights
        self.conv1 = nn.Conv1d(in_channels=768,out_channels=out_channels,kernel_size=1,stride=1)
        self.conv2 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=1,stride=1)
        
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
        self.out_size = self.compute_out_size(length=length_samples,sample_rate=sample_rate)

        try:
            if pretrained is not None:
                self.load_state_dict(torch.load(pretrained.replace('\\','/')))
                print(f'succesfully loaded weights from {pretrained}')
            else:
                print("NO pretrained weights loaded")
        except:
            print(f'failed to loaded weights from {pretrained}, encoder initialised with random weights')

    def compute_out_size(self, length, sample_rate):
        dummy_input = torch.rand((1,1,length*sample_rate)).to(self.device)
        out = self.forward(dummy_input)

        return torch.numel(out)

    def forward(self, x, lengths=None):
        # pass data through wav2vec2
        w2v2_features, valid_lengths = self.wav2vec2(x, lengths=lengths) 
        # process the features and apply weighted average
        embedding = torch.stack(w2v2_features, axis=1)
        embedding = embedding*self.weighted_average[None,:,None,None]
        embedding = torch.sum(embedding,1)/torch.sum(self.weighted_average)
        # setting channels first
        embedding = torch.transpose(embedding,1,2)

        # zero-ing the invalid lengths
        mask = torch.ones(embedding.shape).to(self.device)
        if valid_lengths is not None:
            for batch,valid in enumerate(valid_lengths):
                mask[batch,:,valid:]=0
    
        masked_embedding = embedding*mask
        
        # apply cnn layers
        outs = self.conv1(masked_embedding)
        outs = self.relu(outs)
        outs = self.dropout(outs)
        outs = self.conv2(outs)
        outs = self.relu(outs)
        outs = self.dropout(outs)

        mask = torch.ones(outs.shape).to(self.device)
        if valid_lengths is not None:
            for batch,valid in enumerate(valid_lengths):
                mask[batch,:,valid:]=0

        # return global average
        return masked_mean(outs, mask, dim=2)


def masked_mean(tensor, mask, dim):
    #from: https://discuss.pytorch.org/t/equivalent-of-numpy-ma-array-to-mask-values-in-pytorch/53354/5
    masked = torch.mul(tensor, mask)  # Apply the mask using an element-wise multiply
    return masked.sum(dim=dim) / mask.sum(dim=dim)