from typing import Any
import numpy as np
import random
import scipy
import torch
from torchvision import transforms


# Augmentations based on:
# https://openreview.net/pdf?id=bSC_xo8VQ1b
# https://arxiv.org/pdf/2206.07656.pdf

class GaussianNoise():
    def __init__(self,mean=0,std=0.2):
        """
        adds gaussion noise to the data
        Args:
            mean: mean value for the gaussion noise
            std: standard deviation for the gaussian noise
        """
        self.mean=mean
        self.std = std

    def __call__(self,x):
        x_noise = x + torch.empty(x.shape).normal_(mean=self.mean,std=self.std)
        return x_noise


class HorizontalFlip():
    def __init__(self):
        """
        flips data on horizontal axis
        Args:
            none
        """
        pass

    def __call__(self,x):
        return x.flip(-1)


class VerticalFlip():
    def __init__(self):
        """
        flips data on vertical axis
        Args:
            none
        """
        pass

    def __call__(self,x):
        return -1*x


class Scale():
    #change to random scale instead of fixed scale
    def __init__(self,max_scale=1.5):
        """
        scales all value by a random amount
        Args:
            max_scale: maximum value to scale with
        """
        self.max_scale=max_scale

    def __call__(self,x):
        scale = torch.rand(1)*self.max_scale
        return scale*x


class ZeroMasking():
    def __init__(self,mask_length=10):
        """
        adds a zero mask on a random position to the data
        Args:
            mask_length: length of the mask
        """
        self.mask_length=mask_length

    def __call__(self,x):
        x = torch.clone(x)
        max_idx = x.shape[-1]-self.mask_length
        start_idx = random.randint(0,max_idx)
        end_idx = start_idx+self.mask_length
        x[...,start_idx:end_idx]=0
        return x


class Permutation():
    def __init__(self, max_segments=5,min_segments=2):
        """
        randomly segments the data and permutes the segments
        Args:
            max_segments: maximum number of segments
            min_segments: minimum number of segments
        """
        self.max_segments = max_segments
        self.min_segments = min_segments

    def __call__(self, x):
        channels, signal = x.shape
        orig_steps = torch.arange(signal)
        num_segs = np.random.randint(self.min_segments, self.max_segments)
        ret = torch.zeros_like(x)

        if num_segs > 1:
            splits = torch.tensor_split(orig_steps, num_segs,dim=-1)
            permuted_splits = torch.cat([splits[i] for i in torch.randperm(len(splits))])

            # Permute each channel separately
            for c in range(channels):
                ret[c] = torch.index_select(x[c], 0, permuted_splits)
        else:
            ret = x

        return ret


class Rotation():
    def __init__(self):
        """
        rotates the data
        Args:
            none
        """
        pass

    def __call__(self, x):
        flip = np.random.choice([-1, 1], size=(x.shape))
        return torch.tensor(flip) * x


class TimeWarping:
    def __init__(self, warp_factor, num_segments):
        """
        creates segments and 'warps' some values on the time axis while 'squishing' others
        Args:
            warp_factor: factor of warping
            num_segments: number of segments to make (only half of them are warped)
        """
        self.warp_factor = warp_factor
        self.num_segments = num_segments

    def __call__(self, x):
        channels, signal_length = x.shape
        segment_length = signal_length // self.num_segments

        # Split the signal into segments
        segments = torch.split(x, segment_length, dim=-1)

        # Randomly select half of the segments to warp
        num_segments_to_warp = self.num_segments // 2
        segments_to_warp_indices = torch.randperm(self.num_segments)[:num_segments_to_warp]

        # Apply time warping to the selected segments
        warped_segments = []
        for i, segment in enumerate(segments):
            if i in segments_to_warp_indices:
                # Apply speed up (stretch) to the segment
                warp_factor = self.warp_factor
                warped_segment = self.time_stretch(segment, warp_factor)
            else:
                # Apply slow down (squeeze) to the segment
                warp_factor = 1.0 / self.warp_factor
                warped_segment = self.time_stretch(segment, warp_factor)
            warped_segments.append(warped_segment)

        # Concatenate the warped segments
        warped_signal = torch.cat(warped_segments, dim=-1)
         
        out_signal = torch.zeros((channels,signal_length))
        for channel in range(channels):
            out_signal[channel] = torch.tensor(scipy.signal.resample(warped_signal[channel,:],signal_length))
        return out_signal

    def time_stretch(self, signal, warp_factor):
        num_channels, signal_length = signal.shape
        new_length = int(signal_length * warp_factor)
        new_signal = torch.zeros((num_channels,new_length))

        for channel in range(num_channels):
            new_signal[channel] = torch.tensor(scipy.signal.resample(signal[channel,:],new_length))

        return new_signal


class TimeShifting():
    def __init__(self,max_shift):
        """
        shifts data on time axis
        Args:
            max_shift: maximum shift to apply
        """
        self.max_shift = max_shift

    def __call__(self, x):
        direction  = np.random.choice([-1, 1])
        shift = int(random.random()*self.max_shift)
        return torch.roll(x,direction*shift,-1)


class no_augmentation():
    def __init__(self):
        """
        no augmentation applied
        Args:
            none
        """
        pass
    def __call__(self, x):
        return x 


augmentations_dict = {
    'gaussian_noise': GaussianNoise,
    'scale': Scale,
    'rotation': Rotation,
    'permutation': Permutation,
    'time_shifting': TimeShifting,
    "time_warping": TimeWarping,
    "zero_masking": ZeroMasking,
    "vertical_flip": VerticalFlip,
    "horizontal_flip": HorizontalFlip,
    "no_augmentation": no_augmentation
}


def compose_random_augmentations(config_dict):
    """
    composes the augmentations to apply
    Args:
        config_dict: dictionary containing the augmentation to add and their parameters
    """
    transforms_list = []
    for key in config_dict:
        if key in augmentations_dict.keys():
            if 'parameters' in config_dict[key]:
                augmentation = augmentations_dict[key](**config_dict[key]['kwargs'])
            else:
                augmentation = augmentations_dict[key]()
            print(f"added {key} augmentation with probability {config_dict[key]['probability']}")
            transforms_list.append(transforms.RandomApply([augmentation], p=config_dict[key]['probability']))
        else:
            print(f"{key} not found in augmentations dict")
            print(f"avalaible options: {augmentations_dict.keys()}")
    return transforms_list