"""
augmentations.py
====================================
Module containing all augmentations for self-supervised learning
"""

import numpy as np
import random
import scipy
import torch
from torchvision import transforms


# Augmentations based on:
# https://openreview.net/pdf?id=bSC_xo8VQ1b
# https://arxiv.org/pdf/2206.07656.pdf

class GaussianNoise:
    def __init__(self, mean=0, std=0.2):
        """ Adds Gaussian noise to the data.

        Parameters
        ----------
        mean : float, optional
            Mean value for the Gaussian noise. Default is 0.
        std : float, optional
            Standard deviation for the Gaussian noise. Default is 0.2.
        """
        self.mean = mean
        self.std = std

    def __call__(self, x):
        """ Apply Gaussian noise to the input data.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Noisy data after applying Gaussian noise.
        """
        x_noise = x + torch.normal(mean=self.mean, std=self.std, size=x.shape)
        return x_noise


class HorizontalFlip:
    def __init__(self):
        """
        Flips data on the horizontal axis.

        Parameters
        ----------
        None
        """
        pass

    def __call__(self, x):
        """
        Apply horizontal flip to the input data.

        Parameters
        ----------
        x : numpy.ndarray
            Input data.

        Returns
        -------
        numpy.ndarray
            Flipped data along the horizontal axis.
        """
        return np.flip(x, axis=-1)



class VerticalFlip:
    def __init__(self):
        """
        Flips data on the vertical axis.

        Parameters
        ----------
        None
        """
        pass

    def __call__(self, x):
        """
        Apply vertical flip to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Vertically flipped data.
        """
        return -1 * x


class Scale:
    def __init__(self, max_scale=1.5):
        """
        Scales all values by a random amount.

        Parameters
        ----------
        max_scale : float, optional
            Maximum value to scale with. Default is 1.5.
        """
        self.max_scale = max_scale

    def __call__(self, x):
        """
        Apply scaling to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Scaled data.
        """
        scale = torch.rand(1) * self.max_scale
        return scale * x


class ZeroMasking:
    def __init__(self, mask_length=10):
        """
        Adds a zero mask on a random position to the data.

        Parameters
        ----------
        mask_length : int, optional
            Length of the mask. Default is 10.
        """
        self.mask_length = mask_length

    def __call__(self, x):
        """
        Apply zero masking to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Data after applying zero masking.
        """
        x = torch.clone(x)
        max_idx = x.shape[-1] - self.mask_length
        start_idx = random.randint(0, max_idx)
        end_idx = start_idx + self.mask_length
        x[..., start_idx:end_idx] = 0
        return x


class Permutation:
    def __init__(self, max_segments=5, min_segments=2):
        """
        Randomly segments the data and permutes the segments.

        Parameters
        ----------
        max_segments : int, optional
            Maximum number of segments. Default is 5.
        min_segments : int, optional
            Minimum number of segments. Default is 2.
        """
        self.max_segments = max_segments
        self.min_segments = min_segments

    def __call__(self, x):
        """
        Apply permutation to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Permuted data.
        """
        channels, signal = x.shape
        orig_steps = torch.arange(signal)
        num_segs = np.random.randint(self.min_segments, self.max_segments)
        ret = torch.zeros_like(x)

        if num_segs > 1:
            splits = torch.tensor_split(orig_steps, num_segs, dim=-1)
            permuted_splits = torch.cat([splits[i] for i in torch.randperm(len(splits))])

            # Permute each channel separately
            for c in range(channels):
                ret[c] = torch.index_select(x[c], 0, permuted_splits)
        else:
            ret = x

        return ret

class Rotation:
    def __init__(self):
        """
        Rotates the data.

        Parameters
        ----------
        None
        """
        pass

    def __call__(self, x):
        """
        Apply rotation to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Rotated data.
        """
        flip = np.random.choice([-1, 1], size=x.shape)
        return torch.tensor(flip) * x


class TimeWarping:
    def __init__(self, warp_factor, num_segments):
        """
        Creates segments and 'warps' some values on the time axis while 'squishing' others.

        Parameters
        ----------
        warp_factor : float
            Factor of warping.
        num_segments : int
            Number of segments to make (only half of them are warped).
        """
        self.warp_factor = warp_factor
        self.num_segments = num_segments

    def __call__(self, x):
        """
        Apply time warping to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Data after applying time warping.
        """
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

        out_signal = torch.zeros((channels, signal_length))
        for channel in range(channels):
            out_signal[channel] = torch.tensor(scipy.signal.resample(warped_signal[channel, :], signal_length))
        return out_signal

    def time_stretch(self, signal, warp_factor):
        """
        Apply time stretching to a signal.

        Parameters
        ----------
        signal : torch.Tensor
            Input signal.
        warp_factor : float
            Warp factor.

        Returns
        -------
        torch.Tensor
            Stretched signal.
        """
        num_channels, signal_length = signal.shape
        new_length = int(signal_length * warp_factor)
        new_signal = torch.zeros((num_channels, new_length))

        for channel in range(num_channels):
            new_signal[channel] = torch.tensor(scipy.signal.resample(signal[channel, :], new_length))

        return new_signal


class TimeShifting:
    def __init__(self, max_shift):
        """
        Shifts data on the time axis.

        Parameters
        ----------
        max_shift : int
            Maximum shift to apply.
        """
        self.max_shift = max_shift

    def __call__(self, x):
        """
        Apply time shifting to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Data after applying time shifting.
        """
        direction = np.random.choice([-1, 1])
        shift = int(random.random() * self.max_shift)
        return torch.roll(x, direction * shift, dims=-1)

class NoAugmentation:
    def __init__(self):
        """
        No augmentation applied.

        Parameters
        ----------
        None
        """
        pass

    def __call__(self, x):
        """
        Returns the input unchanged.

        Parameters
        ----------
        x : any
            Input data.

        Returns
        -------
        any
            Unchanged input data.
        """
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
    "no_augmentation": NoAugmentation
}


def compose_random_augmentations(config_dict):
    """
    Composes augmentations to apply.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the augmentations to add and their parameters.

    Returns
    -------
    list
        List of RandomApply transformations to be applied in sequence.
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