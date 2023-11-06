"""
Module containing classes for augmentations for signals
"""

import numpy as np
import random
import scipy
import torch

# Augmentations based on:
# https://openreview.net/pdf?id=bSC_xo8VQ1b
# https://arxiv.org/pdf/2206.07656.pdf

class Permutation:
    def __init__(self, max_segments=5, min_segments=3):
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
        #check if given values are possible or not, else replace them.
        #assumes datapoints last
        max_segments = min(x.shape[-1],abs(self.max_segments))
        min_segments = max(1,self.min_segments)

        channels, signal = x.shape
        orig_steps = torch.arange(signal)
        num_segs = np.random.randint(min_segments, max_segments)
        ret = torch.zeros_like(x)

        if num_segs > 1:
            splits = torch.tensor_split(orig_steps, num_segs, dim=-1)
            permuted_splits = torch.cat([splits[i] for i in torch.randperm(len(splits))])

            # Permute each channel separately
            # needs to be replaced in the future by a version specifically for this
            # in case we want independent permutation per channel or not
            for c in range(channels):
                ret[c] = torch.index_select(x[c], 0, permuted_splits)
        else:
            ret = x

        return ret


class TimeWarping:
    def __init__(self, warp_factor=2, num_segments=4):
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
    def __init__(self, min_shift=1, max_shift=5):
        """
        Shifts data on the time axis.

        Parameters
        ----------
        max_shift : int
            Maximum shift to apply.
        """
        self.max_shift = max_shift
        self.min_shift = min_shift

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
        shift = int(self.max_shift + random.random() * self.max_shift)
        return torch.roll(x, direction * shift, dims=-1)