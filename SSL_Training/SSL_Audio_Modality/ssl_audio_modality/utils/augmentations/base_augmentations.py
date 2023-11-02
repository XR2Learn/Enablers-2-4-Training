"""
Module containing classes for general augmentations
"""

import numpy as np
import random
import torch


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
        x : torch.Tensor
            Input data.

        Returns
        -------
        x_noise : torch.Tensor
            Noisy data after applying Gaussian noise.
        """
        x_noise = x + np.random.normal(loc=self.mean, scale=self.std, size=x.shape)
        return x_noise
    
class Reverse:
    def __init__(self):
        """
        Reverses the data, left becomes right (and right becomes left)
        """
        pass

    def __call__(self, x):
        """
        Apply reverse to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Flipped data along the horizontal axis.
        """
        return torch.tensor(np.fliplr(x).copy())

class SignFlip:
    def __init__(self):
        """
        Changes the sign of all data
        """
        pass

    def __call__(self, x):
        """
        Apply sign flip to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            sign flipped data.
        """
        return -1 * x

class ChannelFlip:
    def __init__(self):
        """
        the top channel becomes the bottom channel (and the bottom channel becomes the top channel)
        """
    def __call__(self, x):
        """
        Apply ChannelFlip to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            channel flipped data.
        """
        return torch.tensor(np.flipud(x).copy())
    
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
        Adds a zero mask on a random position to the data across all channels.

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
    
class NoAugmentation:
    def __init__(self):
        """
        No augmentation applied.
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