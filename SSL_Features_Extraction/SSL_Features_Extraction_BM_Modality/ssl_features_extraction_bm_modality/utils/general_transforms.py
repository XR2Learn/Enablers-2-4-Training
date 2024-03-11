import scipy
import torch

# A set of general useful transforms


class Normalize:
    """ Normalization to zero mean and unit variance
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


class ToTensor:
    """ Conversion to PyTorch tensor
    """
    def __call__(self, x):
        return torch.tensor(x)


class Permute:
    """ Permutation of dimension to certain order (shape)
    """
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return x.permute(self.shape)


class ToFloat:
    """ Converting the input tensor to float values
    """
    def __call__(self, x):
        return x.float()


class Resample:
    """ Resampling input signals to a certain length (used for sequential data)
    """
    def __init__(self, samples):
        self.samples = samples

    def __call__(self, x):
        return scipy.signal.resample(x, self.samples)
