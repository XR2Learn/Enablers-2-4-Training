from torchvision import transforms
from .base_augmentations import (
    GaussianNoise, Reverse, SignFlip, ChannelFlip,
    Scale, ZeroMasking, NoAugmentation
    )
from .signal_augmentations import Permutation, TimeShifting, TimeWarping


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

    Examples
    --------
    >>> config_example = {
    ...     'augmentation1': {'probability': 0.5, 'kwargs': {parameter1}},
    ...     'augmentation2': {'probability': 0.2, 'kwargs': {parameter2}}
    ... }
    >>> compose_random_augmentations(config_example)
    [RandomApply(augmentation1, p=0.5), RandomApply(augmentation2, p=0.2)]
    """
    # all augmentations assume channel first, and no batch, input shape:
    # [channels, dataseries]
    augmentations_dict = {
        "gaussian_noise": GaussianNoise,
        "Reverse": Reverse,
        "sign_flip": SignFlip,
        "channels_flip": ChannelFlip,
        "scale": Scale,
        "zero_masking": ZeroMasking,
        "no_augmentation": NoAugmentation,

        "permutation": Permutation,
        "time_shifting": TimeShifting,
        "time_warping": TimeWarping,
    }
    transforms_list = []
    for key in config_dict:
        if key in augmentations_dict.keys():
            if 'kwargs' in config_dict[key]:
                augmentation = augmentations_dict[key](
                    **config_dict[key]['kwargs']
                    )
            else:
                augmentation = augmentations_dict[key]()
            print(f"added {key} augmentation with probability {config_dict[key]['probability']}")
            transforms_list.append(
                transforms.RandomApply([augmentation],
                                       p=config_dict[key]['probability'])
            )
        else:
            print(f"{key} not found in augmentations dict")
            print(f"available options: {augmentations_dict.keys()}")
    return transforms_list