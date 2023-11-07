import unittest
import torch
from torchvision import transforms
from supervised_audio_modality.utils.augmentations.base_augmentations import (
    GaussianNoise, Reverse, SignFlip, ChannelFlip, Scale, ZeroMasking,
    NoAugmentation
)
from supervised_audio_modality.utils.augmentations.signal_augmentations import (
    Permutation, TimeShifting, TimeWarping
)

from supervised_audio_modality.utils.augmentations import compose_random_augmentations


class SupervisedTestCase(unittest.TestCase):
    """ Implements a set of basic tests for augmentations

    """

    def setUp(self):

        self.GaussianNoise = GaussianNoise()
        self.Reverse = Reverse()
        self.SignFlip = SignFlip()
        # channel flip can fail if the data is 1D as the actual values are not
        # changed, so an if-else statement is put in the tests now
        self.ChannelFlip = ChannelFlip()
        self.Scale = Scale(max_scale=10)
        self.ZeroMasking = ZeroMasking()
        self.NoAugmentation = NoAugmentation()

        # permutation could fail in case min_segments can be equal to 0 or 1
        self.Pemutation = Permutation(min_segments=70, max_segments=137)
        self.Timeshifting = TimeShifting()
        self.TimeWarping = TimeWarping()

        self.original_data_1D = torch.rand(1, 134)
        self.original_data_2D = torch.rand(3, 142)
        self.base_augmentations = [
            self.GaussianNoise,
            self.Reverse,
            self.SignFlip,
            self.ChannelFlip,
            self.Scale,
            self.ZeroMasking,
        ]

        self.signal_augmentations = [
            self.Pemutation,
            self.Timeshifting,
            self.TimeWarping
        ]

    def test_base_augmentations_1d(self):
        """ test the base augmentations in case of 1D data

        """
        for i, aug in enumerate(self.base_augmentations):
            with self.subTest(f"{aug.__class__.__name__}", i=i):
                aug_data = aug(self.original_data_1D)
                self.assertEqual(aug_data.shape, self.original_data_1D.shape)
                # channel flip has no effect on the data itself in 1D.
                if not isinstance(aug, ChannelFlip):
                    self.assertFalse(
                        torch.equal(
                            aug_data,
                            self.original_data_1D
                        )
                    )
                self.assertEqual(type(aug_data), type(self.original_data_1D))
                self.assertEqual(aug_data.dtype, self.original_data_1D.dtype)

    def test_signal_augmentations_1d(self):
        """ test the signal augmentations in case of 1D data

        """
        for i, aug in enumerate(self.signal_augmentations):
            with self.subTest(f"{aug.__class__.__name__}", i=i):
                aug_data = aug(self.original_data_1D)
                self.assertEqual(aug_data.shape, self.original_data_1D.shape)
                self.assertFalse(
                    torch.equal(
                        aug_data,
                        self.original_data_1D
                    )
                )
                self.assertEqual(type(aug_data), type(self.original_data_1D))
                self.assertEqual(aug_data.dtype, self.original_data_1D.dtype)

    def test_base_augmentations_2d(self):
        """ test the base augmentations in case of 2D data

        """
        for i, aug in enumerate(self.base_augmentations):
            with self.subTest(f"{aug.__class__.__name__}", i=i):
                aug_data = aug(self.original_data_2D)
                self.assertEqual(aug_data.shape, self.original_data_2D.shape)
                self.assertFalse(
                    torch.equal(
                        aug_data,
                        self.original_data_2D
                    )
                )
                self.assertEqual(type(aug_data), type(self.original_data_2D))
                self.assertEqual(aug_data.dtype, self.original_data_2D.dtype)

    def test_signal_augmentations_2d(self):
        """ test the signal augmentations in case of 2D data

        """
        for i, aug in enumerate(self.signal_augmentations):
            with self.subTest(f"{aug.__class__.__name__}", i=i):
                aug_data = aug(self.original_data_2D)
                self.assertEqual(aug_data.shape, self.original_data_2D.shape)
                self.assertFalse(
                    torch.equal(
                        aug_data,
                        self.original_data_2D
                        )
                    )
                self.assertEqual(type(aug_data), type(self.original_data_2D))
                self.assertEqual(aug_data.dtype, self.original_data_1D.dtype)

    def test_compose_augmentation_full_probability(self):
        augmentations_cfg = {
                "gaussian_noise": {
                    "probability": 1,
                    "kwargs": {
                        "mean": 0,
                        "std": 0.2
                    }
                },
                "scale": {
                    "probability": 1,
                    "kwargs": {
                        "max_scale": 1.3
                    }
                }
            }
        aug = compose_random_augmentations(augmentations_cfg)
        aug = transforms.Compose(aug)

        aug_data = aug(self.original_data_2D)
        self.assertEqual(aug_data.shape, self.original_data_2D.shape)
        self.assertFalse(
            torch.equal(
                aug_data,
                self.original_data_2D
                )
            )
        self.assertEqual(type(aug_data), type(self.original_data_2D))
        self.assertEqual(aug_data.dtype, self.original_data_2D.dtype)

        aug_data = aug(self.original_data_1D)
        self.assertEqual(aug_data.shape, self.original_data_1D.shape)
        self.assertFalse(
            torch.equal(
                aug_data,
                self.original_data_1D
                )
            )
        self.assertEqual(type(aug_data), type(self.original_data_1D))
        self.assertEqual(aug_data.dtype, self.original_data_1D.dtype)

    def test_compose_augmentation_zero_probability(self):
        augmentations_cfg = {
                "gaussian_noise": {
                    "probability": 0,
                    "kwargs": {
                        "mean": 0,
                        "std": 0.2
                    }
                },
                "scale": {
                    "probability": 0,
                    "kwargs": {
                        "max_scale": 1.3
                    }
                }
            }
        aug = compose_random_augmentations(augmentations_cfg)
        aug = transforms.Compose(aug)

        aug_data = aug(self.original_data_2D)
        self.assertEqual(aug_data.shape, self.original_data_2D.shape)
        self.assertTrue(
            torch.equal(
                aug_data,
                self.original_data_2D
                )
            )
        self.assertEqual(type(aug_data), type(self.original_data_2D))
        self.assertEqual(aug_data.dtype, self.original_data_2D.dtype)

        aug_data = aug(self.original_data_1D)
        self.assertEqual(aug_data.shape, self.original_data_1D.shape)
        self.assertTrue(
            torch.equal(
                aug_data,
                self.original_data_1D
                )
            )
        self.assertEqual(type(aug_data), type(self.original_data_1D))
        self.assertEqual(aug_data.dtype, self.original_data_1D.dtype)
