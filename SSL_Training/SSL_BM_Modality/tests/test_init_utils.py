import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
from pytorch_lightning import Trainer

from ssl_bm_modality.encoders.cnn1d import CNN1D
# from ssl_bm_modality.ssl_methods.SimCLR import SimCLR
# from ssl_bm_modality.ssl_methods.VICReg import VICReg
from ssl_bm_modality.ssl_methods.projection_heads import NonLinearProjection
from ssl_bm_modality.utils.init_utils import (
    init_augmentations,
    init_encoder,
    init_transforms,
    setup_ssl_model
)


class InitEncodersTestCase(unittest.TestCase):
    def test_init_encoder_cnn(self):
        in_channels = 10
        len_seq = 500
        out_channels = [16, 32, 32, 64]
        kernel_sizes = [3, 3, 3, 5]
        stride = 2
        cfg_cnn = {
            "from_module": "ssl_bm_modality.encoders.cnn1d",
            "class_name": "CNN1D",
            "kwargs": {
                "in_channels": in_channels,
                "len_seq": len_seq,
                "out_channels": out_channels,
                "kernel_sizes": kernel_sizes,
                "stride": stride
            }
        }

        encoder = init_encoder(model_cfg=cfg_cnn)
        self.assertTrue(
            isinstance(
                encoder,
                CNN1D,
            ),
            "CNN: model class mismatch"
        )
        self.assertEqual(
            len(encoder.convolutional_blocks),
            len(cfg_cnn["kwargs"]["out_channels"]),
            "Conv blocks number of layers mismatch"
        )
        self.assertEqual(
            torch.nn.Flatten()(encoder(torch.randn(64, in_channels, len_seq))).shape,
            (64, encoder.out_size),
            "CNN forward pass: unexpected shape"
        )
        # Test if model is initialized from checkpoint correctly
        test_dir = tempfile.mkdtemp()
        model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

        trainer = Trainer(default_root_dir=test_dir)
        trainer.strategy.connect(encoder)
        trainer.save_checkpoint(model_path)

        encoder_from_checkpoint = init_encoder(model_cfg=cfg_cnn, ckpt_path=model_path)

        shared_input = torch.randn(64, in_channels, len_seq)
        encoder.eval()
        encoder_from_checkpoint.eval()

        self.assertTrue(
            torch.allclose(encoder(shared_input), encoder_from_checkpoint(shared_input)),
            "CNN: Incorrect model loaded from checkpoint (state_dict)"
        )

        shutil.rmtree(test_dir)


class InitTransformsTestCase(unittest.TestCase):
    def test_init_transforms(self):
        sys.path.append("./ssl_bm_modality/")
        cfg_transforms = {
            "transforms": [
                {
                    "class_name": "ToTensor",
                    "from_module": "general_transforms",
                    "transform_name": "to_tensor",
                    "in_test": True
                },
                {
                    "class_name": "Permute",
                    "from_module": "general_transforms",
                    "transform_name": "permutation",
                    "in_test": True,
                    "kwargs": {
                        "shape": [1, 0]
                    }
                },
                {
                    "class_name": "ToFloat",
                    "from_module": "general_transforms",
                    "transform_name": "to_float",
                    "in_test": True
                }
            ]
        }
        train_transforms, test_transforms = init_transforms(cfg_transforms["transforms"])

        self.assertEqual(len(train_transforms.transforms), 3, "Mismatch in number of train transforms")
        self.assertEqual(len(test_transforms.transforms), 3, "Mismatch in number of test transforms")

        rand_input = np.random.rand(10, 128)
        train_transformed = train_transforms(rand_input)
        test_transformed = test_transforms(rand_input)

        self.assertEqual(
            train_transformed.numel(),
            test_transformed.numel(),
            "Train and test transforms produce different numbers of elements"
        )
        self.assertEqual(
            train_transformed.numel(),
            rand_input.size,
            "Train transforms produce different numbers of elements compared to input"
        )


class TestInitAugTestCase(unittest.TestCase):
    def test_init_aug(self):
        augmentations_cfg = {
            "augmentations": {
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
        }

        augmentations = init_augmentations(aug_dict=augmentations_cfg["augmentations"])

        rand_input = torch.rand(128, 10)
        augmented = augmentations(rand_input)

        self.assertEqual(
            augmented.shape,
            rand_input.shape,
            "Initialized augmentations changed data shape"
        )
        self.assertFalse(
            torch.allclose(
                rand_input,
                augmented
            ),
            "Initialized augmentations did not affect data values"
        )


class SetupSSLModelTestCase(unittest.TestCase):
    def test_setup_ssl_model_cnn_simclr(self):
        in_channels = 10
        len_seq = 500
        out_channels = [16, 32, 32, 64]
        kernel_sizes = [3, 3, 3, 5]
        stride = 2
        cfg_cnn = {
            "from_module": "ssl_bm_modality.encoders.cnn1d",
            "class_name": "CNN1D",
            "kwargs": {
                "in_channels": in_channels,
                "len_seq": len_seq,
                "out_channels": out_channels,
                "kernel_sizes": kernel_sizes,
                "stride": stride
            }
        }

        simclr_config = {
            "from_module": "ssl_methods.SimCLR",
            "ssl_framework": "SimCLR",
            "input_type": "test_input",
            "epochs": 20,
            "batch_size": 2,
            "kwargs": {
                "temperature": 0.1,
                "n_views": 2
            }
        }

        encoder = init_encoder(model_cfg=cfg_cnn)
        ssl_model = setup_ssl_model(encoder, simclr_config)

        self.assertTrue(isinstance(ssl_model.encoder, CNN1D))
        self.assertTrue(isinstance(ssl_model.projection, NonLinearProjection))
