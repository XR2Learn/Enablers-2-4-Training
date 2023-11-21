import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ssl_features_extraction_audio_modality.utils.init_utils import init_encoder, init_transforms
from ssl_features_extraction_audio_modality.generate_and_save import generate_and_save


class InitTransformsTestCase(unittest.TestCase):
    def setUp(self):
        sys.path.append("./ssl_features_extraction_audio_modality")
        self.dataset_size = 10
        # CNN
        in_channels = 10
        len_seq = 500
        out_channels = [16, 32, 32, 64]
        kernel_sizes = [3, 3, 3, 5]
        stride = 2
        self.cnn_config = {
            "from_module": "ssl_features_extraction_audio_modality.encoders.cnn1d",
            "class_name": "CNN1D",
            "kwargs": {
                "in_channels": in_channels,
                "len_seq": len_seq,
                "out_channels": out_channels,
                "kernel_sizes": kernel_sizes,
                "stride": stride
            }
        }
        self.cnn_input_shape_dataset = (
            self.dataset_size,
            self.cnn_config["kwargs"]["len_seq"],
            self.cnn_config["kwargs"]["in_channels"],
        )

        # W2V2 + CNN
        len_seq = 10
        sample_rate = 16000
        out_channels = [128, 128]

        self.w2v2_base_config = {
            "from_module": "ssl_features_extraction_audio_modality.encoders.w2v",
            "class_name": "Wav2Vec2CNN",
            "kwargs": {
                "length_samples": len_seq,
                "sample_rate": sample_rate,
                "w2v2_type": "base",
                "freeze": "true",
                "out_channels": out_channels
            }
        }
        self.w2v2_base_input_shape_dataset = (
            self.dataset_size,
            self.w2v2_base_config["kwargs"]["length_samples"] * self.w2v2_base_config["kwargs"]["sample_rate"],
        )

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
        self.train_transforms, self.test_transforms = init_transforms(cfg_transforms["transforms"])
        self.transforms = {
            "train": self.train_transforms,
            "val": self.train_transforms,
            "test": self.test_transforms
        }

        self.encoders_configs = [
            {
             "config": self.cnn_config,
             "input_shape": self.cnn_input_shape_dataset
            },
            {
             "config": self.w2v2_base_config,
             "input_shape": self.w2v2_base_input_shape_dataset
            },
        ]

        self.outputs_path = tempfile.mkdtemp()
        self.features_path = os.path.join(self.outputs_path, "features")
        os.makedirs(self.features_path, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.outputs_path)

    def test_generate_features(self):
        for i, encoder_config in enumerate(self.encoders_configs):
            encoder = init_encoder(encoder_config["config"])
            dataset = torch.rand(*encoder_config["input_shape"])
            with self.subTest(
                f"""
                    Encoder: {encoder.__class__.__name__}
                    Use train.csv, val.csv, test.csv
                """
            ):
                self._common_test_generate_features(encoder, dataset, generate_csv=True)
            with self.subTest(
                f"""
                    Encoder: {encoder.__class__.__name__}
                    Use folder with data only
                """
            ):
                self._common_test_generate_features(encoder, dataset, generate_csv=False)

    def _common_test_generate_features(self, encoder, dataset, generate_csv):
        data_path = self._create_dataset(dataset, generate_csv)
        if generate_csv:
            data_paths = ["train.csv", "val.csv", "test.csv"]
        else:
            data_paths = [self.features_path]
        for data_path in data_paths:
            generate_and_save(
                encoder=encoder,
                data_path=data_path,
                outputs_folder=self.outputs_path,
                input_type="features",
                output_type="SSL_features",
                transforms=self.transforms
            )
        generated_feature_files = os.listdir(os.path.join(self.outputs_path, "SSL_features"))
        self.assertEqual(
            len(generated_feature_files),
            self.dataset_size,
            "Mismatch between dataset size and number of generated files with features"
        )
        for i in range(self.dataset_size):
            ssl_feature = np.load(
                os.path.join(
                    self.outputs_path,
                    "SSL_features",
                    f"instance_{i}.npy"
                    )
            )
            self.assertEqual(
                encoder.out_size,
                ssl_feature.size,
                "Mismatch between the size of generated features and encoder out size"
            )

    def _create_dataset(self, dataset, generate_csv):
        df_list = []
        labels = np.empty((self.dataset_size,))
        splits = np.random.choice(["train", "val", "test"], size=(self.dataset_size,), p=[0.6, 0.2, 0.2])

        for i, instance in enumerate(dataset):
            file_path = os.path.join(self.features_path, f"instance_{i}.npy")
            np.save(file_path, instance)
            df_list.append((
                file_path,
                labels[i],
                splits[i]
            ))

        if generate_csv:
            df = pd.DataFrame(df_list)
            df.columns = ["files", "labels", "split"]

            df[df["split"] == "train"][["files", "labels"]].to_csv(os.path.join(self.outputs_path, "train.csv"))
            df[df["split"] == "val"][["files", "labels"]].to_csv(os.path.join(self.outputs_path, "val.csv"))
            df[df["split"] == "test"][["files", "labels"]].to_csv(os.path.join(self.outputs_path, "test.csv"))
