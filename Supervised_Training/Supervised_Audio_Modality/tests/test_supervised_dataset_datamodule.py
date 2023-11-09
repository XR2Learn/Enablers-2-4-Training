import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from supervised_audio_modality.supervised_dataset import SupervisedDataModule
from supervised_audio_modality.utils.init_utils import init_augmentations, init_transforms


class SupervisedDatasetDataModuleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data_path = tempfile.mkdtemp()
        # egemaps / ssl_features proxy
        self.dataset_size = 1000
        self.input_size_1d = (1, 88)
        self.features_1d = np.random.rand(self.dataset_size, *self.input_size_1d)
        self.features_1d_path = os.path.join(self.data_path, "features_1d")
        os.makedirs(self.features_1d_path, exist_ok=True)
        # mfcc proxy
        self.input_size_2d = (28, 100)
        self.features_2d = np.random.rand(self.dataset_size, *self.input_size_2d)
        self.features_2d_path = os.path.join(self.data_path, "features_2d")
        os.makedirs(self.features_2d_path, exist_ok=True)

        labels = np.random.randint(0, 8, (self.dataset_size,))
        self.label_mapping = {i: i for i in range(8)}

        splits = np.random.choice(["train", "val", "test"], size=(self.dataset_size,), p=[0.6, 0.2, 0.2])

        dataset = []
        for i in range(len(self.features_1d)):
            file_path_1d = os.path.join(self.features_1d_path, f"feature_1d_{i}.npy")
            file_path_2d = os.path.join(self.features_2d_path, f"feature_2d_{i}.npy")
            np.save(file_path_1d, self.features_1d[i])
            np.save(file_path_2d, self.features_2d[i])
            dataset.append((
                file_path_1d,
                file_path_2d,
                labels[i],
                splits[i]
            ))

        df = pd.DataFrame(dataset)
        df.columns = ["file_path_1d", "file_path_2d", "labels", "split"]

        train = df[df["split"] == "train"]
        val = df[df["split"] == "val"]
        test = df[df["split"] == "test"]

        (
            train[["file_path_1d", "labels"]]
            .rename(columns={'file_path_1d': 'files'})
            .to_csv(os.path.join(self.data_path, "train_1d.csv"))
        )
        (
            val[["file_path_1d", "labels"]]
            .rename(columns={'file_path_1d': 'files'})
            .to_csv(os.path.join(self.data_path, "val_1d.csv"))
        )
        (
            test[["file_path_1d", "labels"]]
            .rename(columns={'file_path_1d': 'files'})
            .to_csv(os.path.join(self.data_path, "test_1d.csv"))
        )
        (
            test[["file_path_1d", "labels"]]
            .rename(columns={'file_path_1d': 'files'})
            .to_csv(os.path.join(self.data_path, "test_1d.csv"))
        )
        (
            train[["file_path_2d", "labels"]]
            .rename(columns={'file_path_2d': 'files'})
            .to_csv(os.path.join(self.data_path, "train_2d.csv"))
        )
        (
            val[["file_path_2d", "labels"]]
            .rename(columns={'file_path_2d': 'files'})
            .to_csv(os.path.join(self.data_path, "val_2d.csv"))
        )
        (
            test[["file_path_2d", "labels"]]
            .rename(columns={'file_path_2d': 'files'})
            .to_csv(os.path.join(self.data_path, "test_2d.csv"))
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.data_path)

    def test_supervised_datamodules__no_transforms(self):
        self._common_test_supervised_datamodules__shapes(self.input_size_1d, suffix="_1d")
        self._common_test_supervised_datamodules__shapes(self.input_size_2d, suffix="_2d")

    def test_supervised_datamodules__transforms_augmentations(self):
        sys.path.append("./supervised_audio_modality/")
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

        # Transforms only
        self._common_test_supervised_datamodules__shapes(
            (self.input_size_1d[1], self.input_size_1d[0]),  # to account for permute operation
            suffix="_1d",
            train_transforms=train_transforms,
            test_transforms=test_transforms
        )
        self._common_test_supervised_datamodules__shapes(
            (self.input_size_2d[1], self.input_size_2d[0]),  # to account for permute operation
            suffix="_2d",
            train_transforms=train_transforms,
            test_transforms=test_transforms
        )

        augmentations_cfg = {
            "augmentations": {
                "gaussian_noise": {
                    "probability": 0.5,
                    "kwargs": {
                        "mean": 0,
                        "std": 0.2
                    }
                },
                "scale": {
                    "probability": 0.5,
                    "kwargs": {
                        "max_scale": 1.3
                    }
                }
            }
        }

        augmentations = init_augmentations(aug_dict=augmentations_cfg["augmentations"])

        # Transforms and augmentations
        self._common_test_supervised_datamodules__shapes(
            (self.input_size_1d[1], self.input_size_1d[0]),  # to account for permute operation
            suffix="_1d",
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            augmentations=augmentations
        )
        self._common_test_supervised_datamodules__shapes(
            (self.input_size_2d[1], self.input_size_2d[0]),  # to account for permute operation
            suffix="_2d",
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            augmentations=augmentations
        )

    def _common_test_supervised_datamodules__shapes(
            self,
            input_size,
            suffix,
            train_transforms=None,
            test_transforms=None,
            augmentations=None
    ):
        with self.subTest(
            f"""
            Datamodule shapes: features{suffix};
            Transforms: {train_transforms is not None};
            Augmentations: {augmentations is not None}
            """
        ):
            split_paths_1d = {
                "train": f"train{suffix}.csv",
                "val": f"val{suffix}.csv",
                "test": f"test{suffix}.csv"
            }
            datamodule = SupervisedDataModule(
                path=self.data_path,
                input_type=f"features{suffix}",
                batch_size=int(self.dataset_size // 10),
                split=split_paths_1d,
                label_mapping=self.label_mapping,
                train_transforms=train_transforms,
                test_transforms=test_transforms,
                augmentations=augmentations,
            )
            # Init dataloaders for splits and check if shapes of batches are consistent
            # with input size
            datamodule._init_dataloaders(stage="TrainerFn.FITTING")
            train_batch = next(iter(datamodule.train_dataloader()))
            self.assertEqual(
                train_batch[0].shape,
                (int(self.dataset_size // 10), *input_size),
                "Train Dataloader: generated data batch shape is inconsistent with the input data shape."
            )
            self.assertEqual(
                train_batch[1].shape,
                (int(self.dataset_size // 10),),
                "Train Dataloader: generated labels batch shape is inconsistent with the input labels shape."
            )

            val_batch = next(iter(datamodule.val_dataloader()))
            self.assertEqual(
                val_batch[0].shape,
                (int(self.dataset_size // 10), *input_size),
                "Val Dataloader: generated data batch shape is inconsistent with the input data shape."
            )
            self.assertEqual(
                val_batch[1].shape,
                (int(self.dataset_size // 10),),
                "Val Dataloader: generated labels batch shape is inconsistent with the input labels shape."
            )

            datamodule._init_dataloaders(stage="TrainerFn.TESTING")
            test_batch = next(iter(datamodule.test_dataloader()))
            self.assertEqual(
                test_batch[0].shape,
                (int(self.dataset_size // 10), *input_size),
                "Test Dataloader: generated data batch shape is inconsistent with the input data shape."
            )
            self.assertEqual(
                test_batch[1].shape,
                (int(self.dataset_size // 10),),
                "Test Dataloader: generated labels batch shape is inconsistent with the input labels shape."
            )
