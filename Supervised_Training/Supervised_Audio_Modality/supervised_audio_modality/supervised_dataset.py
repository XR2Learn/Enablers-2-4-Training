import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule


class SupervisedTorchDataset(Dataset):
    """ Torch Dataset class for supervised learning. The dataset is expected to be stored as a
        numpy array with metadata in csv with the following structure.

    Arguments:
        data_path: root folder of the dataset
        input_type: the type of input to load in this dataset
        split_path: path to the csv file containing the files and labels associated to the split
        transforms: transforms to apply to the input data
        augmentations: augmentations to apply to the input data
        n_views: number of views to create for the SSL
    """
    def __init__(
            self,
            data_path: str,
            input_type: str,
            split_path: str,
            label_mapping: Dict[Union[str, int], int],
            transforms: Optional[transforms.Compose] = None,
            augmentations: Optional[transforms.Compose] = None,       
    ):
        self.data_path = data_path
        self.input_type = input_type
        self.label_mapping = label_mapping
        # subjects and labels to retrieve
        self.split_path = split_path
        self.transforms = transforms
        self.augmentations = augmentations

        self._process_recordings()

    def _process_recordings(self):
        """ Parse recording from the file structure of a dataset.
        """
        print(os.path.join(self.data_path, self.split_path))
        meta_data = pd.read_csv(os.path.join(self.data_path, self.split_path), index_col=0)
        self.labels = meta_data['labels']
        data_paths = meta_data['files']
        self.data = []
        for p in tqdm(data_paths, total=meta_data.shape[0]):
            data = np.load(os.path.join(self.data_path, self.input_type, p).replace("\\", "/"))
            if len(data.shape) <= 1:
                data = np.expand_dims(data, axis=-1)
            self.data.append(data)

        self.data = [self.transforms(frame) if self.transforms is not None else frame for frame in self.data]

        self.labels = [self.label_mapping[label] for label in self.labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # apply augmentations if available
        output = (
            self.augmentations(self.data[idx]).float() if self.augmentations is not None else self.data[idx].float(),
            self.labels[idx],
        )
        return output


class SupervisedDataModule(LightningDataModule):
    """ LightningDataModule wrapper for SupervisedDataset

    Arguments:
        path: root folder of the dataset
        input_type: the type of input to load in this dataset
        batch_size: number of samples to include in a single batch
        split: path to the csv files containing the files and labels associated to the split
        train_transforms: transforms to apply to the input training data
        train_transforms: transforms to apply to the input testing data
        num_workers: number of workers for dataloaders
        augmentations: augmentations to apply to the input data
    """
    def __init__(
            self,
            path: str,
            input_type: str,
            batch_size: int,
            split: str,
            label_mapping: Dict[Union[str, int], int],
            train_transforms: Optional[transforms.Compose] = None,
            test_transforms: Optional[transforms.Compose] = None,
            num_workers: int = 1,
            augmentations: Optional[transforms.Compose] = None,
         
    ):
        super().__init__()
        self.path = path
        self.input_type = input_type
        self.batch_size = batch_size
        self.split = split
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.num_workers = num_workers
        self.augmentations = augmentations
        self.label_mapping = label_mapping

    def _init_dataloaders(self, stage):
        if str(stage) == "TrainerFn.FITTING":
            train_dataset = self._create_train_dataset()
            self.train = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            self.train = None

        if "val" in self.split and str(stage) == "TrainerFn.FITTING":
            val_dataset = self._create_val_dataset()
            self.val = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            self.val = None

        if "test" in self.split and str(stage) == "TrainerFn.TESTING":
            test_dataset = self._create_test_dataset()
            self.test = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            self.test = None

    def setup(self, stage=None):
        self._init_dataloaders(stage)

    def _create_train_dataset(self):
        print('Reading Supervised train data:')
        return SupervisedTorchDataset(
            self.path,
            self.input_type,
            self.split['train'],
            label_mapping=self.label_mapping,
            transforms=self.train_transforms,
            augmentations=self.augmentations,
        )

    def _create_val_dataset(self):
        print('Reading Supervised val data:')
        return SupervisedTorchDataset(
            self.path,
            self.input_type,
            self.split['val'],
            label_mapping=self.label_mapping,
            transforms=self.test_transforms,
            augmentations=self.augmentations,
        )

    def _create_test_dataset(self):
        print('Reading Supervised test data:')
        return SupervisedTorchDataset(
            self.path,
            self.input_type,
            self.split['test'],
            label_mapping=self.label_mapping,
            transforms=self.test_transforms,
            augmentations=None,
        )

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val if self.val is not None else None

    def test_dataloader(self):
        return self.test if self.test is not None else None
