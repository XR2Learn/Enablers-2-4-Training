import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Optional

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule


class SSLTorchDataset(Dataset):
    """ Torch dataset class for bio-measurement processing

        Args:
            data_path: path to dataset
            input_type: subfolder: type of input data (e.g., normalize, standardize)
            split_path: path to csv with current split
            transforms: transforms to apply to input data
            augmentations: augmentations to apply
            n_views: number of views to generate. Currently, 1 or 2 augmented views are supported
    """
    def __init__(
            self,
            data_path: str,
            input_type: str,
            split_path: str,
            transforms: Optional[transforms.Compose] = None,
            augmentations: Optional[transforms.Compose] = None,
            n_views: int = 2,
    ):
        self.data_path = data_path
        self.input_type = input_type
        self.split_path = split_path
        self.transforms = transforms
        self.augmentations = augmentations
        self.n_views = n_views

        assert self.n_views in [1, 2], "Number of views n_views supported: [1, 2]"
        self._process_recordings()

    def _process_recordings(self):
        """ Iterates through all data in the data_path and loads them into the class
        """

        # read meta data
        print(os.path.join(self.data_path, self.split_path))
        meta_data = pd.read_csv(os.path.join(self.data_path, self.split_path), index_col=0)
        data_paths = meta_data['files']
        self.data = []
        # go over all files in the given meta data
        for path in tqdm(data_paths, total=meta_data.shape[0]):
            # load from .npy file
            data = np.load(os.path.join(self.data_path, self.input_type, path).replace("\\", "/"))
            # add channel dimension if necessary
            if len(data.shape) <= 1:
                data = np.expand_dims(data, axis=-1)
            self.data.append(data)

        self.data = [self.transforms(frame) if self.transforms is not None else frame for frame in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # apply augmentations if available
        output = (
            self.augmentations(self.data[idx]) if self.augmentations is not None else self.data[idx],
            self.augmentations(self.data[idx]) if (
                self.augmentations is not None and self.n_views == 2
                ) else self.data[idx]
        )
        return output


class SSLDataModule(LightningDataModule):
    """ LightningDataModule for SSL

    Args:
        path: root folder of the dataset
        input_type: the type of input to load in this dataset
        batch_size: number of samples to include in a single batch
        split: path to the csv files containing the files associated to the split
        train_transforms: transforms to apply to the input training data
        test_transforms: transforms to apply to the input testing data
        n_views: number of views to create for the SSL
        augmentations: augmentations to apply to the input data, created by using torchvision.transforms.Compose
    """

    def __init__(self,
                 path: str,
                 input_type: str,
                 batch_size: int,
                 split: str,
                 train_transforms: transforms.Compose = None,
                 test_transforms: transforms.Compose = None,
                 n_views: int = 2,
                 num_workers: int = 1,
                 augmentations: transforms.Compose = None):
        super().__init__()
        self.path = path
        self.input_type = input_type
        self.batch_size = batch_size
        self.split = split
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.n_views = n_views
        self.num_workers = num_workers
        self.augmentations = augmentations

    def _init_dataloaders(self, stage):
        if str(stage) == "TrainerFn.FITTING":
            train_dataset = self._create_train_dataset()
            self.train = DataLoader(train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=self.num_workers,
                                    pin_memory=True)
        else:
            self.train = None

        if "val" in self.split and str(stage) == "TrainerFn.FITTING":
            val_dataset = self._create_val_dataset()
            self.val = DataLoader(val_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=self.num_workers,
                                  pin_memory=True)
        else:
            self.val = None

        if "test" in self.split and str(stage) == "TrainerFn.TESTING":
            test_dataset = self._create_test_dataset()
            self.test = DataLoader(test_dataset,
                                   batch_size=self.batch_size,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=self.num_workers,
                                   pin_memory=True)
        else:
            self.test = None

    def setup(self, stage=None):
        # TrainerFn.FITTING,TrainerFn.TESTING
        self._init_dataloaders(stage)

    def _create_train_dataset(self):
        print('Reading SSL train data:')
        return SSLTorchDataset(self.path,
                               self.input_type,
                               self.split['train'],
                               transforms=self.train_transforms,
                               augmentations=self.augmentations,
                               n_views=self.n_views, )

    def _create_val_dataset(self):
        print('Reading SSL val data:')
        return SSLTorchDataset(self.path,
                               self.input_type,
                               self.split['val'],
                               transforms=self.test_transforms,
                               augmentations=self.augmentations,
                               n_views=self.n_views, )

    def _create_test_dataset(self):
        print('Reading SSL test data:')
        return SSLTorchDataset(self.path,
                               self.input_type,
                               self.split['test'],
                               transforms=self.test_transforms,
                               augmentations=None,
                               n_views=self.n_views, )

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val if self.val is not None else None

    def test_dataloader(self):
        return self.test if self.test is not None else None
