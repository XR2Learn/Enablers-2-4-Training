import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningDataModule


class SSLTorchDataset(Dataset):
    """ Torch dataset class

    arguments
    ----------
    data_path: str
        root folder of the dataset
    input_type: str
        the type of input to load in this dataset
    split_path: str
        path to the csv file containing the files associated to the split
    transforms: torchvision.transforms.transforms.Compose
        transforms to apply to the input data,  created by using torchvision.transforms.Compose
    augmentations: torchvision.transforms.transforms.Compose
        augmentations to apply to the input data, created by using torchvision.transforms.Compose
    n_views: int
        number of views to create for the SSL
    """

    def __init__(self,
                 data_path,
                 input_type,
                 split_path,
                 transforms=None,
                 augmentations=None,
                 n_views=2,
                 ):
        self.data_path = data_path
        self.input_type = input_type
        self.split_path = split_path
        self.transforms = transforms
        self.augmentations = augmentations
        self.n_views = n_views

        self._process_recordings()

    def _process_recordings(self):
        """ Function (i) iterates through all data in the data_path and loads them into the class

        Parameters
        ----------
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
            None,  # None placeholder for label
            self.augmentations(self.data[idx]) if (
                self.augmentations is not None and self.n_views == 2
                ) else self.data[idx]
        )
        return output


class SSLDataModule(LightningDataModule):
    """ LightningDataModule

    arguments
    ----------
    path: str
        root folder of the dataset
    input_type: str
        the type of input to load in this dataset
    batch_size: float
        number of samples to include in a single batch
    split: str
        path to the csv files containing the files associated to the split
    train_transforms: torchvision.transforms.transforms.Compose
        transforms to apply to the input training data, created by using torchvision.transforms.Compose
    test_transforms: torchvision.transforms.transforms.Compose
        transforms to apply to the input testing data, created by using torchvision.transforms.Compose
    n_views: int
        number of views to create for the SSL
    augmentations: torchvision.transforms.transforms.Compose
        augmentations to apply to the input data, created by using torchvision.transforms.Compose
    """

    def __init__(self,
                 path,
                 input_type,
                 batch_size,
                 split,
                 train_transforms={},
                 test_transforms={},
                 n_views=2,
                 num_workers=1,
                 augmentations=None):
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
