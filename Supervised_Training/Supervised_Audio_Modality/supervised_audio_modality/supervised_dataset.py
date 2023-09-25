import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import scipy

from pytorch_lightning import LightningDataModule

class SupervisedTorchDataset(Dataset):
    """ Torch dataset class for WESAD

    arguments
    ----------
    data_path: str
        root folder of the dataset
    subjects: list
        subjects to be used (if None all subjects are used)
    labels: list
        labels to be used (normally, only 1, 2, 3 and 4 are being used)
    sample_len_sec: int
        length of time window for segmentations
    sensors: list
        sensor names to be used (if None all sensors are used)
    normalize: bool
        if true recordings are normalized for each subject to zero mean and unit variance per channel
    """
    def __init__(self, 
                 data_path,
                 split_path, 
                 transforms=None,
                 augmentations=None,
                 n_views=2,
        ):
        self.data_path = data_path
        # subjects and labels to retrieve
        self.split_path = split_path
        self.transforms = transforms
        self.augmentations = augmentations
        self.n_views = n_views

        self._process_recordings()
                
    def _process_recordings(self, normalize=False):
        """ Function (i) iterates through all subjects' data in the data_path and processes them one by one (normalization, sampling);
                    (ii) merges time windows, subjects and labels from different subjects

        Parameters
        ----------
        normalize : bool, optional
            flag for using normalization, by default False and assumes preprocessed data
        """

        
        # read, normalize and sample recordings
        print(os.path.join(self.data_path,self.split_path))
        meta_data = pd.read_csv(os.path.join(self.data_path,self.split_path),index_col=0)
        self.labels=meta_data['labels']
        data_paths = meta_data['files']
        self.data=[]
        for p in tqdm(data_paths,total=meta_data.shape[0]):
            #TODO: change the save format of audio files to .npy to have a more generic way to read all sorts of data/modalities
            sr,audio = scipy.io.wavfile.read(os.path.join(self.data_path,p))
            if len(audio.squeeze().shape)==1:
                audio = np.expand_dims(audio,axis=-1)
                #print(np.expand_dims(audio,axis=-1).shape)
            self.data.append(audio)

        self.data = [self.transforms(frame) if self.transforms else frame for frame in self.data]

        # re-arrange recordings and merge across subjects

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # apply augmentations if available
        if self.augmentations is not None:
            aug1 = {k:self.augmentations(v) for k,v in self.data[idx].items()} 
            aug2 = {k:self.augmentations(v) for k,v in self.data[idx].items()} if self.n_views == 2 else self.data[idx]

        output = (
            aug1 if self.augmentations is not None else self.data[idx],
            self.labels[idx],
            aug2 if self.augmentations is not None else self.data[idx]
        )
        return output


class SupervisedDataModule(LightningDataModule):
    def __init__(self,
            path,
            batch_size,
            split,
            train_transforms = {},
            test_transforms = {},
            n_views = 2,
            num_workers = 1,
            limited_k=None,
            augmentations = None):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.split = split
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.n_views = n_views
        self.num_workers = num_workers
        self.limited_k = limited_k
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
        #TrainerFn.FITTING,TrainerFn.TESTING 
        self._init_dataloaders(stage)
        
    def _create_train_dataset(self):
        print('Reading SSL train data:')
        return SupervisedTorchDataset(self.path, 
                                 self.split['train'], 
                                 transforms=self.train_transforms,
                                 augmentations=self.augmentations,
                                 n_views=self.n_views,)
    
    def _create_val_dataset(self):
        print('Reading SSL val data:')
        return SupervisedTorchDataset(self.path, 
                                 self.split['val'],  
                                 transforms=self.test_transforms,
                                 augmentations=self.augmentations,
                                 n_views=self.n_views,)
    
    def _create_test_dataset(self):
        print('Reading SSL test data:')
        return SupervisedTorchDataset(self.path, 
                                 self.split['test'], 
                                 transforms=self.test_transforms,
                                 augmentations=None,
                                 n_views=self.n_views,)
        
    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val if self.val is not None else None

    def test_dataloader(self):
        return self.test if self.test is not None else None
    