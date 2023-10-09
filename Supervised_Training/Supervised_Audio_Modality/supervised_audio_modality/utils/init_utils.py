import importlib
from random import sample

#from emorec_toolbox.datasets.iemocap import (IEMOCAPDataManager,
#                                             IEMOCAPDataModule,
#                                             IEMOCAPTorchDataset)
#from emorec_toolbox.datasets.wesad import WESADDataModule, WESADTorchDataset
#from emorec_toolbox.models.mlp import LinearClassifier, MLPClassifier
from utils.augmentations import compose_random_augmentations
from pytorch_lightning.loggers import CSVLogger
from torchvision import transforms

#import models.bm


# Models
def setup_ssl_model(model_cfg):
    """ initializes ssl model and encoders from configs

    Parameters
    ----------   
    model_cfg : dict
        model configurations form YAML


    Returns
    -------
    (dict, pytorch_lightning.LightningModule)
        dictionary with encoders and the whole ssl model
    """    
    # init encoder
    encoders_dict = get_encoders(model_cfg)
    # init ssl framework
    ssl_model = getattr(importlib.import_module(model_cfg['from_module']), model_cfg['ssl_framework'])(
        encoders_dict,
        model_cfg['modalities'],
        model_cfg['grouped'],
        ssl_batch_size=model_cfg['experiment']['batch_size_pre_training'],
        optimizer_name_ssl=model_cfg['ssl_setup']['optimizer_name'],
        **model_cfg['ssl_setup']['kwargs']
    )
    
    return encoders_dict, ssl_model


def init_encoder(model_cfg, ckpt_path=None):
    """ Initialize (pre-trained) encoder from model configuration

    Parameters
    ----------
    model_cfg : dict
        configurations of the model: architecture (e.g., supported by emotion recognition toolbox) and hyperparameter
    ckpt_path : str, optional
        path to the file with the pre-trained model, by default None

    Returns
    -------
    torch.nn.Module
        initialized encoder
    """    
    module = importlib.import_module(f"{model_cfg['from_module']}")
    class_ = getattr(module, model_cfg['class_name'])
    return class_(**model_cfg['kwargs'],pretrained=ckpt_path)


def get_encoders(model_cfg):
    """ Initialize encoders for each modality.

    Parameters
    ----------
    model_cfg : dict
        configurations of models for all modalities

    Returns
    -------
    dict
        dictionary of encoder for each modality group
    """    
    encoders_dict = {}
    input_groups = model_cfg['modalities'] if model_cfg['grouped'] is None else model_cfg['grouped']
    for i, modality in enumerate(input_groups):
        model = init_encoder(model_cfg['encoders'][modality]['enc_architecture'])
        encoders_dict[modality] = model
    return encoders_dict


# Transforms and augmentations
def init_transforms(transforms_cfg, ssl_random_augmentations=False, random_augmentations_dict={}):
    # TODO: Docstring
    train = []
    test = []
    if transforms_cfg is not None:
        for t in transforms_cfg:
            module = importlib.import_module(f"utils.{t['from_module']}")
            class_ = getattr(module, t['class_name'])

            if "kwargs" in t:
                transform = class_(**t['kwargs'])
            else:
                transform = class_()

            train.append(transform)
            if t['in_test']:
                test.append(transform)
                
            print(f"added {t['class_name']} transformation")

    # if ssl_random_augmentations:
    #     train.extend(compose_random_augmentations(modality, random_augmentations_dict))
    composed_train_transform = transforms.Compose(train)
    composed_test_transform = transforms.Compose(test)

    #train_transforms = {modality: composed_train_transform }
    #test_transforms = {modality: composed_test_transform }

    return composed_train_transform,composed_test_transform


def init_augmentations(aug_dict):
    augmentations = None
    augmentations = compose_random_augmentations(aug_dict)
    augmentations = transforms.Compose(augmentations)


# Data splits
def init_random_split(split_pool, num_val=1, num_test=1):
    """ Initialize random data split based on split pool (e.g., subjects  or sessions)
        For example, if a dataset contains data from multiple subjects ['s1', 's2', 's3', 's4'], the function return the random split, such as
        {
            'train': ['s2', 's4'],
            'val': ['s1'],
            'test': ['s3']
        }
        The split is later passed to datamodule and used to split data.

    Parameters
    ----------
    split_pool : list
        pool of indentifiers used for split
    num_val : int, optional
        number of instances from split pool for validation, by default 1
    num_test : int, optional
        number of instances from split pool for test, by default 1

    Returns
    -------
    dict
        split pool randomly split into train, val, and test
    """    
    split = {}
    split['train'] = split_pool.copy()

    split['test'] = sample(split['train'], num_test)
    for t in split['test']: split['train'].remove(t)

    split['val'] = sample(split['train'], num_val)
    for v in split['val']: split['train'].remove(v)
    return split


# Data classes from emotion recognition toolbox for initialization of datamodule
# TODO: Explore how we can exclude this hard assignment
DATASET_CLASSES = {
    #'wesad': {
	#	'dataset': WESADTorchDataset,
    #    'datamodule': WESADDataModule
	#},
    #'iemocap':{
    #    'dataset': IEMOCAPTorchDataset,
    #    'datamodule': IEMOCAPDataModule,
    #    'manager': IEMOCAPDataManager
    #}
}


def init_datamodule(
        data_path, 
        dataset_name, 
        preprocessing_configs,
        batch_size,
        split, 
        train_transforms, 
        test_transforms,
        ssl = False, 
        n_views = 2, 
        num_workers = 1, 
        limited_k=None,
        train_augmentations=None
    ):
    """ Initialize PyTorch Lightning Datamodule for given dataset and parameters

    Parameters
    ----------
    data_path : str
        path to data on a local drive
    dataset_name : str
        name of a dataset
    preprocessing_configs : dict
        configs for pre-processing needed for datamodule
    batch_size : int
        number of instances in a single batch
    split : dict
        split dictionary. If no splits are needed, pass all split pool values in train, such as {'train': [...]}
    train_transforms : list
        list of transforms to be applied to train data, can be empty
    test_transforms : _type_
        list of transforms to be applied to test data, can be empty
    dataset_protocol : str
        labeling protocol supported by a dataset
    ssl : bool, optional
        flag for using self-supervised learning, by default False
    n_views : int, optional
       FUNC TO BE ADDED: number of views in ssl, by default 2
    num_workers : int, optional
        number of workers for processing data, by default 1
    limited_k : int, optional
        FUNC TO BE ADDED: limit the number of annotated data points, by default None
    train_augmentations : list, optional
        list of augmentations to be applied to train data, by default None

    Returns
    -------
    pytorch_lightning.LightningDataModule
        initialize datamodule
    """ 

    # TODO: Investigate a generic way to initialize all dataloaders: 
    # current implementation initializes datamodules for different datasets separately using DATASET_CLASSES

    # pick the specified dataset classes
    dataset_properties = DATASET_CLASSES[dataset_name]

    # init datamodule
    if dataset_name == 'wesad':
        data_module = dataset_properties['datamodule'](
            path=data_path, 
            sensors=list(preprocessing_configs['sensors'].keys()),
            batch_size=batch_size, 
            split=split, 
            train_transforms=train_transforms, 
            test_transforms=test_transforms, 
            train_augmentations=train_augmentations,
            ssl=ssl, 
            n_views=n_views, 
            num_workers=num_workers, 
            limited_k=limited_k,
            sample_len_sec=preprocessing_configs['general_pre_processing']['sequence_length'],
            overlap=preprocessing_configs['general_pre_processing']['overlap'],
            normalize=preprocessing_configs['general_pre_processing']['normalize']
        )
    
    elif dataset_name == 'iemocap':
        data_form = [d for d in list(preprocessing_configs['sensors'].keys()) if d != 'lengths']
        #data_df = IEMOCAPDataManager(data_path).data_df
        data_df=None
        data_module = dataset_properties['datamodule'](
            data_df=data_df,
            path=data_path,
            data_form=data_form,
            batch_size=batch_size,
            split=split,
            train_transforms=train_transforms, 
            test_transforms=test_transforms, 
            sample_len_sec=preprocessing_configs['general_pre_processing']['sequence_length'],
            target_sr=preprocessing_configs['general_pre_processing']['sample_rate'],
            ssl=ssl,
            num_workers=num_workers,
        )

    return data_module


# init loggers
def init_loggers(logger_names, logs_path, experiment_name, experiment_id):
    """ Initialize loggers. Currently supports CSVLogger only.

    Parameters
    ----------
    logger_names : list 
        list of logger names; supported loggers: ['csv']
    logs_path : str
        path to logs
    experiment_name : str
        name of the experiment
    experiment_id : str
        experiment id

    Returns
    -------
    list
        list of the initialized loggers that can be passed to pytorch_lightning.Trainer
    """  
    loggers = []
    if 'csv' in logger_names:
        loggers.append(CSVLogger(logs_path, name=experiment_name, version=experiment_id))
    return loggers