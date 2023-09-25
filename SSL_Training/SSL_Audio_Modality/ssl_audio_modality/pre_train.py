# Python code here
import os
import torch
import torch

from pytorch_lightning import Trainer, seed_everything
from conf import CUSTOM_SETTINGS,MAIN_FOLDER
from ssl_dataset import SSLDataModule
from callbacks.setup_callbacks import setup_callbacks
from utils.init_utils import (init_augmentations, init_datamodule,
                              init_loggers, init_random_split, init_transforms,
                              setup_ssl_model)
from utils.utils import copy_file, generate_experiment_id, load_yaml_to_dict

from encoders.cnn1d import CNN1D,CNN1D1L
from ssl_methods.SimCLR import SimCLR

def run_pre_training():
    print(CUSTOM_SETTINGS)
    splith_paths = {'train':"outputs/train.csv",'val':"outputs/val.csv",'test':"outputs/test.csv"}

    train_transforms = {}
    test_transforms = {}
    #for modality in  preprocessing_configs['sensors'].keys():
    #    transform_cfg =  preprocessing_configs['sensors'][modality]['transforms']
    #    cur_train_transforms, cur_test_transforms = init_transforms(modality, transform_cfg)
    #    train_transforms.update(cur_train_transforms)
    #    test_transforms.update(cur_test_transforms)
    if 'transforms' in CUSTOM_SETTINGS.keys():
        train_transforms,test_transforms = init_transforms(CUSTOM_SETTINGS['transforms'])

    if 'augmentations' in CUSTOM_SETTINGS.keys():
        augmentations = init_augmentations(CUSTOM_SETTINGS['augmentations'])

    datamodule = SSLDataModule(
        path=MAIN_FOLDER,
        batch_size=CUSTOM_SETTINGS['ssl_config']['batch_size'],
        split=splith_paths,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        n_views=2,
        num_workers=2,
        augmentations=augmentations
    )
    #initialise encoder
    encoder = CNN1D(in_channels=1,
                    len_seq=CUSTOM_SETTINGS["pre_processing_config"]['max_length']*CUSTOM_SETTINGS["pre_processing_config"]['target_sr'],
                    out_channels=[2,2,2],
                    kernel_sizes=[7,7,7],
                    stride=4
                    )
    #initialise ssl model with configured SLL method
    ssl_model = SimCLR(encoder=encoder,ssl_batch_size=CUSTOM_SETTINGS['ssl_config']['batch_size'],**CUSTOM_SETTINGS['ssl_config']['kwargs'])
    ssl_model = SimCLR(encoder=encoder,ssl_batch_size=CUSTOM_SETTINGS['ssl_config']['batch_size'],**CUSTOM_SETTINGS['ssl_config']['kwargs'])

    print(ssl_model)
    #init callbacks  # initialize callbacks
    callbacks = setup_callbacks(
        early_stopping_metric="val_loss",
        no_ckpt=False,
        patience=15,
        patience=15,
    )
    # initialize Pytorch-Lightning Training
    trainer = Trainer(
        #logger=loggers,
        #accelerator='cpu' if args.gpus == 0 else 'gpu',
        #devices=None if args.gpus == 0 else args.gpus,
        deterministic=True, 
        deterministic=True, 
        default_root_dir=os.path.join(MAIN_FOLDER,'outputs','SSL_Training'),
        callbacks=callbacks,
        max_epochs=CUSTOM_SETTINGS['ssl_config']['epochs']
        max_epochs=CUSTOM_SETTINGS['ssl_config']['epochs']
    )

    # pre-train and report test loss
    trainer.fit(ssl_model, datamodule)
    metrics = trainer.test(ssl_model, datamodule, ckpt_path='best')
    print(metrics)

    #save weights
    torch.save(encoder.state_dict(), os.path.join(MAIN_FOLDER,'outputs','SSL_Training','test_encoder.pt'))


if __name__ == '__main__':
    run_pre_training()
