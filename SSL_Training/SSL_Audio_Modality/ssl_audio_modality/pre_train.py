# Python code here
import os
import torch
import pathlib

from pytorch_lightning import Trainer, seed_everything
from conf import CUSTOM_SETTINGS, OUTPUTS_FOLDER, COMPONENT_OUTPUT_FOLDER,EXPERIMENT_ID
from ssl_dataset import SSLDataModule
from callbacks.setup_callbacks import setup_callbacks
from utils.init_utils import (init_augmentations, init_datamodule,
                              init_loggers, init_random_split, init_transforms,
                              setup_ssl_model, init_encoder)
from utils.utils import copy_file, generate_experiment_id, load_yaml_to_dict

from encoders.cnn1d import CNN1D, CNN1D1L
from ssl_methods.SimCLR import SimCLR
from ssl_methods.VICReg import VICReg


def run_pre_training():
    """
    file to pretrain the encoder model
    Args:
        None
    """
    print(CUSTOM_SETTINGS)
    splith_paths = {'train': "train.csv", 'val': "val.csv", 'test': "test.csv"}

    train_transforms = {}
    test_transforms = {}
    if 'transforms' in CUSTOM_SETTINGS.keys():
        train_transforms, test_transforms = init_transforms(CUSTOM_SETTINGS['transforms'])

    if 'augmentations' in CUSTOM_SETTINGS.keys():
        augmentations = init_augmentations(CUSTOM_SETTINGS['augmentations'])

    datamodule = SSLDataModule(
        path=OUTPUTS_FOLDER,
        input_type=CUSTOM_SETTINGS['encoder_config']['input_type'],
        batch_size=CUSTOM_SETTINGS['ssl_config']['batch_size'],
        split=splith_paths,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        n_views=2,
        num_workers=0,
        augmentations=augmentations
    )
    # initialise encoder
    encoder = init_encoder(CUSTOM_SETTINGS["encoder_config"],
                           CUSTOM_SETTINGS['encoder_config']['pretrained'] if "pretrained" in CUSTOM_SETTINGS[
                               'encoder_config'].keys() else None
                           )
    # initialise ssl model with configured SLL method
    # ssl_model = VICReg(encoder=encoder,ssl_batch_size=CUSTOM_SETTINGS['ssl_config']['batch_size'],**CUSTOM_SETTINGS['ssl_config']['kwargs'])
    ssl_model = setup_ssl_model(encoder, model_cfg=CUSTOM_SETTINGS['ssl_config'])
    print(ssl_model)
    # init callbacks  # initialize callbacks
    callbacks = setup_callbacks(
        early_stopping_metric="val_loss",
        no_ckpt=False,
        patience=15,
    )
    # initialize Pytorch-Lightning Training
    trainer = Trainer(
        # logger=loggers,
        # accelerator='cpu' if args.gpus == 0 else 'gpu',
        # devices=None if args.gpus == 0 else args.gpus,
        deterministic=True,
        default_root_dir=os.path.join(COMPONENT_OUTPUT_FOLDER),
        callbacks=callbacks,
        max_epochs=CUSTOM_SETTINGS['ssl_config']['epochs'],
        log_every_n_steps=5
    )

    # pre-train and report test loss
    trainer.fit(ssl_model, datamodule)
    metrics = trainer.test(ssl_model, datamodule, ckpt_path='best')
    print(metrics)

    #load in best weights
    ssl_model.load_from_checkpoint(callbacks[1].best_model_path)
    # save weights
    # pathlib.Path(os.path.join(OUTPUTS_FOLDER,'SSL_Training')).mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(COMPONENT_OUTPUT_FOLDER, f'{EXPERIMENT_ID}_encoder.pt'))


if __name__ == '__main__':
    run_pre_training()
