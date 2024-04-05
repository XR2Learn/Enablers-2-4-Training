import os

import torch
from pytorch_lightning import Trainer

from conf import (
    CUSTOM_SETTINGS,
    MODALITY_FOLDER,
    EXPERIMENT_ID,
    COMPONENT_OUTPUT_FOLDER
)
from ssl_dataset import SSLDataModule
from callbacks.setup_callbacks import setup_callbacks
from utils.init_utils import (init_augmentations, init_transforms,
                              setup_ssl_model, init_encoder)


def run_pre_training():
    """
    Pre-train encoder and save checkpoints.
    """
    print(CUSTOM_SETTINGS)

    if (
        'get_ssl' in CUSTOM_SETTINGS['pre_processing_config'] and
        CUSTOM_SETTINGS['pre_processing_config']['get_ssl']
    ):
        prefix = "ssl_"
    else:
        prefix = ""

    splith_paths = {'train': f"{prefix}train.csv", 'val': f"{prefix}val.csv", 'test': f"{prefix}test.csv"}

    train_transforms = None
    test_transforms = None
    if 'transforms' in CUSTOM_SETTINGS.keys():
        train_transforms, test_transforms = init_transforms(CUSTOM_SETTINGS['transforms'])

    if 'augmentations' in CUSTOM_SETTINGS.keys():
        augmentations = init_augmentations(CUSTOM_SETTINGS['augmentations'])
        print(augmentations)

    datamodule = SSLDataModule(
        path=MODALITY_FOLDER,
        input_type=prefix + CUSTOM_SETTINGS['ssl_config']['input_type'],
        batch_size=CUSTOM_SETTINGS['ssl_config']['batch_size'],
        split=splith_paths,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        n_views=2,
        num_workers=0,
        augmentations=augmentations
    )

    # initialise encoder
    encoder = init_encoder(
        CUSTOM_SETTINGS["encoder_config"],
        CUSTOM_SETTINGS['encoder_config']['pretrained'] if (
            "pretrained" in CUSTOM_SETTINGS['encoder_config'].keys()
            ) else None
    )
    print(encoder)

    ssl_model = setup_ssl_model(encoder, model_cfg=CUSTOM_SETTINGS['ssl_config'])
    print(ssl_model)

    modality = CUSTOM_SETTINGS['dataset_config']['modality'] if (
        'modality' in CUSTOM_SETTINGS['dataset_config']
    ) else 'default_modality'

    checkpoint_filename = (
        f"{EXPERIMENT_ID}_"
        f"{CUSTOM_SETTINGS['dataset_config']['dataset_name']}_"
        f"{modality}_"
        f"{CUSTOM_SETTINGS['ssl_config']['input_type']}_"
        f"{CUSTOM_SETTINGS['encoder_config']['class_name']}"
    )

    ssl_checkpoint = checkpoint_filename + "ssl_model"

    # by default lightning does not overwrite checkpoints, but rather creates different versions (v1, v2, etc.)
    # for the sample checkpoint_filename. Thus, in order to enable overwriting, we delete checkpoint if it exists.
    if os.path.exists(os.path.join(COMPONENT_OUTPUT_FOLDER, ssl_checkpoint + '.ckpt')):
        os.remove(os.path.join(COMPONENT_OUTPUT_FOLDER, ssl_checkpoint + '.ckpt'))

    # initialize callbacks
    callbacks = setup_callbacks(
        early_stopping_metric="val_loss",
        no_ckpt=False,
        patience=100,
        dirpath=COMPONENT_OUTPUT_FOLDER,
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        checkpoint_filename=ssl_checkpoint
    )

    # initialize Pytorch-Lightning Training
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        deterministic=True,
        default_root_dir=COMPONENT_OUTPUT_FOLDER,
        callbacks=callbacks,
        max_epochs=CUSTOM_SETTINGS['ssl_config']['epochs'],
        log_every_n_steps=10,
    )

    trainer.fit(ssl_model, datamodule)
    metrics = trainer.test(ssl_model, datamodule, ckpt_path='best')
    print(metrics)

    # if save_last_encoder, the weights of encoder will be taken and saved from the last epoch of ssl pre-training
    if "save_last_encoder" in CUSTOM_SETTINGS["ssl_config"] and CUSTOM_SETTINGS["ssl_config"]["save_last_encoder"]:
        ssl_model = ssl_model.__class__.load_from_checkpoint(
            os.path.join(COMPONENT_OUTPUT_FOLDER, checkpoint_filename + "_last.ckpt"),
            encoder=encoder
        )

    torch.save(
        ssl_model.encoder.state_dict(),
        os.path.join(
            COMPONENT_OUTPUT_FOLDER,
            f'{checkpoint_filename}_encoder.pt'
        )
    )


if __name__ == '__main__':
    run_pre_training()
