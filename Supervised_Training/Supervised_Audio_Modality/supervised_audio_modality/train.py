# Python code here
import os
import torch

from pytorch_lightning import Trainer
from conf import CUSTOM_SETTINGS, OUTPUTS_FOLDER, COMPONENT_OUTPUT_FOLDER, EXPERIMENT_ID
from supervised_dataset import SupervisedDataModule
from callbacks.setup_callbacks import setup_callbacks
from utils.init_utils import (init_augmentations, init_transforms, init_encoder)

from classification_model import SupervisedModel
from classifiers.linear import LinearClassifier


def run_supervised_training():
    print(CUSTOM_SETTINGS)
    splith_paths = {'train': "train.csv", 'val': "val.csv", 'test': "test.csv"}

    train_transforms = {}
    test_transforms = {}

    if 'transforms' in CUSTOM_SETTINGS.keys():
        train_transforms, test_transforms = init_transforms(CUSTOM_SETTINGS['transforms'])

    # for now, don't use augmentations during supervised training
    augmentations = None
    if (CUSTOM_SETTINGS['sup_config']['use_augmentations_in_sup']) and (
            'augmentations' in CUSTOM_SETTINGS.keys()):
        augmentations = init_augmentations(CUSTOM_SETTINGS['augmentations'])
        print("Augmentations loaded successfully")
    else:
        print("No augmentations loaded")

    datamodule = SupervisedDataModule(
        path=OUTPUTS_FOLDER,
        input_type=CUSTOM_SETTINGS['sup_config']['input_type'],
        batch_size=CUSTOM_SETTINGS['sup_config']['batch_size'],
        split=splith_paths,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        n_views=2,
        num_workers=0,
        augmentations=augmentations
    )
    # initialise encoder
    encoder = init_encoder(
        model_cfg=CUSTOM_SETTINGS["encoder_config"],
        ckpt_path=CUSTOM_SETTINGS['encoder_config']['pretrained'] if (
            "pretrained_path" in CUSTOM_SETTINGS['encoder_config'].keys()
        ) else f"{OUTPUTS_FOLDER}/ssl_training/{EXPERIMENT_ID}_encoder.pt" if (
            "pretrained_same_experiment" in CUSTOM_SETTINGS['encoder_config'].keys()
            and CUSTOM_SETTINGS['encoder_config']["pretrained_same_experiment"]
        ) else None
    )

    # add classification head to encoder
    classifier = LinearClassifier(encoder.out_size, CUSTOM_SETTINGS['dataset_config']['number_of_labels'])
    model = SupervisedModel(encoder=encoder, classifier=classifier, **CUSTOM_SETTINGS['sup_config']['kwargs'])

    # initialize callbacks
    callbacks = setup_callbacks(
        early_stopping_metric="val_loss",
        no_ckpt=False,
        patience=50,
        num_classes=CUSTOM_SETTINGS['dataset_config']['number_of_labels']
    )

    # initialize Pytorch-Lightning Trainer
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        deterministic=True,
        default_root_dir=os.path.join(COMPONENT_OUTPUT_FOLDER),
        callbacks=callbacks,
        max_epochs=CUSTOM_SETTINGS['sup_config']['epochs']
    )

    # train model and report metrics
    trainer.fit(model, datamodule)

    # load in best weights
    metrics = trainer.test(model, datamodule)
    print(metrics)

    # save weights
    torch.save(model.state_dict(), os.path.join(COMPONENT_OUTPUT_FOLDER, f'{EXPERIMENT_ID}_model.pt'))
    torch.save(classifier.state_dict(), os.path.join(COMPONENT_OUTPUT_FOLDER, f'{EXPERIMENT_ID}_classifier.pt'))


if __name__ == '__main__':
    run_supervised_training()
