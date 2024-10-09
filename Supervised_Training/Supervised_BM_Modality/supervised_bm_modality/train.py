# Python code here
import json
import os

import torch
from pytorch_lightning import Trainer

from callbacks.setup_callbacks import setup_callbacks
from classification_model import SupervisedModel
from classifiers.linear import LinearClassifier
from conf import CUSTOM_SETTINGS, MODALITY, MODALITY_FOLDER, COMPONENT_OUTPUT_FOLDER, EXPERIMENT_ID, LABEL_TO_ID
from supervised_dataset import SupervisedDataModule
from utils.init_utils import (init_augmentations, init_transforms, init_encoder)


def run_supervised_training():
    print(json.dumps(CUSTOM_SETTINGS, indent=4))
    splith_paths = {'train': "train.csv", 'val': "val.csv", 'test': "test.csv"}

    train_transforms = {}
    test_transforms = {}

    if 'transforms' in CUSTOM_SETTINGS[MODALITY].keys():
        train_transforms, test_transforms = init_transforms(CUSTOM_SETTINGS[MODALITY]['transforms'])

    # for now, don't use augmentations during supervised training
    augmentations = None
    if (CUSTOM_SETTINGS[MODALITY]['sup_config']['use_augmentations_in_sup']) and (
            'augmentations' in CUSTOM_SETTINGS[MODALITY].keys()):
        augmentations = init_augmentations(CUSTOM_SETTINGS[MODALITY]['augmentations'])
        print("Augmentations loaded successfully")
    else:
        print("No augmentations loaded")

    label_mapping = LABEL_TO_ID[CUSTOM_SETTINGS['dataset_config']['dataset_name']]

    datamodule = SupervisedDataModule(
        path=MODALITY_FOLDER,
        input_type=CUSTOM_SETTINGS[MODALITY]['sup_config']['input_type'],
        batch_size=CUSTOM_SETTINGS[MODALITY]['sup_config']['batch_size'],
        split=splith_paths,
        label_mapping=label_mapping,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        augmentations=augmentations,
    )

    ckpt_name = (
        f"{EXPERIMENT_ID}_"
        f"{CUSTOM_SETTINGS['dataset_config']['dataset_name']}_"
        f"{MODALITY}_"
        f"{CUSTOM_SETTINGS[MODALITY]['sup_config']['input_type']}_"
        f"{CUSTOM_SETTINGS[MODALITY]['encoder_config']['class_name']}"
    )

    if "pretrained_path" in CUSTOM_SETTINGS[MODALITY]['encoder_config'].keys():
        ckpt_path = CUSTOM_SETTINGS[MODALITY]['encoder_config']['pretrained_path']
    elif (
        "pretrained_same_experiment" in CUSTOM_SETTINGS[MODALITY]['encoder_config'].keys() and
        CUSTOM_SETTINGS[MODALITY]['encoder_config']["pretrained_same_experiment"]
    ):
        ckpt_path = os.path.join(MODALITY_FOLDER, "ssl_training", f"{ckpt_name}_encoder.pt")
    else:
        ckpt_path = None

    # initialise encoder
    encoder = init_encoder(
        model_cfg=CUSTOM_SETTINGS[MODALITY]["encoder_config"],
        ckpt_path=ckpt_path
    )

    # add classification head to encoder
    num_classes = CUSTOM_SETTINGS['dataset_config'].get("number_of_labels", 3)
    if isinstance(num_classes, dict):
        num_classes = num_classes.get(MODALITY, 3)
    classifier = LinearClassifier(encoder.out_size, num_classes)
    model = SupervisedModel(encoder=encoder, classifier=classifier, **CUSTOM_SETTINGS[MODALITY]['sup_config']['kwargs'])

    checkpoint_filename = f'{ckpt_name}_model'

    # by default lightning does not overwrite checkpoints, but rather creates different versions (v1, v2, etc.)
    # for the sample checkpoint_filename. Thus, in order to enable overwriting, we delete checkpoint if it exists.
    if os.path.exists(os.path.join(COMPONENT_OUTPUT_FOLDER, checkpoint_filename + '.ckpt')):
        os.remove(os.path.join(COMPONENT_OUTPUT_FOLDER, checkpoint_filename + '.ckpt'))

    # initialize callbacks
    callbacks = setup_callbacks(
        early_stopping_metric="val_loss",
        no_ckpt=False,
        num_classes=num_classes,
        patience=50,
        dirpath=COMPONENT_OUTPUT_FOLDER,
        monitor=CUSTOM_SETTINGS[MODALITY]['sup_config']['monitor'] if 'monitor' in CUSTOM_SETTINGS[MODALITY]['sup_config'] else "val_loss",
        checkpoint_filename=checkpoint_filename
    )

    # initialize Pytorch-Lightning Trainer
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        deterministic=True,
        default_root_dir=os.path.join(COMPONENT_OUTPUT_FOLDER),
        callbacks=callbacks,
        max_epochs=CUSTOM_SETTINGS[MODALITY]['sup_config']['epochs']
    )

    # train model and report metrics
    # the model checkpoints (best and last if provided) will be saved in
    # /COMPONENT_OUTPUT_FOLDER/{EXPERIMENT_ID}_model_lightning.ckpt
    trainer.fit(model, datamodule)

    # evaluate model on the test set, by default the best model
    trainer.test(model, datamodule, ckpt_path="best")

    # save weights of the classifier independently for future use with SSL features
    torch.save(
        classifier.state_dict(),
        os.path.join(COMPONENT_OUTPUT_FOLDER, f'{ckpt_name}_classifier.pt')
    )


if __name__ == '__main__':
    run_supervised_training()
