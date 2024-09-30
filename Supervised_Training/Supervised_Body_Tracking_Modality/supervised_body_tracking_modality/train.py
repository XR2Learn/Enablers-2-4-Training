import os
import shutil
import warnings

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader

from conf import CUSTOM_SETTINGS, MODALITY_FOLDER, COMPONENT_OUTPUT_FOLDER, EXPERIMENT_ID, modality
from callbacks.setup_callbacks import setup_callbacks
from classification_model import SupervisedModel
from classifiers.mlp import MLPClassifier
from utils.init_utils import init_encoder

warnings.filterwarnings('ignore')


def run_supervised_training():
    # Construct the full paths for each CSV file
    train_data_path = os.path.join(MODALITY_FOLDER, 'train.csv')
    test_data_path = os.path.join(MODALITY_FOLDER, 'test.csv')
    val_data_path = os.path.join(MODALITY_FOLDER, 'val.csv')

    # Load data from CSV files using the paths
    train_data = pd.read_csv(train_data_path, index_col=0)
    test_data = pd.read_csv(test_data_path, index_col=0)
    val_data = pd.read_csv(val_data_path, index_col=0)

    # Separate features and target labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values

    # Reshape data for Conv1D: (num_samples, segment_size, num_features)
    segment_size = CUSTOM_SETTINGS["pre_processing_config"]["seq_len"] *\
        CUSTOM_SETTINGS["pre_processing_config"]["frequency"]
    X_train = X_train.reshape((X_train.shape[0], segment_size, -1))
    X_test = X_test.reshape((X_test.shape[0], segment_size, -1))
    X_val = X_val.reshape((X_val.shape[0], segment_size, -1))

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)

    # In SUPSI implementation unknown class is also used to train the model
    num_classes = len(np.unique(y_train_encoded))

    train_dataset = TensorDataset(torch.permute(torch.tensor(X_train), [0, 2, 1]), torch.tensor(y_train_encoded))
    train_loader = DataLoader(
        train_dataset,
        batch_size=CUSTOM_SETTINGS["sup_config"]["batch_size"],
        pin_memory=True
    )

    val_dataset = TensorDataset(torch.permute(torch.tensor(X_val), [0, 2, 1]), torch.tensor(y_val_encoded))
    val_loader = DataLoader(
        val_dataset,
        batch_size=CUSTOM_SETTINGS["sup_config"]["batch_size"],
        pin_memory=True
    )

    test_dataset = TensorDataset(torch.permute(torch.tensor(X_test), [0, 2, 1]), torch.tensor(y_test_encoded))
    test_loader = DataLoader(
        test_dataset,
        batch_size=CUSTOM_SETTINGS["sup_config"]["batch_size"],
        pin_memory=True
    )

    # initialise encoder
    CUSTOM_SETTINGS["encoder_config"]["kwargs"]["len_seq"] = X_train.shape[1]
    CUSTOM_SETTINGS["encoder_config"]["kwargs"]["in_channels"] = X_train.shape[2]

    print(CUSTOM_SETTINGS["encoder_config"])

    encoder = init_encoder(
        model_cfg=CUSTOM_SETTINGS["encoder_config"],
    )

    ckpt_name = (
        f"{EXPERIMENT_ID}_"
        f"{CUSTOM_SETTINGS['dataset_config']['dataset_name']}_"
        f"{modality}_"
        f"{CUSTOM_SETTINGS['encoder_config']['class_name']}"
    )

    # add classification head to encoder
    classifier = MLPClassifier(
        encoder.out_size,
        num_classes,
        hidden=CUSTOM_SETTINGS['sup_config'].get("dense_neurons", [64]),
        p_dropout=CUSTOM_SETTINGS['sup_config'].get("dropout", None)
    )
    model = SupervisedModel(encoder=encoder, classifier=classifier, **CUSTOM_SETTINGS['sup_config']['kwargs'])

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
        patience=10,
        dirpath=COMPONENT_OUTPUT_FOLDER,
        monitor=CUSTOM_SETTINGS['sup_config']['monitor'] if 'monitor' in CUSTOM_SETTINGS['sup_config'] else "val_loss",
        checkpoint_filename=checkpoint_filename
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
    # the model checkpoints (best and last if provided) will be saved in
    # /COMPONENT_OUTPUT_FOLDER/{EXPERIMENT_ID}_model_lightning.ckpt
    trainer.fit(model, train_loader, val_loader)

    # evaluate model on the test set, by default the best model
    trainer.test(model, test_loader, ckpt_path="best")

    # save weights of the classifier independently for future use with SSL features
    torch.save(
        classifier.state_dict(),
        os.path.join(COMPONENT_OUTPUT_FOLDER, f'{ckpt_name}_classifier.pt')
    )


if __name__ == '__main__':
    run_supervised_training()
