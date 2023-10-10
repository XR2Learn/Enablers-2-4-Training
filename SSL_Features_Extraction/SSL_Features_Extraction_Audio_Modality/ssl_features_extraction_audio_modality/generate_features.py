# Python code here
import os
import torch
import scipy
import pathlib
import numpy as np
import pandas as pd

from tqdm import tqdm
from conf import CUSTOM_SETTINGS
from pytorch_lightning import Trainer, seed_everything
from conf import CUSTOM_SETTINGS, MAIN_FOLDER, OUTPUTS_FOLDER,EXPERIMENT_ID
from utils.init_utils import (init_augmentations, init_datamodule,
                              init_loggers, init_random_split, init_transforms,
                              setup_ssl_model, init_encoder)


def generate_ssl_features():
    """
    Function to extract SSL features and save to disk
    Args:
        None
    Returns:
        None
    """

    print(CUSTOM_SETTINGS)
    splith_paths = {'train': "train.csv", 'val': "val.csv", 'test': "test.csv"}

    encoder = init_encoder(CUSTOM_SETTINGS["encoder_config"],
                           CUSTOM_SETTINGS['encoder_config']['pretrained'] if "pretrained_path" in CUSTOM_SETTINGS[
                               'encoder_config'].keys()  else f"{OUTPUTS_FOLDER}/SSL_Training/{EXPERIMENT_ID}_encoder.pt" if "pretrained_same_experiment" in CUSTOM_SETTINGS[
                               'encoder_config'].keys() and CUSTOM_SETTINGS['encoder_config']["pretrained_same_experiment"] else None
                           )
    encoder.eval()
    print(encoder)

    save_folder = os.path.join(OUTPUTS_FOLDER, 'SSL_features')
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)

    # TODO: iterate over keys or userspecified csv/files ?
    generate_and_save(encoder, splith_paths['train'], save_folder)
    generate_and_save(encoder, splith_paths['val'], save_folder)
    generate_and_save(encoder, splith_paths['test'], save_folder)


def generate_and_save(encoder, csv_path, out_path):
    """
    generate_and_save : given the encoder, extract the features and save to .npy files

    Args:
        encoder: the pytorch encoder model to extract features from
        csv_path: csv containing the paths to the files for which features have to be extracted and saved
        out_path: output path to save the features to
    Returns:
        none
    """

    meta_data = pd.read_csv(os.path.join(OUTPUTS_FOLDER, csv_path))
    for data_path in tqdm(meta_data['files']):
        # TODO : find replacement for .replace('\\','/')) to have a seperator that works on all OS
        x = np.load(
            os.path.join(OUTPUTS_FOLDER, CUSTOM_SETTINGS['encoder_config']['input_type'], data_path).replace('\\', '/'))
        x_tensor = torch.tensor(np.expand_dims(x, axis=0) if len(x.shape) <= 1 else x)
        features = encoder(x_tensor.T)
        # print(data_path.split(os.path.sep))
        np.save(os.path.join(out_path, data_path.split(os.path.sep)[-1]), features.detach().numpy())


if __name__ == '__main__':
    generate_ssl_features()
