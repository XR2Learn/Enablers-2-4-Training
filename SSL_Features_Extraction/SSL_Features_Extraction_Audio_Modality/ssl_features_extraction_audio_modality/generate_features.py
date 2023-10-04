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
from conf import CUSTOM_SETTINGS,MAIN_FOLDER,OUTPUTS_FOLDER
from encoders.cnn1d import CNN1D,CNN1D1L


def generate_ssl_features():
    """
    Function to extract SSL features and save to disk
    Args:
        None
    Returns:
        None
    """

    print(CUSTOM_SETTINGS)
    splith_paths = {'train':"outputs/train.csv",'val':"outputs/val.csv",'test':"outputs/test.csv"}

    encoder = CNN1D(
        len_seq=CUSTOM_SETTINGS["pre_processing_config"]['max_length'] * CUSTOM_SETTINGS["pre_processing_config"]['target_sr'],
        pretrained=CUSTOM_SETTINGS['encoder_config']['pretrained'] if "pretrained" in CUSTOM_SETTINGS['encoder_config'].keys() else None,
        **CUSTOM_SETTINGS["encoder_config"]['kwargs']
    )
    encoder.eval()
    print(encoder)

    save_folder = os.path.join(OUTPUTS_FOLDER, 'SSL_Features_Exteraction', 'ssl_features')
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)

    #TODO: iterate over keys or userspecified csv/files ?
    generate_and_save(encoder,splith_paths['train'],save_folder)
    generate_and_save(encoder,splith_paths['val'],save_folder)
    generate_and_save(encoder,splith_paths['test'],save_folder)


def generate_and_save(encoder,csv_path,out_path):
    """
    generate_and_save : given the encoder, extract the features and save to .npy files

    Args:
        encoder: the pytorch encoder model to extract features from
        csv_path: csv containing the paths to the files for which features have to be extracted and saved
        out_path: output path to save the features to
    Returns:
        none
    """

    meta_data = pd.read_csv(os.path.join(MAIN_FOLDER,csv_path))
    for data_path in tqdm(meta_data['files']):
        #TODO : find replacement for .replace('\\','/')) to have a seperator that works on all OS
        x = np.load(os.path.join(MAIN_FOLDER,data_path).replace('\\','/'))
        x_tensor = torch.tensor(np.expand_dims(x,axis=0) if len(x.shape)<=1 else x)
        features = encoder(x_tensor)
        #print(data_path.split(os.path.sep))
        np.save(os.path.join(out_path,data_path.split(os.path.sep)[-1]),features.detach().numpy())


if __name__ == '__main__':
    generate_ssl_features()
