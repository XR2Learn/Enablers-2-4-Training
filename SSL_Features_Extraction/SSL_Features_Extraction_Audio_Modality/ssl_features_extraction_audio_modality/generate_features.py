# Python code here
import os
import pathlib
import numpy as np
import pandas as pd

from tqdm import tqdm
from conf import CUSTOM_SETTINGS, OUTPUTS_FOLDER, EXPERIMENT_ID
from utils.init_utils import init_encoder, init_transforms


def generate_ssl_features():
    """
    Function to extract SSL features and save to disk
    Args:
        None
    Returns:
        None
    """

    print(CUSTOM_SETTINGS)

    # currently, use custom train, val, test csv paths
    data_paths = ["train.csv", "val.csv", "test.csv"]

    encoder = init_encoder(
        CUSTOM_SETTINGS["encoder_config"],
        CUSTOM_SETTINGS['encoder_config']['pretrained'] if (
            "pretrained_path" in CUSTOM_SETTINGS['encoder_config']
        ) else f"{OUTPUTS_FOLDER}/ssl_training/{EXPERIMENT_ID}_encoder.pt" if (
            "pretrained_same_experiment" in CUSTOM_SETTINGS['encoder_config'] and
            CUSTOM_SETTINGS['encoder_config']["pretrained_same_experiment"]
        ) else None
    )

    encoder.eval()
    print(encoder)

    save_folder = os.path.join(OUTPUTS_FOLDER, 'SSL_features')
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)

    if 'transforms' in CUSTOM_SETTINGS.keys():
        train_transforms, test_transforms = init_transforms(CUSTOM_SETTINGS['transforms'])
        transforms = {
            "train": train_transforms,
            "val": train_transforms,
            "test": test_transforms
        }

    for path_ in data_paths:
        generate_and_save(encoder, path_, save_folder, transforms)


def generate_and_save(encoder, data_path, out_path, transforms):
    """
    generate_and_save : given the encoder, extract the features and save to .npy files

    Args:
        encoder: the pytorch encoder model to extract features from
        data_path: can take two forms:
            - csv containing the paths to the files for which features have to be extracted and saved
            - path to a folder with .npy files
        out_path: output path to save the features to
    Returns:
        none
    """
    if data_path.endswith(".csv") and not os.path.isdir(data_path):
        files = pd.read_csv(os.path.join(OUTPUTS_FOLDER, data_path))['files']
        filename = os.path.basename(data_path)
        cur_transforms = transforms[filename.split(".")[0]]
    elif os.path.isdir(data_path):
        files = os.path.listdir(data_path)
        cur_transforms = transforms["train"]
    else:
        raise ValueError("Incorrect data_path format")

    for data_path in tqdm(files):
        x = np.load(
            os.path.join(OUTPUTS_FOLDER, CUSTOM_SETTINGS['ssl_config']['input_type'], data_path).replace('\\', '/')
        )
        if len(x.shape) <= 1:
            x = np.expand_dims(x, axis=-1)
        x_tensor = cur_transforms(x)
        features = encoder(x_tensor)
        np.save(os.path.join(out_path, data_path.split(os.path.sep)[-1]), features.detach().numpy())


if __name__ == '__main__':
    generate_ssl_features()
