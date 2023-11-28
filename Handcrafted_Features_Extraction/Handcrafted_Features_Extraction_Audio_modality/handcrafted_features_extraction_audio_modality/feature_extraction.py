import os
from typing import Any, Dict

from tqdm import tqdm
import numpy as np
import torch
import opensmile
import torchaudio.transforms as transforms


def extract_and_save_features(
        all_data_paths: str,
        destination: str,
        handcrafted_features_configs: Dict[str, Any]
):
    """
    Extract features from pre-processed audio files and save to the destination directory.

    Configs format expected:
        handcrafted_features_configs = {
        "MFCC": {
            ...
        },
        "eGeMAPs": {
            ...
        }
        }

    Args:
        all_data_paths: a list containing the paths to all audio subjects
        destination: destination directory path
        configs: configurations for feature extraction algorithm
    """
    # Create paths
    if "MFCC" in handcrafted_features_configs:
        save_folder_mfcc = os.path.join(destination, "MFCC")
        os.makedirs(save_folder_mfcc, exist_ok=True)
    if "MelSpectrogram" in handcrafted_features_configs:
        save_folder_spectrogram = os.path.join(destination, "MelSpectrogram")
        os.makedirs(save_folder_spectrogram, exist_ok=True)
    if "eGeMAPs" in handcrafted_features_configs:
        save_folder_egemaps = os.path.join(destination, "eGeMAPs")
        os.makedirs(save_folder_egemaps, exist_ok=True)

    # Extract features
    for data_path in tqdm(
        all_data_paths,
        desc=f'Extracting Handcrafted Features: {list(handcrafted_features_configs)}'
    ):
        waveform = np.load(data_path)
        if "MFCC" in handcrafted_features_configs:
            mfcc_to_save = extract_mfcc(waveform, handcrafted_features_configs["MFCC"])
            np.save(os.path.join(save_folder_mfcc, data_path.split(os.path.sep)[-1]), mfcc_to_save)
        if "MelSpectrogram" in handcrafted_features_configs:
            spectrogram_to_save = extract_mel_spectrogram(waveform, handcrafted_features_configs["MelSpectrogram"])
            np.save(os.path.join(save_folder_spectrogram, data_path.split(os.path.sep)[-1]), spectrogram_to_save)
        if "eGeMAPs" in handcrafted_features_configs:
            # By default functional egemaps are extracted (88 features for the whole audio sample)
            smile_egemaps = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            egemaps_to_save = extract_egemaps(smile_egemaps, waveform, handcrafted_features_configs["eGeMAPs"])
            np.save(os.path.join(save_folder_egemaps, data_path.split(os.path.sep)[-1]), egemaps_to_save)


def extract_mfcc(
        input_array: np.ndarray,
        configs: Dict[str, Any],
):
    """
    Extract MFCC from from input numpy array with pre-processed audio.

    Args:
        input_array: input array in numpy format
        configs: configurations for feature extraction algorithm
    """
    waveform = torch.tensor(input_array)
    transform = transforms.MFCC(**configs)
    mfcc = transform(waveform).numpy()
    # Transpose to have chanels last
    return mfcc.T


def extract_mel_spectrogram(
        input_array: np.ndarray,
        configs: Dict[str, Any],
):
    """
    Extract Mel Spectrograms from from input numpy array with pre-processed audio.

    Args:
        input_array: input array in numpy format
        configs: configurations for feature extraction algorithm
    """
    waveform = torch.tensor(input_array)
    transform = transforms.MelSpectrogram(**configs)
    spectrogram = transform(waveform).numpy()
    # Transpose to have chanels last
    return spectrogram.T


def extract_egemaps(
        smile_egemaps,
        input_array: np.ndarray,
        configs: Dict[str, Any],
):
    """
    Extract eGeMAPs features from input numpy array with pre-processed audio.

    Args:
        smile_egemaps: egemaps estimator from opensmile
        input_array: input array in numpy format
        configs: configurations for feature extraction algorithm
    """
    waveform = np.trim_zeros(input_array)
    egemaps = smile_egemaps.process_signal(waveform, **configs)
    # Transpose to have chanels last
    return np.array(egemaps.T)
