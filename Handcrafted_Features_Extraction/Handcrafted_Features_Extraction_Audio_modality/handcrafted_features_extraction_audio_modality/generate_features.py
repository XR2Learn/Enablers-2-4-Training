# Python code here
import os
import numpy as np
import pathlib
import torch
import opensmile
import torchaudio.transforms as transforms

from tqdm import tqdm
from conf import CUSTOM_SETTINGS, OUTPUTS_FOLDER


# TODO: add comments and dockstrings etc

def extract_handcrafted_features():
    """
 
    Args:
        none
 
    Returns:
        none
    """
    processed_data_folder = os.path.join(OUTPUTS_FOLDER,
                                         CUSTOM_SETTINGS['pre_processing_config']['process'])

    all_data_paths = os.listdir(processed_data_folder)
    print(f"Found a total of {len(all_data_paths)} files inside the {processed_data_folder} folder.")
    features_to_extract = CUSTOM_SETTINGS["handcrafted_features_config"].keys()

    print(f"Features {features_to_extract}, if supported, will be extracted and saved.")

    full_data_paths = [os.path.join(processed_data_folder, data_path) for data_path in all_data_paths]

    if "MFCC" in features_to_extract:
        save_folder = os.path.join(OUTPUTS_FOLDER, 'MFCC')
        pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        extract_and_save_MFCC(full_data_paths, save_folder)

    if "eGeMAPs" in features_to_extract:
        save_folder = os.path.join(OUTPUTS_FOLDER, 'eGeMAPs')
        pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        extract_and_save_egemaps(full_data_paths, save_folder)

    if "Spectogram" in features_to_extract:
        save_folder = os.path.join(OUTPUTS_FOLDER, 'Spectograms')
        pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        extract_and_save_spectogram(full_data_paths, save_folder)
        # extract_and_save_MFCC(all_data_paths)


def extract_and_save_MFCC(all_data_paths, save_folder):
    """
    extract_and_save_MFCC

    Args:
        all_data_paths: a list containing the paths to all audio subjects
 
    Returns:
        subject_all_normalized_audio: a list containing the normalized numpy arrays with all the audio data of the subject
    """
    for data_path in tqdm(all_data_paths, desc='Extracting MFCCs'):
        waveform = torch.tensor(np.load(data_path))
        transform = transforms.MFCC(
             **CUSTOM_SETTINGS["handcrafted_features_config"]['MFCC']
        )
        mfcc = transform(waveform).numpy()
        np.save(os.path.join(save_folder, data_path.split(os.path.sep)[-1]), mfcc)


def extract_and_save_spectogram(all_data_paths, save_folder):
    """
    extract_and_save_spectogram

    Args:
        all_data_paths: a list containing the paths to all audio subjects
 
    Returns:
        subject_all_normalized_audio: a list containing the normalized numpy arrays with all the audio data of the subject
    """
    for data_path in tqdm(all_data_paths, desc='Extracting Spectograms'):
        waveform = torch.tensor(np.load(data_path))
        #TODO: add custom settings
        transform = transforms.Spectrogram(n_fft=400)
        spectogram = transform(waveform).numpy()
        np.save(os.path.join(save_folder, data_path.split(os.path.sep)[-1]), spectogram)


def extract_and_save_egemaps(all_data_paths, save_folder):
    smile_egemaps = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
    )
    for data_path in tqdm(all_data_paths, desc='Extracting eGeMAPs'):
        waveform = np.load(data_path)
        # TODO: double check if the padding actually gets trimmed
        # TODO: check if egemaps need standardized N(mean=0,std=1) or normalized[-1,1] audio
        waveform = np.trim_zeros(waveform)
        egemaps = smile_egemaps.process_signal(waveform, **CUSTOM_SETTINGS["handcrafted_features_config"]['eGeMAPs'])
        np.save(os.path.join(save_folder, data_path.split(os.path.sep)[-1]), egemaps.T)


if __name__ == '__main__':
    extract_handcrafted_features()
