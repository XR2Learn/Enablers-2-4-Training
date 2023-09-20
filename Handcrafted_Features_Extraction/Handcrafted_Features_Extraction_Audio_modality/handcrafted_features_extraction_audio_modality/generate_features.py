# Python code here
import os
import numpy as np
import pathlib
import scipy
import torch
import opensmile
import torchaudio
import torchaudio.transforms as transforms

from tqdm import tqdm
from conf import CUSTOM_SETTINGS,OUTPUTS_FOLDER

#TODO: add comments and dockstrings etc

def extract_handcrafted_features():
    """
 
    Args:
        none
 
    Returns:
        none
    """
    processed_data_folder = os.path.join(OUTPUTS_FOLDER,'pre-processing-audio',CUSTOM_SETTINGS['pre_processing_config']['process'])

    all_data_paths = os.listdir(processed_data_folder)
    print(f"found a total of {len(all_data_paths)} files inside the {processed_data_folder} folder")
    features_to_extract = CUSTOM_SETTINGS["handcrafted_features_config"]["features_to_extract"]
    print(f"the features that will be extracted, if supported, and saved will be: {features_to_extract}")

    full_data_paths = [os.path.join(processed_data_folder,data_path) for data_path in all_data_paths]

    if "MFCC" in features_to_extract:
        save_folder = os.path.join(OUTPUTS_FOLDER,'handcrafted-features-generation-audio','MFCC')
        pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        extract_and_save_MFCC(full_data_paths,save_folder)

    if "eGeMAPs" in features_to_extract:
        save_folder = os.path.join(OUTPUTS_FOLDER,'handcrafted-features-generation-audio','eGeMAPs')
        pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        extract_and_save_egemaps(full_data_paths,save_folder)

    if "Spectogram" in features_to_extract:
        save_folder = os.path.join(OUTPUTS_FOLDER,'handcrafted-features-generation-audio','Spectograms')
        pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        extract_and_save_spectogram(full_data_paths,save_folder)
        #extract_and_save_MFCC(all_data_paths)


    
def extract_and_save_MFCC(all_data_paths,save_folder):
    """
    extract_and_save_MFCC

    Args:
        all_data_paths: a list containing the paths to all audio subjects
 
    Returns:
        subject_all_normalized_audio: a list containing the normalized numpy arrays with all the audio data of the subject
    """
    for data_path in tqdm(all_data_paths,desc='Extracting MFCCs'):
        waveform, sample_rate = torchaudio.load(data_path)
        transform = transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False})
        mfcc = transform(waveform).numpy()
        np.save(os.path.join(save_folder,data_path.split(os.path.sep)[-1][:-4]),mfcc)

def extract_and_save_spectogram(all_data_paths,save_folder):
    """
    extract_and_save_spectogram

    Args:
        all_data_paths: a list containing the paths to all audio subjects
 
    Returns:
        subject_all_normalized_audio: a list containing the normalized numpy arrays with all the audio data of the subject
    """
    for data_path in tqdm(all_data_paths,desc='Extracting Spectograms'):
        waveform, sample_rate = torchaudio.load(data_path)
        transform = transforms.Spectrogram(n_fft=400)
        spectogram = transform(waveform).numpy()
        np.save(os.path.join(save_folder,data_path.split(os.path.sep)[-1][:-4]),spectogram)

def extract_and_save_egemaps(all_data_paths,save_folder):
    for data_path in tqdm(all_data_paths,desc='Extracting eGeMAPs'):
        sr,audio = scipy.io.wavfile.read(data_path)
        smile_egemaps = opensmile.Smile(
                            feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals,
                        )
        #TODO: double check if the padding actually gets trimmed
        #TODO: check if egemaps need standardized N(mean=0,std=1) or normalized[-1,1] audio
        audio = np.trim_zeros(audio)
        egemaps = smile_egemaps.process_signal(audio, sr)
        np.save(os.path.join(save_folder,data_path.split(os.path.sep)[-1][:-4]),egemaps)


if __name__ == '__main__':
    extract_handcrafted_features()
