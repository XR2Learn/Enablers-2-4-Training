# Python code here
import glob
import os
import pathlib

import numpy as np
import pandas as pd

import scipy
from tqdm import tqdm

from conf import (
    CUSTOM_SETTINGS,
    RAVDESS_LABEL_TO_EMOTION,
    DATASETS_FOLDER,
    OUTPUTS_FOLDER
)
from download_datasets import download_RAVDESS
from preprocessing_utils import (
    resample_audio_signal,
    no_preprocessing,
    normalize,
    standardize
)


def preprocess():
    """
    Preprocesses the wav dataset defined inside the configuration using the given parameters.
    The resulting splits will be stored in outputs as a train/test/val csv.
    The resulting wav files will be stored under outputs in a folder named by the preprocessing used.

    When using a custom or collected dataset it is expected that the dataset is organized as follows:
    .
    └── DATASETS_FOLDER/
        └── custom_dataset/
            ├── subject1/
            │   ├── audio1.wav
            │   ├── audio2.wav
            │   └── ...
            ├── subject2
            ├── ...
            └── subjectN
    """
    dataset_name = CUSTOM_SETTINGS['dataset_config']['dataset_name']
    full_dataset_path = os.path.join(DATASETS_FOLDER, dataset_name)

    # check if dataset folder exist, if not : download and create, forn ow only RAVDESS supported
    if not os.path.isdir(full_dataset_path):
        if dataset_name == "RAVDESS":
            download_RAVDESS()
        else:
            raise ValueError("Unknown dataset name for downloading.")
    else:
        print(f"{dataset_name} folder exists at {full_dataset_path}, will use available data")

    all_subject_dirs = os.listdir(full_dataset_path)
    print(f"Found a total of {len(all_subject_dirs)} subjects inside the {dataset_name} dataset")

    train_split, val_split, test_split = process_dataset(full_dataset_path, all_subject_dirs)

    print('Writing CSV files containing the splits to storage')
    pd.DataFrame.from_dict(train_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'train.csv'))
    pd.DataFrame.from_dict(val_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'val.csv'))
    pd.DataFrame.from_dict(test_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'test.csv'))


def process_dataset(full_dataset_path, all_subjects_dirs):
    """
    Preprocesses the wav dataset.
    saves the processed audio in the output

    Args:
        full_dataset_path: the path to the full dataset
        all_subjects_dirs: a list containing all subdirectories which is assumed to be all the different subjects

    Returns:
        train_split: a dictionary containing the 'files' and 'labels' for training
        val_split: a dictionary containing the 'files' and 'labels' for validation
        test_split: a dictionary containing the 'files' and 'labels' for testing
    """

    train_split = {'files': [], 'labels': []}
    val_split = {'files': [], 'labels': []}
    test_split = {'files': [], 'labels': []}

    # check if datasets has to be split in a 80/10/10 way, else put every subject in training
    if CUSTOM_SETTINGS['pre_processing_config']['create_splits']:
        train_subjects = all_subjects_dirs[int(0.1 * len(all_subjects_dirs)):-int(0.1 * len(all_subjects_dirs))]
        val_subjects = all_subjects_dirs[:int(0.1 * len(all_subjects_dirs))]
        test_subjects = all_subjects_dirs[-int(0.1 * len(all_subjects_dirs)):]
    else:
        train_subjects = all_subjects_dirs
        val_subjects = []
        test_subjects = []

    print('train: ', train_subjects)
    print('val: ', val_subjects)
    print('test: ', test_subjects)

    splits_phase = {'train': train_split, 'val': val_split, 'test': test_split}
    subjects_phase = {'train': train_subjects, 'val': val_subjects, 'test': test_subjects}

    # get the right function to use, and create path to save files to is doesnt exist
    self_functions = {
        "normalize": normalize,
        'standardize': standardize,
        'only_resample': no_preprocessing
    }
    preprocessing_to_apply = self_functions[CUSTOM_SETTINGS['pre_processing_config']['process']]
    pathlib.Path(
        os.path.join(
            OUTPUTS_FOLDER,
            CUSTOM_SETTINGS['pre_processing_config']['process']
        )
    ).mkdir(parents=True, exist_ok=True)

    # go over each phase/split
    for phase in ['train', 'val', 'test']:
        split = splits_phase[phase]
        subjects = subjects_phase[phase]
        for subject_path in tqdm(subjects, desc=f"Preprocessing {phase} set"):
            all_subject_audio_files = glob.glob(os.path.join(full_dataset_path, subject_path, '*.wav'))
            all_subject_audio = []
            loaded_files = []
            for audio_path in all_subject_audio_files:
                sr, audio = scipy.io.wavfile.read(audio_path)

                # check for multi-channel audio
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                resampled_audio = resample_audio_signal(
                    audio,
                    sr,
                    CUSTOM_SETTINGS['pre_processing_config']['target_sr']
                )

                all_subject_audio.append(resampled_audio)
                loaded_files.append(audio_path)

            all_subject_audio_processed = preprocessing_to_apply(all_subject_audio)

            if CUSTOM_SETTINGS['pre_processing_config']['padding']:
                for i, processed_audio in enumerate(all_subject_audio_processed):
                    if len(processed_audio) > (CUSTOM_SETTINGS['pre_processing_config']['target_sr'] *
                                               CUSTOM_SETTINGS['pre_processing_config']['max_length']):
                        processed_audio = processed_audio[:int(CUSTOM_SETTINGS['pre_processing_config']['target_sr'] *
                                                               CUSTOM_SETTINGS['pre_processing_config']['max_length'])]
                    elif len(processed_audio) < (CUSTOM_SETTINGS['pre_processing_config']['target_sr'] *
                                                 CUSTOM_SETTINGS['pre_processing_config']['max_length']):
                        temp = np.zeros((CUSTOM_SETTINGS['pre_processing_config']['target_sr'] *
                                         CUSTOM_SETTINGS['pre_processing_config']['max_length'],))
                        temp[:len(processed_audio)] = processed_audio
                        processed_audio = temp
                    all_subject_audio_processed[i] = processed_audio

            # iterate over files and save them to the outputs folder
            processed_file_paths = []
            processed_file_labels = []
            for file_name, processed_audio in zip(loaded_files, all_subject_audio_processed):
                filename = '_'.join(file_name.split(os.sep)[-3:])
                processed_file_labels.append(RAVDESS_LABEL_TO_EMOTION[file_name.split('-')[2]])
                filepath = os.path.join(
                    OUTPUTS_FOLDER, 
                    CUSTOM_SETTINGS['pre_processing_config']['process'],
                    filename[:-3] + 'npy'
                )
                processed_file_paths.append(filepath.split(os.sep)[-1])
                np.save(filepath, processed_audio.astype(np.float32))

            split['files'].extend(processed_file_paths)
            split['labels'].extend(processed_file_labels)

    return train_split, val_split, test_split


if __name__ == '__main__':
    preprocess()
