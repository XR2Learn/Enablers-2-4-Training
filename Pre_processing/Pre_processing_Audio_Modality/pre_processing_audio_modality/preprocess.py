# Python code here
import os
import requests
import zipfile
import io
import glob
import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm
# for now i choose scipy as it offers a lot  without having to install additional libraries but maybe librosa can also be an option
import scipy
from conf import CUSTOM_SETTINGS, RAVDESS_EMOTION_TO_LABEL, RAVDESS_LABEL_TO_EMOTION, MAIN_FOLDER, DATASETS_FOLDER, \
    OUTPUTS_FOLDER


# TODO: add comments and dockstrings etc

def preprocess():
    """
    Preprocesses the wav dataset defined inside the configuration using the given parameters.
    The resulting splits will be stored in outputs as a train/test/val csv.
    The resulting wav files will be stored under outputs in a folder named by the preprocessing used
 
    Args:
        none
 
    Returns:
        none
    """
    dataset_name = CUSTOM_SETTINGS['dataset_config']['dataset_name']
    full_dataset_path = os.path.join(DATASETS_FOLDER, dataset_name)

    # check if dataset folder exist, if not : download and create, forn ow only RAVDESS supported
    if not os.path.isdir(full_dataset_path):
        print(f"No existing {dataset_name} folder found, download will start")
        zip_file_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
        r = requests.get(zip_file_url, stream=True)
        progress_bar = tqdm(total=int(r.headers.get('content-length', 0)), unit='B', unit_scale=True,
                            desc='Download progress of RAVDESS dataset')
        dat = b''.join(x for x in r.iter_content(chunk_size=16384) if progress_bar.update(len(x)) or True)
        z = zipfile.ZipFile(io.BytesIO(dat))
        z.extractall(full_dataset_path)
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

    # check if datasets has to be split in a 60/20/20 way, else put every subject in training 
    if CUSTOM_SETTINGS['pre_processing_config']['create_splits']:
        train_subjects = all_subjects_dirs[int(0.2 * len(all_subjects_dirs)):-int(0.2 * len(all_subjects_dirs))]
        val_subjects = all_subjects_dirs[:int(0.2 * len(all_subjects_dirs))]
        test_subjects = all_subjects_dirs[-int(0.2 * len(all_subjects_dirs)):]
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
    self_functions = {"normalize": normalize, 'standardize': standardize, 'only_resample': no_preprocessing}
    preprocessing_to_apply = self_functions[CUSTOM_SETTINGS['pre_processing_config']['process']]
    pathlib.Path(os.path.join(OUTPUTS_FOLDER, CUSTOM_SETTINGS['pre_processing_config']['process'])).mkdir(parents=True,
                                                                                                          exist_ok=True)

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

                # TODO: check if resampling should happen before or after standardization or not
                resampled_audio = resample_audio_signal(audio, sr,
                                                        CUSTOM_SETTINGS['pre_processing_config']['target_sr'])
                # check if resampling rate is as desired by looking at ration between origin/target +/- tolerance
                assert ((sr / CUSTOM_SETTINGS['pre_processing_config']['target_sr']) - 0.1) < (
                        len(audio) / len(resampled_audio)) < ((sr / CUSTOM_SETTINGS['pre_processing_config'][
                    'target_sr']) + 0.1), f"{audio_path} was not resampled correctly"

                all_subject_audio.append(resampled_audio)
                loaded_files.append(audio_path)

            all_subject_audio_processed = preprocessing_to_apply(all_subject_audio)

            # check if padding or cutting is necessary, has to be doen after processing to not unfluence min/max or mean/std
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
                processed_file_labels.append(RAVDESS_LABEL_TO_EMOTION[file_name.split('-')[3]])
                filepath = os.path.join(OUTPUTS_FOLDER, CUSTOM_SETTINGS['pre_processing_config']['process'],
                                        filename[:-3] + 'npy')
                processed_file_paths.append(filepath.split(os.sep)[-1])
                np.save(filepath, processed_audio.astype(np.float32))

            split['files'].extend(processed_file_paths)
            split['labels'].extend(processed_file_labels)

    return train_split, val_split, test_split


def normalize(subject_all_audio):
    """
    normalize : transformed into a range between -1 and 1 by normalization for each speaker

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
 
    Returns:
        subject_all_normalized_audio: a list containing the normalized numpy arrays with all the audio data of the subject
    """

    # could be heavily influenced by outliers
    min = np.min(np.hstack(subject_all_audio))
    max = np.max(np.hstack(subject_all_audio))

    subject_all_normalized_audio = [2 * (au - min) / (max - min) - 1 for au in subject_all_audio]

    # check if min/max values are within a given tolerance
    assert np.min(np.hstack(subject_all_normalized_audio)) > -1.05, "min is smaller than -1"
    assert np.max(np.hstack(subject_all_normalized_audio)) < 1.05, "max is bigger than -1"

    return subject_all_normalized_audio


def standardize(subject_all_audio):
    """
    normalize : divided by the standard deviation after the mean has been subtracted 0 mean, unit variance for each speaker

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
 
    Returns:
        subject_all_standardized_audio: a list containing the standardized numpy arrays with all the audio data of the subject
    """

    # as the encoding is float32, could maybe cause under/overflow?
    mean = np.mean(np.hstack(subject_all_audio))
    std = np.std(np.hstack(subject_all_audio))
    subject_all_standardized_audio = [(au - mean) / std for au in subject_all_audio]

    # check if mean is 0+-tol and std 1+-tol
    assert -0.05 < np.mean(np.hstack(subject_all_standardized_audio)) < 0.05, "mean is not equal to 0"
    assert 0.95 < np.std(np.hstack(subject_all_standardized_audio)) < 1.05, "std is not equal to 1"

    return subject_all_standardized_audio


def no_preprocessing(subject_all_audio):
    """
    no_preprocessing : don't apply any pre-processing and return audio as is (resampled)

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
 
    Returns:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
    """
    return subject_all_audio


def resample_audio_signal(audio, sample_rate, target_rate):
    """
    resample_audio_signal : resample the given (audio) signal to a target frequency

    Args:
        audio: a numpy array with the audio data to resample
        sample_rate: the sample rate of the original signal
        target_rate: the target sample rate
    Returns:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
    """

    number_of_samples = round(len(audio) * float(target_rate) / sample_rate)
    resampled_audio = scipy.signal.resample(audio, number_of_samples)
    return resampled_audio


if __name__ == '__main__':
    preprocess()
