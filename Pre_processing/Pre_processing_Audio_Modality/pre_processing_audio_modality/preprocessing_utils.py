import glob
import os
from typing import Any, Dict, List
import pathlib
from tqdm import tqdm

import numpy as np
import scipy


def process_dataset(
        full_dataset_path: str,
        all_subjects_dirs: List,
        pre_processing_cfg: Dict[str, Any],
        outputs_folder: str,
        label_to_emotion: Dict[str, int],
        dataset: str
):
    """
    Preprocesses the wav dataset.
    saves the processed audio in the output directory.

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
    if pre_processing_cfg['create_splits']:
        test_subjects = np.random.choice(all_subjects_dirs, max(1, int(0.1 * len(all_subjects_dirs))), replace=False)
        no_test = [dir_ for dir_ in all_subjects_dirs if dir_ not in test_subjects]
        val_subjects = np.random.choice(no_test, max(1, int(0.1 * len(all_subjects_dirs))), replace=False)
        train_subjects = [dir_ for dir_ in all_subjects_dirs if dir_ not in test_subjects and dir_ not in val_subjects]
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

    preprocessing_to_apply = self_functions[pre_processing_cfg['process']]
    pathlib.Path(
        os.path.join(
            outputs_folder,
            pre_processing_cfg['process']
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
                    pre_processing_cfg['target_sr']
                )

                all_subject_audio.append(resampled_audio)
                loaded_files.append(audio_path)

            all_subject_audio_processed = preprocessing_to_apply(all_subject_audio)

            if pre_processing_cfg['padding']:
                for i, processed_audio in enumerate(all_subject_audio_processed):
                    if len(processed_audio) > (pre_processing_cfg['target_sr'] * pre_processing_cfg['max_length']):
                        processed_audio = (
                            processed_audio[:int(pre_processing_cfg['target_sr'] *
                                                 pre_processing_cfg['max_length'])]
                        )
                    elif len(processed_audio) < (pre_processing_cfg['target_sr'] * pre_processing_cfg['max_length']):
                        temp = np.zeros(
                            (pre_processing_cfg['target_sr'] * pre_processing_cfg['max_length'],)
                        )
                        temp[:len(processed_audio)] = processed_audio
                        processed_audio = temp
                    all_subject_audio_processed[i] = processed_audio

            # iterate over files and save them to the outputs folder
            processed_file_paths = []
            processed_file_labels = []
            for file_name, processed_audio in zip(loaded_files, all_subject_audio_processed):
                filename = os.path.basename(file_name)
                if dataset == "RAVDESS":
                    processed_file_labels.append(label_to_emotion[filename.split('-')[2]])
                else:
                    processed_file_labels.append(label_to_emotion[filename.split('-')[0]])
                filepath = os.path.join(
                    outputs_folder,
                    pre_processing_cfg['process'],
                    filename[:-3] + 'npy'
                )
                processed_file_paths.append(filepath.split(os.sep)[-1])
                np.save(filepath, processed_audio.astype(np.float32))

            split['files'].extend(processed_file_paths)
            split['labels'].extend(processed_file_labels)

    return train_split, val_split, test_split


def normalize(subject_all_audio):
    """
    normalize: transformed into a range between -1 and 1 by normalization for each speaker (min-max scaling)

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_normalized_audio: a list containing the normalized numpy arrays with audio from a subject
    """

    # could be heavily influenced by outliers
    min = np.min(np.hstack(subject_all_audio))
    max = np.max(np.hstack(subject_all_audio))

    subject_all_normalized_audio = [2 * (au - min) / (max - min) - 1 for au in subject_all_audio]

    return subject_all_normalized_audio


def standardize(subject_all_audio):
    """
    z-normalization to zero mean and unit variance for each speaker

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_standardized_audio: a list containing the standardized numpy arrays with audio from a subject
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
    No pre-processing applied

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
    """
    return subject_all_audio


def resample_audio_signal(
        audio: np.ndarray,
        sample_rate: int,
        target_rate: int
):
    """
    Resample the given (audio) signal to a target frequency

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
