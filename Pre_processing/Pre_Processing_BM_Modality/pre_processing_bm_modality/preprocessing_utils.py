from collections import deque
import glob
import datetime
import os
from typing import Any, Dict, List, Tuple
import pathlib
import pandas as pd
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
    Preprocesses the dataset with bio-measurements in Magic XRoom format.
    Saves the processed audio in the output directory.

    Args:
        full_dataset_path: the path to the full dataset
        all_subjects_dirs: a list containing all subdirectories which is assumed to be all the different subjects
        pre_processing_cfg: configutation for pre-processing
        outputs_folder: path to the outputs folder,
        label_to_emotion: mapping from labels to emotions,
        dataset: name of the dataset

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
    ovr_stats = []
    for phase in ['train', 'val', 'test']:
        split = splits_phase[phase]
        subjects = subjects_phase[phase]
        for subject_path in tqdm(subjects, desc=f"Preprocessing {phase} set"):
            subject_path = os.path.join(full_dataset_path, subject_path)
            # format: data_collection_SESSION_SENSOR_.csv
            sessions = set([x.split("_")[2] for x in os.listdir(subject_path)])
            
            for session in sessions:
                processed_file_paths = []
                processed_file_labels = []

                session_annot = glob.glob(os.path.join(subject_path, f"*{session}*PROGRESS_EVENT_.csv"))[0]
                session_bm = glob.glob(os.path.join(subject_path, f"*{session}*SHIMMER_.csv"))[0]

                processed_session, stats = process_session(
                    session_bm,
                    session_annot,
                    subject_path,
                    session=session
                )
                ovr_stats.append(stats)
                processed_session.to_csv(f"./{os.path.basename(subject_path)}_{session}.csv", index=None)

                # # (data, label)
                # for i, instance in enumerate(processed_session):
                #     filepath = os.path.join(
                #         outputs_folder,
                #         pre_processing_cfg['process'],
                #         f"{session}_{i}_emotion{instance[1]}.npy"
                #     )
                #     preprocessed_instance = preprocessing_to_apply(instance[0])
                #     np.save(filepath, preprocessed_instance.astype(np.float32))

                #     processed_file_paths.append(filepath.split(os.sep)[-1])
                #     processed_file_labels.append(instance[1])
                    
                # split['files'].extend(processed_file_paths)
                # split['labels'].extend(processed_file_labels)

    pd.DataFrame(ovr_stats).sort_values(by=["subject", "session"]).to_csv("./stats_xroom_shimmer.csv", index=None)
    return train_split, val_split, test_split


def process_session(
    session_data_file: str,
    session_annot_file: str,
    subject: str,
    session: str,
    threshold: float = 10,
    offset_hours_data: int = 1
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts session data from Shimmer and Progress Events from Magic XRoom and assigns labels to sensor recordings.
    Collects summary about the session.

    Args:
        session_data_file: path to bio-measurements recordings from the session
        session_annot_file: path to annotations
        subject: subject ID
        session: session ID
        threshold: threshold in seconds to discard irrelevant labels and levels,
            i.e. if the time passed between annotation and the latest completed/failed level is higher than
            this threshold, both annotation and level are discarded
        offset_hours_data: delay in Shimmer data recording in hours caused by different time-zones

    Returns:
        labeled_data: dataframe consisting labeled sensor recordings
        stats: dictionary with stats describing the input session
    """
    # Timestamp format: C# ticks
    annotations = pd.read_csv(session_annot_file)
    annotations["timestamp_dt"] = (
        annotations["timestamp"]
        .apply(lambda x: datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=x // 10))
    )
    annotations = (
        annotations
        .sort_values(by="timestamp_dt")
        .reset_index()
    )

    # Timestamp format: Unix
    data = pd.read_csv(session_data_file)
    data["timestamp_dt"] = (
        data["timestamp"]
        .apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))
    )
    if offset_hours_data >= 1:
        data['timestamp_dt'] += pd.Timedelta(hours=offset_hours_data)
    data = data.drop_duplicates()
    data["label"] = np.nan
    data["interval_num"] = np.nan
    data = (
        data
        .sort_values(by="timestamp_dt")
        .reset_index()
    )

    stack_level_ts = []
    labeled_intervals = deque()

    level_started = False
    # iterate through annotations file to assign labels to level timestamps
    for _, row in annotations.iterrows():
        event_type = row["event_type"]
        if event_type == "LEVEL_STARTED":
            level_started = True
            # it is not expected to have LEVEL_STARTED two times in a row
            start_ts = row["timestamp_dt"]

        elif event_type in ["LEVEL_COMPLETED", "LEVEL_FAILED"]:
            # save interval to stack if level_started
            if level_started:
                stack_level_ts.append((start_ts, row['timestamp_dt']))
                level_started = False

        elif event_type in ["BORED", "ENGAGED", "FRUSTRATED", "SKIP"]:
            last_finished_level_start, last_finished_level_end = stack_level_ts.pop()
            # assign label to the latest level interval (from stack) if time difference is not larger than a threhsold
            if event_type != "SKIP" and (row["timestamp_dt"] - last_finished_level_end).total_seconds() < threshold:
                labeled_intervals.append(
                    (last_finished_level_start, last_finished_level_end, event_type, row["timestamp_dt"])
                )

    shimmer_first_entry = min(data['timestamp_dt'])
    shimmer_last_entry = max(data['timestamp_dt'])
    progress_event_first_entry = min(annotations['timestamp_dt'])
    progress_event_last_entry = max(annotations['timestamp_dt'])

    start, end, label, _ = labeled_intervals.popleft()
    interval_num = 1

    # iterate through data (bio-measurements) to assign labels based on intervals
    for idx, row in data.iterrows():
        if start <= row["timestamp_dt"] <= end:
            data.at[idx, "label"] = label
            data.at[idx, "interval_num"] = interval_num
        elif end < row['timestamp_dt']:
            if not labeled_intervals:
                break
            start, end, label, _ = labeled_intervals.popleft()
            interval_num += 1

    # query labeled data 
    labeled_data = data[~data["label"].isna()]

    # compute length of recorded labeled data and each emotion
    min_max_intervals = (
        labeled_data[["interval_num", "timestamp_dt", "label"]]
        .groupby(by="interval_num")
        .agg([np.min, np.max])
        .reset_index(drop=True)
    )
    min_max_intervals.columns = min_max_intervals.columns.map('_'.join).str.strip('_')
    min_max_intervals = min_max_intervals[["timestamp_dt_min", "timestamp_dt_max", "label_min"]]
    min_max_intervals["interval_length"] = (
        min_max_intervals["timestamp_dt_max"] - min_max_intervals["timestamp_dt_min"]
    )

    length_of_labeled_data = min_max_intervals["interval_length"].sum()

    length_per_min_max_intervals = (
        min_max_intervals
        .groupby(by=["label_min"])
        .sum()
        .reset_index()
    )

    lengths = {
        "length_BORED": pd.Timedelta(0),
        "length_ENGAGED": pd.Timedelta(0),
        "length_FRUSTRATED": pd.Timedelta(0)
    }

    for _, row in length_per_min_max_intervals.iterrows():
        lengths[f"length_{row['label_min']}"] = row["interval_length"]

    stats = {
        "subject": os.path.basename(subject),
        "session": session,
        "shimmer_session_length": shimmer_last_entry - shimmer_first_entry,
        "progress_event_length": progress_event_last_entry - progress_event_first_entry,
        "avg_actual_frequency": data.shape[0] / int((shimmer_last_entry - shimmer_first_entry).total_seconds()),
        "progress_event_first_entry": progress_event_first_entry,
        "progress_event_last_entry": progress_event_last_entry,
        "shimmer_first_entry": shimmer_first_entry,
        "shimmer_last_entry": shimmer_last_entry,
        "labeled_shimmer_data_pct": round(labeled_data.shape[0] / data.shape[0], 2),
        "num_labeled_intervals": len(labeled_data["interval_num"].unique()),
        "length_labeled_intervals": length_of_labeled_data,
    }
    stats = {**stats, **lengths}
    print(stats)
    return labeled_data, stats


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
