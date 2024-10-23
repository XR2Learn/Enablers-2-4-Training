from collections import deque
import glob
import datetime
import os
from typing import Any, Dict, List, Optional, Tuple
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
        seq_len: int = 5,
        overlap: float = 0.,
        frequency: int = 10,
        resample_freq: int = 10,
        use_sensors: Optional[List[str]] = None,
):
    """
    Preprocesses the dataset with bio-measurements in Magic XRoom format.
    Saves the processed audio in the output directory.

    Args:
        full_dataset_path: the path to the full dataset
        all_subjects_dirs: a list containing all subdirectories which is assumed to be all the different subjects
        pre_processing_cfg: configutation for pre-processing
        outputs_folder: path to the outputs folder,
        seq_len: sequence length in seconds
        overlap: overlapping proportion between segments in [0, 1)
        frequency: frequency of the raw signal

    Returns:
        train_split: a dictionary containing the 'files' and 'labels' for training
        val_split: a dictionary containing the 'files' and 'labels' for validation
        test_split: a dictionary containing the 'files' and 'labels' for testing
    """

    train_split = {'files': [], 'labels': []}
    val_split = {'files': [], 'labels': []}
    test_split = {'files': [], 'labels': []}

    get_ssl = pre_processing_cfg["get_ssl"] if "get_ssl" in pre_processing_cfg else False
    get_stats = pre_processing_cfg["get_stats"] if "get_stats" in pre_processing_cfg else False

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
    if get_ssl:
        ssl_train_split = {'files': []}
        ssl_val_split = {'files': []}
        ssl_test_split = {'files': []}
        ssl_splits_phase = {'train': ssl_train_split, 'val': ssl_val_split, 'test': ssl_test_split}
    subjects_phase = {'train': train_subjects, 'val': val_subjects, 'test': test_subjects}

    # get the right function to use, and create path to save files to is doesnt exist
    self_functions = {
        "normalize": normalize,
        'standardize': standardize,
        'raw': no_preprocessing
    }

    preprocessing_to_apply = self_functions[pre_processing_cfg['process']]
    pathlib.Path(
        os.path.join(
            outputs_folder,
            pre_processing_cfg['process']
        )
    ).mkdir(parents=True, exist_ok=True)

    if get_ssl:
        pathlib.Path(
            os.path.join(
                outputs_folder,
                "ssl_" + pre_processing_cfg['process']
            )
        ).mkdir(parents=True, exist_ok=True)

    # go over each phase/split
    ovr_stats = []
    for phase in ['train', 'val', 'test']:
        split = splits_phase[phase]
        if get_ssl:
            ssl_split = ssl_splits_phase[phase]
        subjects = subjects_phase[phase]
        for subject_path in tqdm(subjects, desc=f"Preprocessing {phase} set"):
            subject_path = os.path.join(full_dataset_path, subject_path)
            # format: data_collection_SESSION_SENSOR_.csv
            sessions = set([x.split("_")[2] for x in os.listdir(subject_path)])

            for session in sessions:
                processed_file_paths = []
                if get_ssl:
                    processed_file_paths_ssl = []
                processed_file_labels = []

                session_annot = glob.glob(os.path.join(subject_path, f"*{session}*PROGRESS_EVENT_.csv"))[0]
                session_bm = glob.glob(os.path.join(subject_path, f"*{session}*SHIMMER_.csv"))[0]

                # Assign available labels from annotations to bio-measurement data
                # to obtain dataframe with labeled signals corresponding to multiple levels (intervals)
                processed_session, stats, processed_session_ssl = process_session(
                    session_bm,
                    session_annot,
                    subject_path,
                    session=session,
                    get_ssl=get_ssl,
                    get_stats=get_stats,
                    use_sensors=use_sensors
                )
                if get_stats:
                    if stats:
                        ovr_stats.append(stats)

                # Segment each extracted level into shorter time windows
                # Each level (interval) will be split into multiple segments with the same length
                try:
                    segmented_session, labels = segment_processed_session(
                        processed_session,
                        seq_len,
                        overlap,
                        frequency=frequency
                    )
                except ValueError:
                    segmented_session = None

                if segmented_session is not None:
                    # apply pre-processing (e.g., normalization) for the whole session
                    preprocessed_session = preprocessing_to_apply(segmented_session)

                    for i in range(len(preprocessed_session)):
                        # apply resampling if needed
                        session_to_save = preprocessed_session[i]
                        if resample_freq != frequency:
                            session_to_save = resample_bm(session_to_save, frequency, resample_freq)

                        filepath = os.path.join(
                            outputs_folder,
                            pre_processing_cfg['process'],
                            f"{os.path.basename(subject_path)}_{session}_{i}_emotion_{labels[i]}.npy"
                        )

                        np.save(filepath, session_to_save.astype(np.float32))

                        processed_file_paths.append(filepath.split(os.sep)[-1])
                        processed_file_labels.append(labels[i])

                    split['files'].extend(processed_file_paths)
                    split['labels'].extend(processed_file_labels)

                # repeat the processing for unlabeled ssl data
                if get_ssl:
                    try:
                        segmented_session_ssl = segment_processed_session_ssl(
                            processed_session_ssl,
                            seq_len,
                            overlap,
                            frequency=frequency
                        )
                    except ValueError:
                        segmented_session_ssl = None

                    if segmented_session_ssl is not None:
                        # apply pre-processing (e.g., normalization) for the whole session
                        preprocessed_session_ssl = preprocessing_to_apply(segmented_session_ssl)

                        for i in range(len(preprocessed_session_ssl)):
                            session_to_save = preprocessed_session_ssl[i]
                            if resample_freq != frequency:
                                session_to_save = resample_bm(session_to_save, frequency, resample_freq)

                            filepath = os.path.join(
                                outputs_folder,
                                "ssl_" + pre_processing_cfg['process'],
                                f"{os.path.basename(subject_path)}_{session}_{i}.npy"
                            )

                            np.save(filepath, session_to_save.astype(np.float32))

                            processed_file_paths_ssl.append(filepath.split(os.sep)[-1])

                        ssl_split['files'].extend(processed_file_paths_ssl)

                if segmented_session is None:
                    print(f"""Skipping subject {subject_path} session {session}.
                            Error in pre-processing labeled data: Not enough labeled data""")
                if get_ssl and segment_processed_session_ssl is None:
                    print(f"""Skipping subject {subject_path} session {session}.
                            Error in pre-processing unlabeled data: Not enough unlabeled data""")

    return (
        train_split,
        val_split,
        test_split,
        ovr_stats if get_stats else None,
        ssl_train_split if get_ssl else [],
        ssl_val_split if get_ssl else [],
        ssl_test_split if get_ssl else [],
    )


def process_session(
    session_data_file: str,
    session_annot_file: str,
    subject: str,
    session: str,
    threshold: float = 10,
    offset_hours_data: int = 1,
    get_ssl: bool = False,
    get_stats: bool = False,
    use_sensors: Optional[List[str]] = None,
    cont_to_cat: bool = True,
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
        get_ssl: return unlabeled dataframe together with annotated dataframe
        get_stats: flag for generating stats csv (for labeled data)

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
    if use_sensors is not None:
        save_cols = ["timestamp"]
        save_cols.extend(use_sensors)
        data = data[save_cols]
    # Older version of Magic XRoom collects Shimmer internal UNIX timestamp as 'timestamp' column
    try:
        data["timestamp_dt"] = (
            data["timestamp"]
            .apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))
        )
    # Current version of Magic XRoom uses C# timestamp as 'timestamp' column
    #   and internal UNIX timestamps as 'timestamp_int' column
    except ValueError:
        data["timestamp_dt"] = (
            data["timestamp"]
            .apply(lambda x: datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=x // 10))
        )
        offset_hours_data = 0
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

    # iterate through annotations file to assign labels to level timestamps
    for _, row in annotations.iterrows():
        event_type = row["event_type"]
        info = row["info"]
        if event_type == "LEVEL_STARTED":
            # it is not expected to have LEVEL_STARTED two times in a row
            start_ts = row["timestamp_dt"]

        elif event_type in ["LEVEL_COMPLETED", "LEVEL_FAILED"]:
            # save interval to stack if level_started
            stack_level_ts.append((start_ts, row['timestamp_dt']))

        elif event_type in ["BORED", "ENGAGED", "FRUSTRATED", "SKIP", "FEEDBACK_RECEIVED"]:
            last_finished_level_start, last_finished_level_end = stack_level_ts.pop() if (
                stack_level_ts
            ) else (None, None)
            while stack_level_ts:
                prev_start, prev_end = stack_level_ts.pop()
                if (prev_end - last_finished_level_start).total_seconds() < threshold:
                    last_finished_level_start = prev_start
            # assign label to the latest level interval (from stack) if time difference is not larger than a threhsold
#             print("START-END:", last_finished_level_start, last_finished_level_end)
            if (
                event_type != "SKIP" and
                last_finished_level_end is not None and
                (row["timestamp_dt"] - last_finished_level_end).total_seconds() < threshold
            ):  
                if event_type != "FEEDBACK_RECEIVED":
                    label = event_type
                else:
                    label = continious_to_categorical(info) if cont_to_cat else info
                labeled_intervals.append(
                    (
                        last_finished_level_start,
                        last_finished_level_end,
                        label,
                        row["timestamp_dt"]
                    )
                )

    sensor_first_entry = min(data['timestamp_dt'])
    sensor_last_entry = max(data['timestamp_dt'])
    progress_event_first_entry = min(annotations['timestamp_dt']) if annotations.shape[0] > 0 else None
    progress_event_last_entry = max(annotations['timestamp_dt']) if annotations.shape[0] > 0 else None

    start, end, label, _ = labeled_intervals.popleft() if labeled_intervals else (None, None, None, None)
    interval_num = 1

    # iterate through data (bio-measurements) to assign labels based on intervals
    if None not in [start, end]:
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
    stats = {}
    if get_stats and not labeled_data.empty:
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
        min_max_intervals["interval_length"] = min_max_intervals["interval_length"].dt.total_seconds()
        length_of_labeled_data = min_max_intervals["interval_length"].sum()

        length_per_min_max_intervals = (
            min_max_intervals
            .groupby(by=["label_min"])
            .sum(numeric_only=True)
            .reset_index()
        )

        lengths = {
            "length_seconds_BORED": pd.Timedelta(0),
            "length_seconds_ENGAGED": pd.Timedelta(0),
            "length_seconds_FRUSTRATED": pd.Timedelta(0)
        }

        for _, row in length_per_min_max_intervals.iterrows():
            lengths[f"length_seconds_{row['label_min']}"] = row["interval_length"]

        stats = {
            "subject": os.path.basename(subject),
            "session": session,
            "sensor_session_length": sensor_last_entry - sensor_first_entry,
            "progress_event_length": progress_event_last_entry - progress_event_first_entry,
            "avg_actual_frequency": data.shape[0] / int((sensor_last_entry - sensor_first_entry).total_seconds()),
            "progress_event_first_entry": progress_event_first_entry,
            "progress_event_last_entry": progress_event_last_entry,
            "sensor_first_entry": sensor_first_entry,
            "sensor_last_entry": sensor_last_entry,
            "labeled_sensor_data_pct": round(labeled_data.shape[0] / data.shape[0], 2),
            "num_labeled_intervals_seconds": len(labeled_data["interval_num"].unique()),
            "length_labeled_intervals": length_of_labeled_data,
        }
        stats = {**stats, **lengths}

    return (
        labeled_data,
        stats if get_stats else None,
        data if get_ssl else None
    )


def segment_processed_session(
    session_df: pd.DataFrame,
    seq_len: int,
    overlap: float,
    frequency: float = 10
) -> Tuple[np.ndarray, List[str]]:
    """
    Segmenting processed sessions into time windows of the provided sequence length in seconds using labeled intervals

    Args:
        session_df: Dataframe obtained after calling process_session()
        seq_len: lengths of sequences (time windows) in seconds
        overlap: proportion of overlap between time windows in [0, 1)
        frequency: (expected) frequency of the signal
    """
    window_length = int(seq_len * frequency)
    intervals = session_df["interval_num"].unique()
    segmented_session = []
    labels = []
    for interval in intervals:
        interval_data = session_df[session_df["interval_num"] == interval]
        unique_labels = interval_data["label"].unique()
        if len(unique_labels) > 1:
            raise ValueError("Found multiple labels per interval")
        label = unique_labels[0]
        drop_cols = ["index", "timestamp", "updated_timestamp", "timestamp_dt", "label", "interval_num"]
        interval_data_sensors = np.array(
            interval_data
            .drop([x for x in drop_cols if x in interval_data.columns], axis=1)
        )

        for i in range(0, len(interval_data_sensors) - window_length, int(window_length * (1 - overlap))):
            curr_window = interval_data_sensors[i: i + window_length]
            segmented_session.append(curr_window)
            labels.append(label)
    
    return np.stack(segmented_session), labels


def segment_processed_session_ssl(
    session_df: pd.DataFrame,
    seq_len: int,
    overlap: float,
    frequency: float = 10
) -> Tuple[np.ndarray, List[str]]:
    """
    Segmenting processed sessions into time windows of the provided sequence length in seconds for ssl data

    Args:
        session_df: Dataframe obtained after calling process_session()
        seq_len: lengths of sequences (time windows) in seconds
        overlap: proportion of overlap between time windows in [0, 1)
        frequency: (expected) frequency of the signal
    """
    window_length = int(seq_len * frequency)
    segmented_session = []
    drop_cols = ["index", "timestamp", "updated_timestamp", "timestamp_dt", "label", "interval_num"]
    session_data_sensors = np.array(
        session_df
        .drop([x for x in drop_cols if x in session_df.columns], axis=1)
    )

    for i in range(0, len(session_data_sensors) - window_length, int(window_length * (1 - overlap))):
        curr_window = session_data_sensors[i: i + window_length]
        segmented_session.append(curr_window)

    return np.stack(segmented_session)


def normalize(bm_segments):
    """
    normalize: transformed into a range between -1 and 1 by normalization for each speaker (min-max scaling)

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_normalized_audio: a list containing the normalized numpy arrays with audio from a subject
    """

    # stack segments for the whole session and compute per-channel statistics
    stacked_segments = bm_segments.reshape(-1, bm_segments.shape[-1])
    min_channel = stacked_segments.min(axis=0)
    max_channel = stacked_segments.max(axis=0)

    segments_min_max = (bm_segments - min_channel) / (max_channel - min_channel)

    return segments_min_max


def standardize(bm_segments: np.ndarray):
    """
    z-normalization to zero mean and unit variance for each segment with bio-measurements

    Args:
        bm_segment: 3D-array containing bio-measurement signals from one session (multiple segments per session)

    Returns:
        standardized_signal: a list containing the standardized numpy arrays with audio from a subject
    """

    # stack segments for the whole session and compute per-channel statistics
    stacked_segments = bm_segments.reshape(-1, bm_segments.shape[-1])
    mean_channel = stacked_segments.mean(axis=0)
    std_channel = stacked_segments.std(axis=0)

    bm_z_normalized = (bm_segments - mean_channel) / std_channel

    return bm_z_normalized


def no_preprocessing(bm_segment):
    """
    No pre-processing applied

    Args:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject

    Returns:
        subject_all_audio: a list containing the numpy arrays with all the audio data of the subject
    """
    return bm_segment


def resample_bm(
        bm_segment: np.ndarray,
        sample_rate: int,
        target_rate: int
):
    """
    Resample stacked signals to a target frequency

    Args:
        bm_segment: 3D numpy array with the bio-measurement data to resample
        sample_rate: the sample rate of the original signal
        target_rate: the target sample rate
    Returns:
        resampled: resampled signals
    """
    number_of_samples = round(len(bm_segment) * float(target_rate) / sample_rate)
    resampled = scipy.signal.resample(bm_segment, number_of_samples, axis=0)
    return resampled


def continious_to_categorical(info: str, categories=["BORED", "ENGAGED", "FRUSTRATED"]) -> str:
    """
    Maps a continuous value in the range [0, 1] to a discrete category.

    Args:
        info: A string containing a float value between 0 and 1.
        categories: Categories to map continuous values to.

    """
    num_categories = len(categories)
    try:
        value = float(info)
        if not 0 <= value <= 1:
            raise ValueError("Value must be between 0 and 1")
        
        category_size = 1. / num_categories
        category = categories[min(int(value // category_size), num_categories - 1)]
        
        return str(category)
    except ValueError:
        print(f"Invalid input: {info}. Expected a float between 0 and 1.")
        return "INVALID"