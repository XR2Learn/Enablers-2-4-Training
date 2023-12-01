import os
import pandas as pd

from conf import (
    CUSTOM_LABEL_TO_EMOTION,
    CUSTOM_SETTINGS,
    RAVDESS_LABEL_TO_EMOTION,
    DATASETS_FOLDER,
    OUTPUTS_FOLDER
)
from download_datasets import download_RAVDESS
from preprocessing_utils import process_dataset


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
            │   ├── emotionlabel-audio1.wav
            │   ├── emotionlabel-audio2.wav
            │   └── ...
            ├── subject2
            ├── ...
            └── subjectN
    Each wavfile should be named in the following format to correctly parse the labels: emotionlabel-audio.wav
    Each emotion should be mapped to label in conf.py similarly to examples from RAVDESS:
        - CUSTOM_LABEL_TO_EMOTION
        - CUSTOM_EMOTION_TO_LABEL
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

    if dataset_name == "RAVDESS":
        label_to_emotion = RAVDESS_LABEL_TO_EMOTION
    else:
        label_to_emotion = CUSTOM_LABEL_TO_EMOTION
        assert CUSTOM_LABEL_TO_EMOTION, "Emotion-label mapping for a custom dataset is not defined!"

    all_subject_dirs = os.listdir(full_dataset_path)
    print(f"Found a total of {len(all_subject_dirs)} subjects inside the {dataset_name} dataset")

    train_split, val_split, test_split = process_dataset(
        full_dataset_path,
        all_subject_dirs,
        CUSTOM_SETTINGS["pre_processing_config"],
        OUTPUTS_FOLDER,
        label_to_emotion,
        dataset=dataset_name
    )

    print('Writing CSV files containing the splits to storage')
    pd.DataFrame.from_dict(train_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'train.csv'))
    pd.DataFrame.from_dict(val_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'val.csv'))
    pd.DataFrame.from_dict(test_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'test.csv'))
