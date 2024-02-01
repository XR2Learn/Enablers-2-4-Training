import os

from conf import (
    CUSTOM_SETTINGS,
    BM_LABEL_TO_EMOTION,
    BM_DATA_PATH,
    OUTPUTS_FOLDER
)

from preprocessing_utils import process_dataset


def preprocess():
    print('Pre Processing BM Modality')

    dataset_name = CUSTOM_SETTINGS['dataset_config']['dataset_name']
    label_to_emotion = BM_LABEL_TO_EMOTION

    all_subject_dirs = os.listdir(BM_DATA_PATH)
    print(f"Found a total of {len(all_subject_dirs)} under {BM_DATA_PATH}.")

    train_split, val_split, test_split = process_dataset(
        BM_DATA_PATH,
        all_subject_dirs,
        CUSTOM_SETTINGS["pre_processing_config"],
        OUTPUTS_FOLDER,
        label_to_emotion,
        dataset=dataset_name
    )


if __name__ == '__main__':
    preprocess()
