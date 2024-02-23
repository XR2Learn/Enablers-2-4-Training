import os
import pandas as pd

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

    train_split, val_split, test_split, stats, ssl_train_split, ssl_val_split, ssl_test_split = process_dataset(
        BM_DATA_PATH,
        all_subject_dirs,
        CUSTOM_SETTINGS["pre_processing_config"],
        OUTPUTS_FOLDER,
        label_to_emotion,
        dataset=dataset_name
    )

    if stats is not None:
        stats_df = (
            pd.DataFrame(stats)
            .sort_values(by=["subject", "session"])
        )

        stats_df.to_csv(os.path.join(OUTPUTS_FOLDER, "stats_biomeasurements.csv"), index=None)

    print('Writing CSV files containing the splits to storage')
    pd.DataFrame.from_dict(train_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'train.csv'))
    pd.DataFrame.from_dict(val_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'val.csv'))
    pd.DataFrame.from_dict(test_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'test.csv'))

    if (
        "get_ssl" in CUSTOM_SETTINGS["pre_processing_config"] and
        CUSTOM_SETTINGS["pre_processing_config"]["get_ssl"]
    ):
        print('Writing CSV files containing the SSL splits to storage')
        pd.DataFrame.from_dict(ssl_train_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'ssl_train.csv'))
        pd.DataFrame.from_dict(ssl_val_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'ssl_val.csv'))
        pd.DataFrame.from_dict(ssl_test_split).to_csv(os.path.join(OUTPUTS_FOLDER, 'ssl_test.csv'))


if __name__ == '__main__':
    preprocess()
