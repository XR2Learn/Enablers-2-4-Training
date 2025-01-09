import os
import pandas as pd

from conf import (
    CUSTOM_SETTINGS,
    DATA_PATH,
    MODALITY,
    MODALITY_FOLDER,
)

from preprocessing_utils import process_dataset


def preprocess():
    print('Pre Processing BM Modality')

    all_subject_dirs = os.listdir(DATA_PATH)
    print(f"Found a total of {len(all_subject_dirs)} under {DATA_PATH}.")

    train_split, val_split, test_split, stats, ssl_train_split, ssl_val_split, ssl_test_split = process_dataset(
        DATA_PATH,
        all_subject_dirs,
        CUSTOM_SETTINGS[MODALITY]["pre_processing_config"],
        MODALITY_FOLDER,
        seq_len=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("seq_len", 5),  # in seconds
        overlap=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("overlap", 0.),  # between 0 and 1 (proportion)
        frequency=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("frequency", 10),  # in Hz
        resample_freq=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get(
            "resample_freq",
            CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("frequency", 10)
        ),  # resampling if needed
        use_sensors=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("use_sensors", ["gsr"]),
        borders=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"].get("borders", None)
    )

    if stats is not None:
        stats_df = (
            pd.DataFrame(stats)
            .sort_values(by=["subject", "session"])
        )

        stats_df.to_csv(os.path.join(MODALITY_FOLDER, "stats_biomeasurements.csv"), index=None)

    print('Writing CSV files containing the splits to storage')

    pd.DataFrame.from_dict(train_split).to_csv(
        os.path.join(
            MODALITY_FOLDER,
            'train.csv'
        )
    )

    pd.DataFrame.from_dict(val_split).to_csv(
        os.path.join(
            MODALITY_FOLDER,
            'val.csv'
        )
    )
    pd.DataFrame.from_dict(test_split).to_csv(
        os.path.join(
            MODALITY_FOLDER,
            'test.csv'
        )
    )

    if (
        "get_ssl" in CUSTOM_SETTINGS[MODALITY]["pre_processing_config"] and
        CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["get_ssl"]
    ):
        print('Writing CSV files containing the SSL splits to storage')
        pd.DataFrame.from_dict(ssl_train_split).to_csv(
            os.path.join(
                MODALITY_FOLDER,
                'ssl_train.csv'
            )
        )
        pd.DataFrame.from_dict(ssl_val_split).to_csv(
            os.path.join(
                MODALITY_FOLDER,
                'ssl_val.csv'
            )
        )
        pd.DataFrame.from_dict(ssl_test_split).to_csv(
            os.path.join(
                MODALITY_FOLDER,
                'ssl_test.csv'
            )
        )


if __name__ == '__main__':
    preprocess()
