import os
import pathlib
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from conf import CUSTOM_SETTINGS, DATA_PATH, MODALITY_FOLDER, EMOTION_TO_LABEL, MODALITY

warnings.filterwarnings('ignore')


def continious_to_categorical(info: str, categories=["BORED", "ENGAGED", "FRUSTRATED"], borders=None) -> str:
    """
    Maps a continuous value in the range [0, 1] to a discrete category.

    Args:
        info: A string containing a float value between 0 and 1.
        categories: Categories to map continuous values to.
        borders: A list of float values defining the borders for each category.
                 Must be in ascending order and have len(categories) - 1 elements.

    """
    num_categories = len(categories)
    try:
        value = float(info)
        if not 0 <= value <= 1:
            raise ValueError("Value must be between 0 and 1")

        if borders is None:
            category_size = 1. / num_categories
            category_index = min(int(value // category_size), num_categories - 1)
        else:
            if len(borders) != num_categories - 1:
                raise ValueError("Borders list must have len(categories) - 1 elements.")
            category_index = next((i for i, border in enumerate(borders) if value < border), num_categories - 1)

        category = categories[category_index]

        return str(category)
    except ValueError:
        print(f"Invalid input: {info}. Expected a float between 0 and 1.")
        return "INVALID"


def call_component():
    specified_labels = list(EMOTION_TO_LABEL.keys())

    data_folder_path = DATA_PATH

    def aggregate_data_from_subfolder(folder_path, file_contains):
        all_data = pd.DataFrame()
        i = 1
        for file_name in os.listdir(folder_path):
            if file_contains in file_name and file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                temp_df = pd.read_csv(file_path)
                if (
                    file_contains == "EVENT_" and
                    not any(col in temp_df.columns for col in ["timestamp", "event_type", "info"])
                ):
                    # Reset the DataFrame with proper column names
                    temp_df = pd.read_csv(file_path, names=["timestamp", "event_type", "info"])
                temp_df["session_num"] = file_name.split("_")[2]
                all_data = pd.concat([all_data, temp_df], ignore_index=True)
                i += 1
        if all_data.empty:
            print(f"No data aggregated for pattern '{file_contains}' in {folder_path}")
        return all_data

    def label_VR_data(VR_df, event_df, cont_to_cat: bool = True, borders=None):
        if 'event_type' not in event_df.columns:
            print("Error: 'event_type' column not found in event data.")
            return VR_df
        filtered_event_df = event_df[
            (
                event_df['event_type'].isin(specified_labels)
            ) | (
                event_df['event_type'].isin(["FEEDBACK_RECEIVED", "LEVEL_COMPLETED", "LEVEL_FAILED", "LEVEL_STARTED"])
            )
        ]
        VR_df.sort_values('timestamp', inplace=True)
        filtered_event_df.sort_values('timestamp', inplace=True)
        VR_df['event_label'] = 'UNLABELED'
        last_started_timestamp = None
        last_completed_timestamp = None
        label = None
        for _, event_row in filtered_event_df.iterrows():
            if event_row["event_type"] == "LEVEL_STARTED":
                last_started_timestamp = event_row["timestamp"]
                last_completed_timestamp = None
                label = None
            if (
                event_row["event_type"] in ["LEVEL_COMPLETED", "LEVEL_FAILED"] and
                last_started_timestamp is not None
            ):
                last_completed_timestamp = event_row["timestamp"]
            if (
                event_row["event_type"] in specified_labels and
                last_started_timestamp is not None and
                last_completed_timestamp is not None
            ):
                label = event_row["event_type"]
            elif (
                event_row["event_type"] == "FEEDBACK_RECEIVED" and
                last_started_timestamp is not None and
                last_completed_timestamp is not None
            ):
                label = continious_to_categorical(event_row["info"], borders=borders) if cont_to_cat else event_row["info"]
            if (
                last_started_timestamp is not None and
                last_completed_timestamp is not None and
                label is not None
            ):
                assert last_started_timestamp < last_completed_timestamp, \
                    f"Last started timestamp {last_started_timestamp} is greater than last completed timestamp {last_completed_timestamp}"
                VR_df.loc[(VR_df['timestamp'] > last_started_timestamp) & (VR_df['timestamp'] <= last_completed_timestamp), 'event_label'] = label
                last_started_timestamp = None
                last_completed_timestamp = None
                label = None
        return VR_df[~VR_df["event_label"].isin(["UNLABELED", "INVALID"])]

    # List to hold each participant's data
    participant_data = []
    participants = os.listdir(data_folder_path)

    def standardize_column_names(df, borders=None):
        """ In different versions of Magic XRoom, column names can be slightly different.
            These differences have been spotted between versions 1.0 and 1.2,
            for Y position of the controllers:
                - lcontroller_pposY -> lcontroller_posY
                - rcontroller_pposY -> rcontroller_posY
        """
        rename_dict = {}
        for col in df.columns:
            # Remove 'p' from 'pposX', 'pposY', 'pposZ' if present
            new_col = re.sub(r'p(pos[XYZ])', r'\1', col)
            if new_col != col:
                rename_dict[col] = new_col
        return df.rename(columns=rename_dict)

    # Process each participant's data
    for participant in participants:
        participant_folder_path = os.path.join(data_folder_path, participant)
        aggregated_VR_data = aggregate_data_from_subfolder(participant_folder_path, 'VR_')
        aggregated_VR_data = standardize_column_names(aggregated_VR_data)
        aggregated_event_data = aggregate_data_from_subfolder(participant_folder_path, 'EVENT_')
        if not aggregated_event_data.empty and 'event_type' in aggregated_event_data.columns:
            labeled_VR_data = label_VR_data(
                aggregated_VR_data,
                aggregated_event_data,
                borders=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["borders"]
            )
            participant_data.append(labeled_VR_data)
        else:
            print(f"No event data or 'event_type' column for participant {participant}, skipping labeling.")

    # Separate data for the first two participants and the rest
    participant1_data, participant2_data = participant_data[:2]  # Data for the first two participants
    rest_participants_data = pd.concat(participant_data[2:], ignore_index=True)  # Data for the rest

    def correct_row(row):
        head_rotW_value = str(row['head_rotW'])
        matched_values = re.findall(r'-?\d+\.\d+|-?\d+', head_rotW_value)
        if len(matched_values) > 1:
            row['head_rotW'] = matched_values[0]
            row['rcontroller_rotW'] = row['lcontroller_rotW']
            row['lcontroller_rotW'] = matched_values[1]
        elif len(matched_values) == 1:
            row['head_rotW'] = matched_values[0]
        return row

    # Apply the correction function to each row
    participant1_data = participant1_data.apply(correct_row, axis=1)
    participant2_data = participant2_data.apply(correct_row, axis=1)
    rest_participants_data = rest_participants_data.apply(correct_row, axis=1)

    participant1_data['head_rotW'] = pd.to_numeric(participant1_data['head_rotW'], errors='coerce')
    participant2_data['head_rotW'] = pd.to_numeric(participant2_data['head_rotW'], errors='coerce')
    rest_participants_data['head_rotW'] = pd.to_numeric(rest_participants_data['head_rotW'], errors='coerce')

    participant1_data['lcontroller_rotW'] = pd.to_numeric(participant1_data['lcontroller_rotW'], errors='coerce')
    participant2_data['lcontroller_rotW'] = pd.to_numeric(participant2_data['lcontroller_rotW'], errors='coerce')
    rest_participants_data['lcontroller_rotW'] = pd.to_numeric(rest_participants_data['lcontroller_rotW'],
                                                               errors='coerce')

    participant1_data['head_rotW'].fillna(participant1_data['head_rotW'].mean(), inplace=True)
    participant2_data['head_rotW'].fillna(participant2_data['head_rotW'].mean(), inplace=True)
    rest_participants_data['head_rotW'].fillna(rest_participants_data['head_rotW'].mean(), inplace=True)

    participant1_data['lcontroller_rotW'].fillna(participant1_data['lcontroller_rotW'].mean(), inplace=True)
    participant2_data['lcontroller_rotW'].fillna(participant2_data['lcontroller_rotW'].mean(), inplace=True)
    rest_participants_data['lcontroller_rotW'].fillna(rest_participants_data['lcontroller_rotW'].mean(), inplace=True)

    print(f"Total rows in labeled VR data for participant 1: {len(participant1_data)}")
    print(f"Total rows in labeled VR data for participant 2: {len(participant2_data)}")
    print(f"Total rows in labeled VR data for the rest of the participants: {len(rest_participants_data)}")

    def prepare_data(data):
        data = data[data["event_label"] != "UNLABELED"]
        X = data.drop(['event_label', 'session_num'], axis=1)
        y = data['event_label']
        session_nums = data['session_num']
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        segment_size = CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["seq_len"] *\
            CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["frequency"]

        def create_fixed_size_segments(features, labels, session_nums, segment_size, overlap=0):
            features = np.array(features)
            labels = np.array(labels)
            session_nums = np.array(session_nums)
            overlap_steps = int(segment_size * overlap)
            segmented_features = []
            segmented_labels = []
            for i in range(0, len(features) - segment_size, segment_size - overlap_steps):
                segment_session_nums = session_nums[i: i + segment_size]
                # Make sure that all the samples in the segment are from the same session
                if len(np.unique(segment_session_nums)) > 1:
                    continue
                segmented_features.append(features[i: i + segment_size])
                label_segment = labels[i: i + segment_size]
                segmented_labels.append(np.bincount(label_segment).argmax())
            return np.stack(segmented_features), np.array(segmented_labels)

        X_segmented, y_segmented = create_fixed_size_segments(
            X,
            y_encoded,
            session_nums,
            segment_size,
            overlap=CUSTOM_SETTINGS[MODALITY]["pre_processing_config"]["overlap"]
        )

        print("Shape of segmented features:", X_segmented.shape)
        print("Shape of segmented labels:", y_segmented.shape)

        decoded_labels = label_encoder.inverse_transform(y_segmented)
        label_counts = Counter(decoded_labels)
        sorted_label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

        print("Number of samples per class after segmentation:")
        for label, count in sorted_label_counts:
            print(f"{label}: {count} samples")

        X_flattened = X_segmented.reshape(X_segmented.shape[0], -1)
        return X_flattened, label_encoder, y_encoded, y_segmented, decoded_labels

    # Drop the timestamp from the training data
    participant1_data = participant1_data.iloc[:, 1:]
    participant2_data = participant2_data.iloc[:, 1:]
    rest_participants_data = rest_participants_data.iloc[:, 1:]

    # Process and store each set of data
    participant1_flattened, participant1_label_encoder, participant1_y_encoded, participant1_y_segmented, participant1_decoded_labels = prepare_data(
        participant1_data)
    participant2_flattened, participant2_label_encoder, participant2_y_encoded, participant2_y_segmented, participant2_decoded_labels = prepare_data(
        participant2_data)
    rest_flattened, rest_label_encoder, rest_y_encoded, rest_y_segmented, rest_decoded_labels = prepare_data(
        rest_participants_data)

    val = participant2_flattened
    test = participant1_flattened
    train = rest_flattened
    print("Data preparation and segmentation complete.")

    # Standardize the data --------------------------------------------------------------
    scaler = StandardScaler()
    stacked_data1 = np.array(val)
    stacked_data2 = np.array(test)
    stacked_data3 = np.array(train)

    val = scaler.fit_transform(stacked_data1)
    test = scaler.fit_transform(stacked_data2)
    train = scaler.fit_transform(stacked_data3)
    # ------------------------------------------------------------------------------------

    # Add the corresponding labels to the data
    val_labels = participant2_decoded_labels
    test_labels = participant1_decoded_labels
    train_labels = rest_decoded_labels

    val_df = pd.DataFrame(val)
    val_df['label'] = val_labels

    test_df = pd.DataFrame(test)
    test_df['label'] = test_labels

    train_df = pd.DataFrame(train)
    train_df['label'] = train_labels

    # Save as CSV files
    pathlib.Path(
        os.path.join(
            MODALITY_FOLDER,
        )
    ).mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        os.path.join(
            MODALITY_FOLDER,
            'train.csv'
        )
    )

    val_df.to_csv(
        os.path.join(
            MODALITY_FOLDER,
            'val.csv'
        )
    )

    test_df.to_csv(
        os.path.join(
            MODALITY_FOLDER,
            'test.csv'
        )
    )

    print("CSV files are saved.")


if __name__ == '__main__':
    call_component()
