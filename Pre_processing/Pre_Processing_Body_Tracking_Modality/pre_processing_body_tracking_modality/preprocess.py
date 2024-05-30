from conf import CUSTOM_SETTINGS, DATA_PATH, MODALITY_FOLDER, EMOTION_TO_LABEL

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pathlib
import warnings
warnings.filterwarnings('ignore')

def call_component():

    specified_labels = list(EMOTION_TO_LABEL.keys())

    data_folder_path = DATA_PATH

    def aggregate_data_from_subfolder(folder_path, file_contains):
        all_data = pd.DataFrame()
        for file_name in os.listdir(folder_path):
            if file_contains in file_name and file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                temp_df = pd.read_csv(file_path)
                all_data = pd.concat([all_data, temp_df], ignore_index=True)
        if all_data.empty:
            print(f"No data aggregated for pattern '{file_contains}' in {folder_path}")
        return all_data

    def label_VR_data(VR_df, event_df):
        if 'event_type' not in event_df.columns:
            print("Error: 'event_type' column not found in event data.")
            return VR_df
        filtered_event_df = event_df[event_df['event_type'].isin(specified_labels)]
        VR_df.sort_values('timestamp', inplace=True)
        filtered_event_df.sort_values('timestamp', inplace=True)
        VR_df['event_label'] = 'UNLABELED'
        last_timestamp = 0
        for _, event_row in filtered_event_df.iterrows():
            timestamp, label = event_row['timestamp'], event_row['event_type']
            VR_df.loc[(VR_df['timestamp'] > last_timestamp) & (VR_df['timestamp'] <= timestamp), 'event_label'] = label
            last_timestamp = timestamp
        return VR_df

    all_labeled_VR_data = pd.DataFrame()

    # List to hold each participant's data
    participant_data = []
    participants = os.listdir(data_folder_path)

    # Process each participant's data
    for participant in participants:
        participant_folder_path = os.path.join(data_folder_path, participant)
        aggregated_VR_data = aggregate_data_from_subfolder(participant_folder_path, 'VR_')
        aggregated_event_data = aggregate_data_from_subfolder(participant_folder_path, 'EVENT_')
        if not aggregated_event_data.empty and 'event_type' in aggregated_event_data.columns:
            labeled_VR_data = label_VR_data(aggregated_VR_data, aggregated_event_data)
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
    rest_participants_data['lcontroller_rotW'] = pd.to_numeric(rest_participants_data['lcontroller_rotW'], errors='coerce')

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
        X = data.drop(['event_label'], axis=1)
        y = data['event_label']
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        segment_size = CUSTOM_SETTINGS["pre_processing_config"]["segment_size"]
        
        def create_fixed_size_segments(features, labels, segment_size):
            max_index = len(features) // segment_size * segment_size
            features_truncated = features.iloc[:max_index]
            labels_truncated = labels[:max_index]
            segmented_features = np.array(features_truncated).reshape(-1, segment_size, features_truncated.shape[1])
            segmented_labels = []
            for i in range(0, len(labels_truncated), segment_size):
                segmented_labels.append(labels_truncated[i])
            return segmented_features, np.array(segmented_labels)

        X_segmented, y_segmented = create_fixed_size_segments(X, y_encoded, segment_size)

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

    # Drop the first column from each DataFrame
    participant1_data = participant1_data.iloc[:, 1:]
    participant2_data = participant2_data.iloc[:, 1:]
    rest_participants_data = rest_participants_data.iloc[:, 1:]

    # Process and store each set of data
    participant1_flattened, participant1_label_encoder, participant1_y_encoded, participant1_y_segmented, participant1_decoded_labels = prepare_data(participant1_data)
    participant2_flattened, participant2_label_encoder, participant2_y_encoded, participant2_y_segmented, participant2_decoded_labels = prepare_data(participant2_data)
    rest_flattened, rest_label_encoder, rest_y_encoded, rest_y_segmented, rest_decoded_labels = prepare_data(rest_participants_data)

    # Process and store each set of data
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

    # Add the corresponding labels with actual names to the data
    val_labels = participant2_decoded_labels
    test_labels = participant1_decoded_labels
    train_labels = rest_decoded_labels

    val_df = pd.DataFrame(val)
    val_df['label'] = val_labels 

    test_df = pd.DataFrame(test)
    test_df['label'] = test_labels

    train_df = pd.DataFrame(train)
    train_df['label'] = train_labels

    # Save the DataFrames to CSV files in the current directory
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
