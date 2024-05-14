from conf import CUSTOM_SETTINGS, DATA_PATH, MODALITY_FOLDER, EMOTION_TO_LABEL


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def call_component():
    print('Call component Pre Processing Body Tracking Modality')
    
    specified_labels = list(EMOTION_TO_LABEL.keys())

    data_folder_path = DATA_PATH  # Corrected main folder name

    # Aggregate data from multiple CSV files within a subfolder based on file name pattern
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

    # Process and label the aggregated VR data for a participant based on their aggregated event data
    def label_VR_data(VR_df, event_df):
        if 'event_type' not in event_df.columns:
            print("Error: 'event_type' column not found in event data.")
            return VR_df  # Return VR_df without labeling if 'event_type' is missing
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

    # Iterate over each participant's subfolders within the main data folder
    for participant in os.listdir(data_folder_path):
        participant_folder_path = os.path.join(data_folder_path, participant)
        aggregated_VR_data = aggregate_data_from_subfolder(participant_folder_path, 'VR_')
        aggregated_event_data = aggregate_data_from_subfolder(participant_folder_path, 'EVENT_')
        if not aggregated_event_data.empty and 'event_type' in aggregated_event_data.columns:
            labeled_VR_data = label_VR_data(aggregated_VR_data, aggregated_event_data)
            all_labeled_VR_data = pd.concat([all_labeled_VR_data, labeled_VR_data], ignore_index=True)
        else:
            print(f"No event data or 'event_type' column for participant {participant}, skipping labeling.")
    print("Labeling complete. Data ready for further processing.")


    import re   
    def correct_row(row):
        # Extract the 'head_rotW' value 
        head_rotW_value = str(row['head_rotW'])
        # find and split concatenated values like '1.-0'
        matched_values = re.findall(r'-?\d+\.\d+|-?\d+', head_rotW_value)
        if len(matched_values) > 1:
            # If two numbers are found, assign the first to 'head_rotW' and the second to 'lcontroller_rotW'
            row['head_rotW'] = matched_values[0]
            # Before shifting existing 'lcontroller_rotW' to 'rcontroller_rotW', save it
            row['rcontroller_rotW'] = row['lcontroller_rotW']
            row['lcontroller_rotW'] = matched_values[1]
        elif len(matched_values) == 1:
            # Only one number found, assign it to 'head_rotW'
            row['head_rotW'] = matched_values[0]
        return row

    # Apply the correction function to each row
    corrected_data = all_labeled_VR_data.apply(correct_row, axis=1)
    # Convert the 'head_rotW' and 'lcontroller_rotW' columns in the corrected_data from string to float
    corrected_data['head_rotW'] = pd.to_numeric(corrected_data['head_rotW'], errors='coerce')
    corrected_data['lcontroller_rotW'] = pd.to_numeric(corrected_data['lcontroller_rotW'], errors='coerce')
    # Check and handle NaNs that may have been introduced during conversion
    # Fill NaNs with the mean of the column 
    corrected_data['head_rotW'].fillna(corrected_data['head_rotW'].mean(), inplace=True)
    corrected_data['lcontroller_rotW'].fillna(corrected_data['lcontroller_rotW'].mean(), inplace=True)
    all_labeled_VR_data = corrected_data

    # Print the total number of rows to verify it matches expectations
    print(f"Total rows in labeled VR data: {len(all_labeled_VR_data)}")
    # Count the number of observations for each class (label)
    observations_per_class = all_labeled_VR_data['event_label'].value_counts()
    # Print the number of observations for each class
    print("Number of observations per class:")
    print(observations_per_class)
    observations_per_class_df = observations_per_class.reset_index()
    observations_per_class_df.columns = ['Event Label', 'Number of Observations']
    print(" ")


    # Prepare the features and target variable
    X = all_labeled_VR_data.drop(['event_label'], axis=1)  # Drop 'event_label' as it's the target variable
    # Drop the first column from X
    X = X.iloc[:, 1:]
    y = all_labeled_VR_data['event_label']

    # Convert columns with 'object' dtype to categorical if they are not numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')

    # encode categorical columns as numeric using their category codes
    categorical_cols = X.select_dtypes(['category']).columns
    X[categorical_cols] = X[categorical_cols].apply(lambda x: x.cat.codes)
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)


    # Frames Segmentation --------------------------------------------------------
    def create_fixed_size_segments(features, labels, segment_size):
        # Ensure that the total number of rows is a multiple of segment_size
        max_index = len(features) // segment_size * segment_size
        # Truncate features and labels to a size that's a multiple of segment_size
        features_truncated = features.iloc[:max_index]
        labels_truncated = labels[:max_index]
        # Reshape features into segments
        segmented_features = np.array(features_truncated).reshape(-1, segment_size, features_truncated.shape[1])
        # For labels, we can either take the first label of each segment or use a majority vote
        segmented_labels = []
        for i in range(0, len(labels_truncated), segment_size):
            # Example: Taking the first label of each segment
            segmented_labels.append(labels_truncated[i])
        return segmented_features, np.array(segmented_labels)

    # Segment size
    segment_size = CUSTOM_SETTINGS["pre_processing_config"]["seq_len"]*CUSTOM_SETTINGS["pre_processing_config"]["frequency"]

    # Applying the function to segment X and y_encoded
    X_segmented, y_segmented = create_fixed_size_segments(X, y_encoded, segment_size)
    # Print the shape of the segmented data to verify
    print("Shape of segmented features:", X_segmented.shape)
    print("Shape of segmented labels:", y_segmented.shape)
    print(" ")


    from collections import Counter
    decoded_labels = label_encoder.inverse_transform(y_segmented)  # Decode the labels to their original string form
    label_counts = Counter(decoded_labels)  # Count each label
    # Sort labels based on count in descending order
    sorted_label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    # Print the number of samples per class after segmentation, ordered by count
    print("Number of samples per class after segmentation:")
    for label, count in sorted_label_counts:
        print(f"{label}: {count} samples")
    print(" ")


    # Flatten the segmented features from 3D to 2D
    X_flattened = X_segmented.reshape(X_segmented.shape[0], -1)


    # Standardize the data --------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    stacked_data = np.array(X_flattened)
    scaler = StandardScaler()
    # Fit the scaler to the data and transform it
    X_flattened = scaler.fit_transform(stacked_data)
    # ------------------------------------------------------------------------------------


    # Save CSVs-----------------------------
    from sklearn.model_selection import train_test_split
    # Split the data into training and temporary sets
    X_train, X_temp, y_train_encoded, y_temp_encoded = train_test_split(
      X_flattened, y_segmented, test_size=0.3, random_state=42, stratify=y_segmented)
    # Split the temporary set into test and validation sets
    X_test, X_val, y_test_encoded, y_val_encoded = train_test_split(
      X_temp, y_temp_encoded, test_size=0.5, random_state=42, stratify=y_temp_encoded)
    # Decode the numeric labels back to original string labels
    y_train_decoded = label_encoder.inverse_transform(y_train_encoded)
    y_test_decoded = label_encoder.inverse_transform(y_test_encoded)
    y_val_decoded = label_encoder.inverse_transform(y_val_encoded)
    # Convert the numpy arrays to pandas DataFrames
    train_df = pd.DataFrame(X_train)
    train_df['Label'] = y_train_decoded  # Add decoded labels
    test_df = pd.DataFrame(X_test)
    test_df['Label'] = y_test_decoded  # Add decoded labels
    val_df = pd.DataFrame(X_val)
    val_df['Label'] = y_val_decoded  # Add decoded labels
    # Save the DataFrames to CSV files
    train_df.to_csv('train_set.csv', index=False)
    test_df.to_csv('test_set.csv', index=False)
    val_df.to_csv('validation_set.csv', index=False)

    import pathlib


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


if __name__ == '__main__':
    call_component()
