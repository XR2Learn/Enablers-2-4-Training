
from conf import CUSTOM_SETTINGS, DATA_PATH, MODALITY_FOLDER, EMOTION_TO_LABEL

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import pathlib
import warnings
warnings.filterwarnings('ignore')

def call_component():

    specified_labels = list(EMOTION_TO_LABEL.keys())
    data_folder_path = DATA_PATH

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

    specified_labels = ['FRUSTRATED', 'ENGAGED', 'BORED']
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

    # Prepare the features and target variable
    X = all_labeled_VR_data.drop(['event_label'], axis=1)  # Drop 'event_label' as it's the target variable
    # Drop the first column from X
    # X = X.iloc[:, 1:]

    vr_data = X
    # Correct the column names 
    vr_data.columns = vr_data.columns.str.replace('ppos', 'pos')
    # timestamps 
    timestamps = vr_data.iloc[:, 0]
    original_column_names = vr_data.columns  # Save column names before converting to NumPy array
    # Conversion to NumPy array
    vr_data_array = vr_data.to_numpy(dtype='float64')
    # Converting back to DataFrame with column names
    vr_data = pd.DataFrame(vr_data_array, columns=original_column_names)

    #Feature Extraction----------------------------------------------------------------------------------------
    motion_features = {}

    # 1.Velocity 
    velocity = {}
    for i, prefix in enumerate(['head', 'lcontroller', 'rcontroller']):
        pos_columns = [f'{prefix}_posX', f'{prefix}_posY', f'{prefix}_posZ']
        position_diffs = vr_data[pos_columns].diff().fillna(0)  # fill NaN values that appear as a result of diff with zero
        velocities = np.sqrt((position_diffs**2).sum(axis=1)) / np.mean(np.diff(vr_data['timestamp']))
        motion_features[f'mean_velocity_{prefix}'] = np.mean(velocities)
        motion_features[f'std_velocity_{prefix}'] = np.std(velocities)
    #----------------------------------------------------------------------------------------

    # 2.Acceleration 
    for i, prefix in enumerate(['head', 'lcontroller', 'rcontroller']):
        velocity_diffs = vr_data[pos_columns].diff().fillna(0).diff().fillna(0)
        accelerations = np.sqrt((velocity_diffs**2).sum(axis=1)) / np.mean(np.diff(vr_data['timestamp']))
        motion_features[f'mean_acceleration_{prefix}'] = np.mean(accelerations)
        motion_features[f'std_acceleration_{prefix}'] = np.std(accelerations)
    #----------------------------------------------------------------------------------------

    # 3.Range of Motion 
    window_size = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["rom_win_size"]  # Define the size of the window for the local range of motion
    # store motion features with the same length as vr_data
    motion_features = pd.DataFrame(index=vr_data.index)
    # Calculate time intervals
    timestamps = pd.Series(vr_data['timestamp'])
    time_intervals = timestamps.diff().fillna(method='bfill')  # Back-fill the first NaN value
    # Calculate localized range of motion within each window
    for prefix in ['head', 'lcontroller', 'rcontroller']:
        pos_columns = [f'{prefix}_posX', f'{prefix}_posY', f'{prefix}_posZ']
        for col in pos_columns:
            # Rolling window operation
            rom = vr_data[col].rolling(window=window_size, min_periods=1).apply(lambda x: x.max() - x.min(), raw=True)
            motion_features[f'local_range_of_motion_{col}'] = rom
    #----------------------------------------------------------------------------------------

    # 4.Jerk using positional data
    def calculate_jerk(pos_data):
        velocity = pos_data.diff().fillna(0)
        acceleration = velocity.diff().fillna(0)
        jerk = acceleration.diff().fillna(0)
        return jerk / (time_intervals ** CUSTOM_SETTINGS["handcrafted_feature_extraction"]["jerk_time_interval"])
    # 5.Difference over time using positional data
    def calculate_difference_over_time(pos_data):
        diffs = pos_data.diff().fillna(0)
        return diffs / time_intervals
    # 4 and 5.jerk and difference over time for each coordinate of each body part
    for prefix in ['head', 'lcontroller', 'rcontroller']:
        for axis in ['X', 'Y', 'Z']:
            pos_column = f'{prefix}_pos{axis}'
            pos_data = vr_data[pos_column]
            jerk_over_time = calculate_jerk(pos_data)
            motion_features[f'jerk_{prefix}_{axis}'] = jerk_over_time
            difference_over_time = calculate_difference_over_time(pos_data)
            motion_features[f'difference_{prefix}_{axis}'] = difference_over_time
    #----------------------------------------------------------------------------------------

    # 6. Correlation
    # DataFrame to store correlation features
    correlation_features = pd.DataFrame(index=vr_data.index)
    # correlation over time between different sets of positional data
    def calculate_correlation(pos_data1, pos_data2):
        # Using rolling window to calculate dynamic correlation over time
        window_size = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["correlation_win_size"]  
        correlation_over_time = pos_data1.rolling(window=window_size).corr(pos_data2)
        return correlation_over_time
    # Compute correlation for each combination of body parts and axes
    body_parts = ['head', 'lcontroller', 'rcontroller']
    axes = ['X', 'Y', 'Z']
    for i in range(len(body_parts)):
        for j in range(i + 1, len(body_parts)):
            for axis in axes:
                pos_data1 = vr_data[f'{body_parts[i]}_pos{axis}']
                pos_data2 = vr_data[f'{body_parts[j]}_pos{axis}']
                corr_over_time = calculate_correlation(pos_data1, pos_data2)
                correlation_features[f'corr_{body_parts[i]}_{body_parts[j]}_{axis}'] = corr_over_time
    # Fill NaN values that result from the rolling correlation
    correlation_features.fillna(method='bfill', inplace=True)
    #----------------------------------------------------------------------------------------

    # 7.Power Specteral Density
    # Calculate PSD using Welch's method
    def calculate_psd(pos_data, fs, nperseg=CUSTOM_SETTINGS["handcrafted_feature_extraction"]["psd_sample_per_segment"], noverlap=CUSTOM_SETTINGS["handcrafted_feature_extraction"]["psd_sample_segment_overlap"]):
        frequencies, power = welch(pos_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        return frequencies, power
    # Calculate time differences in seconds
    time_diffs = np.diff(vr_data['timestamp']).astype('timedelta64[s]').astype(int)
    # Calculate sampling frequency (fs) safely
    fs = 1 / np.mean(time_diffs) if time_diffs.size > 0 else 1  # prevent division by zero
    # Window length and overlap for PSD calculation
    window_length = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["psd_win_len"]  # Length of the window for PSD calculation
    overlap = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["psd_win_len_overlap"]
    # DataFrame to store PSD features, ensuring it matches the index of vr_data
    psd_features = pd.DataFrame(index=vr_data.index)
    # Compute PSD for each coordinate of each body part using a sliding window
    body_parts = ['head', 'lcontroller', 'rcontroller']
    axes = ['X', 'Y', 'Z']
    for part in body_parts:
        for axis in axes:
            pos_data_full = vr_data[f'{part}_pos{axis}']
            mean_powers = np.empty(len(pos_data_full))
            peak_frequencies = np.empty(len(pos_data_full))
            # Sliding window
            for i in range(0, len(pos_data_full) - window_length + 1, window_length - overlap):
                end_idx = i + window_length
                pos_data = pos_data_full[i:end_idx]
                frequencies, power = calculate_psd(pos_data, fs)
                mean_power = power.mean()
                peak_frequency = frequencies[np.argmax(power)]
                mean_powers[i:end_idx] = mean_power
                peak_frequencies[i:end_idx] = peak_frequency
            # Assign results to DataFrame
            psd_features[f'psd_mean_power_{part}_{axis}'] = mean_powers
            psd_features[f'psd_peak_frequency_{part}_{axis}'] = peak_frequencies
    # Interpolate to fill gaps for initial and final segments
    psd_features.interpolate(method='linear', inplace=True)
    psd_features.fillna(method='bfill', inplace=True)
    psd_features.fillna(method='ffill', inplace=True)
    #----------------------------------------------------------------------------------------

    # 8.Speed
    # DataFrame to store speed features
    speed_features = pd.DataFrame(index=vr_data.index)
    # Calculate speed using the Euclidean norm of the velocity vector for each body part
    def calculate_speed(pos_data_x, pos_data_y, pos_data_z, timestamps):
        # Calculate time intervals
        time_intervals = timestamps.diff().fillna(method='bfill')
        # Compute velocity components for each axis
        vel_x = pos_data_x.diff().fillna(0) / time_intervals
        vel_y = pos_data_y.diff().fillna(0) / time_intervals
        vel_z = pos_data_z.diff().fillna(0) / time_intervals
        # Calculate speed as the magnitude of the velocity vector
        speed = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        return speed
    # Calculate speed for each coordinate of each body part
    body_parts = ['head', 'lcontroller', 'rcontroller']
    axes = ['X', 'Y', 'Z']
    for part in body_parts:
        pos_data_x = vr_data[f'{part}_posX']
        pos_data_y = vr_data[f'{part}_posY']
        pos_data_z = vr_data[f'{part}_posZ']
        timestamps = vr_data['timestamp']  
        speed_over_time = calculate_speed(pos_data_x, pos_data_y, pos_data_z, timestamps)
    #---------------------------------------------------------------------------------------

    # 9.Angular_velocity and Angular_acceleration
    # DataFrame to store angular features
    angular_features = pd.DataFrame(index=vr_data.index)
    # Calculate angular velocity and acceleration
    def calculate_angular_derivatives(rot_data, timestamps):
        # Calculate time intervals
        time_intervals = timestamps.diff().fillna(method='bfill')
        # Compute angular velocity (first derivative of rotation data)
        angular_velocity = rot_data.diff().fillna(0) / time_intervals
        # Compute angular acceleration (second derivative of rotation data)
        angular_acceleration = angular_velocity.diff().fillna(0) / time_intervals
        return angular_velocity, angular_acceleration
    # Process each body part and each rotation axis
    body_parts = ['head', 'lcontroller', 'rcontroller']
    axes = ['X', 'Y', 'Z']
    for part in body_parts:
        for axis in axes:
            rot_data = vr_data[f'{part}_rot{axis}']
            timestamps = vr_data['timestamp']  
            ang_velocity, ang_acceleration = calculate_angular_derivatives(rot_data, timestamps)
            angular_features[f'angular_velocity_{part}_{axis}'] = ang_velocity
            angular_features[f'angular_acceleration_{part}_{axis}'] = ang_acceleration
    #----------------------------------------------------------------------------------------

    # 10.Path_length
    # DataFrame to store computed features
    path_length_features = pd.DataFrame()
    descriptive_stats = pd.DataFrame()
    # Calculate Path Length for each coordinate of each body part
    def calculate_path_length(pos_data):
        # Calculate distances between consecutive points
        distances = np.sqrt((pos_data.diff().fillna(0) ** 2).sum(axis=1))
        # Cumulative sum of distances gives the path length over time
        path_length = distances.cumsum()
        return path_length
    #----------------------------------------------------------------------------------------

    # 11.Statistical Features
    def compute_descriptive_stats(data, feature_name, stats_features):
        # Compute stats and store them in provided DataFrame
        stats_features[f'mean_{feature_name}'] = np.mean(data)
        stats_features[f'std_{feature_name}'] = np.std(data)
        stats_features[f'min_{feature_name}'] = np.min(data)
        stats_features[f'max_{feature_name}'] = np.max(data)
    # Parameters for sliding window
    window_size = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["stat_win_size"]  # Length of the window for calculations
    overlap = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["stat_win_size_overlap"]   # Number of overlapping data points
    # store statistics
    stats_features = pd.DataFrame(index=vr_data.index)
    # calculate path length
    def calculate_path_length(pos_data):
        diffs = np.sqrt(np.sum(np.diff(pos_data, axis=0)**2, axis=1))
        return np.sum(diffs)
    # Process each body part and each coordinate for path length and descriptive stats
    body_parts = ['head', 'lcontroller', 'rcontroller']
    axes = ['X', 'Y', 'Z']
    for part in body_parts:
        # Calculate path length features with sliding windows
        pos_data = vr_data[[f'{part}_pos{axis}' for axis in axes]].dropna()
        path_lengths = []
        for start in range(0, len(pos_data) - window_size + 1, window_size - overlap):
            end = start + window_size
            window_data = pos_data.iloc[start:end]
            path_length = calculate_path_length(window_data)
            path_lengths.extend([path_length] * (window_size - overlap))
        pad_length = len(vr_data) - len(path_lengths)
        stats_features[f'path_length_{part}'] = np.pad(path_lengths, (0, pad_length), 'edge')
        # Compute descriptive statistics for each position and rotation data using sliding windows
        for axis in axes:
            for data_type in ['pos', 'rot']:
                feature_name = f'{part}_{data_type}{axis}'
                pos_data = vr_data[f'{part}_{data_type}{axis}'].dropna()
                means, stds, mins, maxs = [], [], [], []
                for start in range(0, len(pos_data) - window_size + 1, window_size - overlap):
                    end = start + window_size
                    window_data = pos_data.iloc[start:end].to_numpy()
                    means.append(np.mean(window_data))
                    stds.append(np.std(window_data))
                    mins.append(np.min(window_data))
                    maxs.append(np.max(window_data))
                # Pad the results arrays to ensure they match the input data length
                pad_length = len(vr_data) - len(means)
                stats_features[f'mean_{feature_name}'] = np.pad(means, (0, pad_length), 'edge')
                stats_features[f'std_{feature_name}'] = np.pad(stds, (0, pad_length), 'edge')
                stats_features[f'min_{feature_name}'] = np.pad(mins, (0, pad_length), 'edge')
                stats_features[f'max_{feature_name}'] = np.pad(maxs, (0, pad_length), 'edge')
    # Handle edge cases and fill gaps if any
    stats_features.interpolate(method='linear', inplace=True)
    stats_features.ffill(inplace=True)
    stats_features.bfill(inplace=True)
    #---------------------------------------------------------------------------------------

    # 12.FFT Features
    # Convert 'timestamp' to datetime and calculate sampling rate
    timestamps = pd.to_datetime(vr_data['timestamp'])
    time_diffs = np.diff(timestamps).astype('timedelta64[s]').astype(float)
    mean_diff_seconds = np.mean(time_diffs) if len(time_diffs) > 0 else 1
    sampling_rate = 1 / mean_diff_seconds if mean_diff_seconds > 0 else 1
    # frequency analysis using FFT on a window of data
    def frequency_analysis(pos_data, sampling_rate):
        # Ensure the input data is a numpy array for FFT
        pos_data = np.asarray(pos_data)  # Ensure data is in numpy array format
        # Perform FFT and get frequency spectrum
        yf = fft(pos_data)
        xf = fftfreq(len(pos_data), 1 / sampling_rate)
        # Compute the amplitude spectrum
        amp_spectrum = np.abs(yf)
        # Find dominant frequency
        idx_max = np.argmax(amp_spectrum)
        dominant_frequency = xf[idx_max]
        mean_frequency = np.sum(xf * amp_spectrum) / np.sum(amp_spectrum)  # Weighted average
        energy_spectrum = np.sum(amp_spectrum**2) / len(pos_data)
        return dominant_frequency, mean_frequency, energy_spectrum
    # Parameters for sliding window
    window_size = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["fft_win_size"]  # Length of the window for FFT calculation
    overlap = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["fft_win_size_overlap"]  # Number of overlapping data points
    # DataFrame to store frequency features
    frequency_features = pd.DataFrame(index=vr_data.index)
    # Process each body part and each coordinate for frequency analysis
    body_parts = ['head', 'lcontroller', 'rcontroller']
    axes = ['X', 'Y', 'Z']
    for part in body_parts:
        for axis in axes:
            pos_data_full = vr_data[f'{part}_pos{axis}'].dropna()
            dominant_freqs = []
            mean_freqs = []
            energy_specs = []
            # Apply sliding window technique
            for start in range(0, len(pos_data_full) - window_size + 1, window_size - overlap):
                end = start + window_size
                window_data = pos_data_full.iloc[start:end].to_numpy()  # Explicitly convert to numpy array
                dominant_freq, mean_freq, energy_spec = frequency_analysis(window_data, sampling_rate)
                # Append results for each window
                dominant_freqs.extend([dominant_freq] * (window_size - overlap))
                mean_freqs.extend([mean_freq] * (window_size - overlap))
                energy_specs.extend([energy_spec] * (window_size - overlap))
            # Ensure the results match the original data length
            pad_length = len(pos_data_full) - len(dominant_freqs)
            frequency_features[f'dominant_freq_{part}_{axis}'] = np.pad(dominant_freqs, (0, pad_length), 'edge')
            frequency_features[f'mean_freq_{part}_{axis}'] = np.pad(mean_freqs, (0, pad_length), 'edge')
            frequency_features[f'energy_spectrum_{part}_{axis}'] = np.pad(energy_specs, (0, pad_length), 'edge')
    # Handle edge cases and fill gaps if any
    frequency_features.interpolate(method='linear', inplace=True)
    frequency_features.ffill(inplace=True)
    frequency_features.bfill(inplace=True)
    #----------------------------------------------------------------------------------------

    # 13.Symmetry
    def calculate_symmetry(data_left, data_right):
        # Calculate the absolute difference between left and right body parts
        symmetry_diff = np.abs(data_left - data_right)
        return symmetry_diff.mean(), symmetry_diff.std()
    # Parameters for sliding window
    window_size = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["sym_win_size"]  # Length of the window for symmetry calculation
    overlap = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["sym_win_size_overlap"]       # Number of overlapping data points
    # DataFrame to store symmetry features, matching the index of vr_data
    symmetry_features = pd.DataFrame(index=vr_data.index)
    # Calculate symmetry for each axis
    axes = ['X', 'Y', 'Z']
    for axis in axes:
        data_left = vr_data[f'lcontroller_pos{axis}'].dropna()
        data_right = vr_data[f'rcontroller_pos{axis}'].dropna()
        mean_symmetries = []
        std_symmetries = []
        # Apply sliding window technique
        for start in range(0, len(data_left) - window_size + 1, window_size - overlap):
            end = start + window_size
            window_data_left = data_left.iloc[start:end].to_numpy()  # Explicitly convert to numpy array
            window_data_right = data_right.iloc[start:end].to_numpy()  # Explicitly convert to numpy array
            mean_symmetry, std_symmetry = calculate_symmetry(window_data_left, window_data_right)
            # Append results for each window
            mean_symmetries.extend([mean_symmetry] * (window_size - overlap))
            std_symmetries.extend([std_symmetry] * (window_size - overlap))
        # Ensure the results match the original data length
        pad_length_mean = len(data_left) - len(mean_symmetries)
        pad_length_std = len(data_right) - len(std_symmetries)
        symmetry_features[f'symmetry_mean_{axis}'] = np.pad(mean_symmetries, (0, pad_length_mean), 'edge')
        symmetry_features[f'symmetry_std_{axis}'] = np.pad(std_symmetries, (0, pad_length_std), 'edge')
    # Handle edge cases and fill gaps if any
    symmetry_features.interpolate(method='linear', inplace=True)
    symmetry_features.ffill(inplace=True)
    symmetry_features.bfill(inplace=True)
    #----------------------------------------------------------------------------------------

    # 14.Harmonics
    def calculate_harmonics(data, sampling_rate):
        # Perform FFT
        yf = fft(data)
        xf = fftfreq(len(data), 1 / sampling_rate)
        # Filter out negative frequencies
        mask = xf > 0
        xf = xf[mask]
        yf = yf[mask]
        # Get amplitude spectrum
        amplitude_spectrum = np.abs(yf)
        # Identify harmonics: find peaks in the amplitude spectrum
        peaks, _ = find_peaks(amplitude_spectrum, height=0)
        # Return frequencies of the harmonic peaks and their amplitudes
        harmonic_freqs = xf[peaks]
        harmonic_amps = amplitude_spectrum[peaks]
        return harmonic_freqs, harmonic_amps
    # Parameters for sliding window
    window_size = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["harmo_win_size"]  # Length of the window for harmonic analysis
    overlap = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["harmo_win_size_overlap"]   # Number of overlapping data points
    # DataFrame to store harmonic features, matching the index of vr_data
    harmonics_features = pd.DataFrame(index=vr_data.index)
    # Calculate harmonics for each body part and coordinate using sliding windows
    for part in ['head', 'lcontroller', 'rcontroller']:
        for axis in ['X', 'Y', 'Z']:
            pos_data = vr_data[f'{part}_pos{axis}'].dropna()
            # lists to collect harmonic data
            all_harmonic_freqs = np.array([])
            all_harmonic_amps = np.array([])
            # Apply sliding window technique
            for start in range(0, len(pos_data) - window_size + 1, window_size - overlap):
                end = start + window_size
                window_data = pos_data.iloc[start:end].to_numpy()
                harmonic_freqs, harmonic_amps = calculate_harmonics(window_data, sampling_rate)
                # Combine harmonic data from each window
                all_harmonic_freqs = np.concatenate((all_harmonic_freqs, harmonic_freqs))
                all_harmonic_amps = np.concatenate((all_harmonic_amps, harmonic_amps))
            # Store results in DataFrame, handling different data lengths
            pad_length_freqs = len(vr_data) - len(all_harmonic_freqs)
            pad_length_amps = len(vr_data) - len(all_harmonic_amps)
            harmonics_features[f'harmonic_freqs_{part}_{axis}'] = np.pad(all_harmonic_freqs, (0, pad_length_freqs), 'constant', constant_values=(np.nan,))
            harmonics_features[f'harmonic_amps_{part}_{axis}'] = np.pad(all_harmonic_amps, (0, pad_length_amps), 'constant', constant_values=(np.nan,))
    #----------------------------------------------------------------------------------------

    # 15.Joint distance
    # calculate Euclidean distance between two points in 3D space
    def calculate_joint_distance(pos_data1, pos_data2):
        # Compute the squared differences along each axis
        squared_diffs = (pos_data1 - pos_data2) ** 2
        # Sum the squared differences and take the square root to get Euclidean distance
        distance = np.sqrt(squared_diffs.sum(axis=1))
        return distance
    # DataFrame to store joint distance features
    joint_distance_features = pd.DataFrame(index=vr_data.index)
    # Combinations of body parts to compare
    combinations = [
        ('head', 'lcontroller'),
        ('head', 'rcontroller'),
        ('lcontroller', 'rcontroller')
    ]
    # Calculate distances for combinations of joints across their X, Y, Z coordinates
    for (part1, part2) in combinations:
        # Calculate joint distance for each axis and combine
        distance = pd.Series(0, index=vr_data.index)
        for axis in ['X', 'Y', 'Z']:
            pos_data1 = vr_data[f'{part1}_pos{axis}']
            pos_data2 = vr_data[f'{part2}_pos{axis}']
            distance += (pos_data1 - pos_data2) ** 2
        joint_distance_features[f'distance_{part1}_{part2}'] = np.sqrt(distance)
    # ----------------------------------------------------------------------------------------

    # Convert each feature into a DataFrame if not already (handling scalar and series cases)
    features_list = [
        velocities, velocity_diffs, accelerations, rom, jerk_over_time, difference_over_time,
        correlation_features, psd_features, speed_features, path_length_features, angular_features, stats_features,
        frequency_features,symmetry_features,joint_distance_features
    ]
    # Convert each feature into a DataFrame if not already (handling scalar and series cases)
    feature_frames = []
    for feature in features_list:
        if isinstance(feature, pd.Series):
            feature = feature.to_frame()  # Convert series to a DataFrame without transposing
        elif isinstance(feature, np.ndarray):
            feature = pd.DataFrame(feature)  # Convert array to DataFrame
        elif np.isscalar(feature):
            feature = pd.DataFrame([feature] * number_of_samples)  # Repeat the scalar for each sample
        feature_frames.append(feature)
    # Concatenate all feature DataFrames horizontally
    all_features_df = pd.concat(feature_frames, axis=1)
    # Print the resulting DataFrame's shape to confirm the structure
    print("Resulting DataFrame shape:", all_features_df.shape)

    # Frames Segmentation --------------------------------------------------------
    X = all_features_df
    y = all_labeled_VR_data['event_label']
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

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
    segment_size = CUSTOM_SETTINGS["handcrafted_feature_extraction"]["frame_segment_size"]

    # Applying the function to segment X and y_encoded
    X_segmented, y_segmented = create_fixed_size_segments(X, y_encoded, segment_size)
    # Print the shape of the segmented data to verify
    print("Shape of segmented features:", X_segmented.shape)
    print("Shape of segmented labels:", y_segmented.shape)
    print(" ")
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
    stacked_data = np.array(X_flattened)
    scaler = StandardScaler()
    # Fit the scaler to the data and transform it
    X_flattened = scaler.fit_transform(stacked_data)
    # ------------------------------------------------------------------------------------

    # Save CSVs-----------------------------
    # Split the data into training and temporary sets
    X_train, X_temp, y_train_encoded, y_temp_encoded = train_test_split(
      X_flattened, y_segmented, test_size=CUSTOM_SETTINGS["handcrafted_feature_extraction"]["test_size"], random_state=42, stratify=y_segmented)
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
    # train_df.to_csv('train_set.csv', index=False)
    # test_df.to_csv('test_set.csv', index=False)
    # val_df.to_csv('validation_set.csv', index=False)

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
