import numpy as np
import torch
import torchaudio
import pandas as pd
import os
import glob

from conf import DATA_PATH, EMOTIONS, EMOTION_INTENSITY, RAVDESS_DATA_PATH, SAMPLE_RATE, MAIN_FOLDER

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(device)


def create_and_save_dataset_audio_paths():
    """
    Function to create and save a dataset in CSV form from the RAVDESS dataset.
    """
    columns_df = ['Emotion', 'Emotion Intensity', 'Gender', 'Path', 'Timestamp']
    df = pd.DataFrame(columns=columns_df)

    path_files = glob.glob(os.path.join(DATA_PATH, '*', '*.wav'))

    for path_file in path_files:
        file_name = os.path.basename(path_file)
        file_name = file_name.replace('.wav', '')
        # This timestamp is not actually the timestamp. It is a placeholder bc this dataset does not have timestamp.
        timestamp = file_name.replace('-', '')
        identifiers = file_name.split('-')
        if int(identifiers[6]) % 2 == 0:
            gender = 'female'
        else:
            gender = 'male'
        new_row_dict = {'Emotion': EMOTIONS[int(identifiers[2])],
                        'Emotion Intensity': EMOTION_INTENSITY[int(identifiers[3])],
                        'Gender': gender,
                        'Path': path_file,
                        'Timestamp': timestamp
                        }
        df = pd.concat([df, pd.DataFrame([new_row_dict])], ignore_index=True)
    if not os.path.exists(RAVDESS_DATA_PATH):
        os.makedirs(RAVDESS_DATA_PATH)
    df.to_csv(os.path.join(RAVDESS_DATA_PATH, 'ravdess_dataset_with_paths.csv'), index=False)


def generate_dataset_features(path_dataset):
    """
    Generates dataset from features extracted from wav2vec2 and save it into a CSV file.
    :param path_dataset: string. The path for CSV file with the dataset containing labels and path to audio files.
    :return: None
    """
    path_df = pd.read_csv(path_dataset)
    df_features = pd.DataFrame(columns=['Emotion', 'Features Path', 'Timestamp'])

    dir_path = os.path.join(MAIN_FOLDER, 'datasets', 'RAVDESS_features')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for idx, row in path_df.iterrows():
        features_np = generate_features(row['Path'])
        # Using timestamp as file name
        name_file = str(row['Timestamp']) + '.npy'
        file_path = os.path.join(dir_path, name_file)
        np.save(file_path, features_np, fix_imports=False)

        # creating and adding the new row on the dataframe
        new_row_dict = {'Emotion': row['Emotion'],
                        'Features Path': file_path,
                        'Timestamp': row['Timestamp']
                        }
        df_features = pd.concat([df_features, pd.DataFrame([new_row_dict])], ignore_index=True)

        # np_file = np.load(file_path, allow_pickle=True, fix_imports=False)
        break

    df_features.to_csv(os.path.join(MAIN_FOLDER, 'datasets', 'ravdess_dataset_features.csv'), index=False)


def generate_features(path_audio_file):
    """
    Function to generate the features from a wav file using a wav2vec2 model to generate the features.
    :param path_audio_file: the path where to find the wav audio file
    :return: nparray representation of the tensors, i.e., features extracted from wav file
    """
    waveform, sample_rate = torchaudio.load(path_audio_file)
    waveform = waveform.to(device)

    if SAMPLE_RATE != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    # Call wav2vec2 to extract features
    features, _ = model.extract_features(waveform)
    features_np = [f.detach().numpy().astype('float32') for f in features]
    return features_np


if __name__ == '__main__':
    # create_and_save_dataset_audio_paths()
    path_dataset = os.path.join(RAVDESS_DATA_PATH, 'ravdess_dataset_with_paths.csv')
    generate_dataset_features(path_dataset)
