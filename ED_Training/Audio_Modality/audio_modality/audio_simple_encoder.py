import os
import random
import numpy as np
import torch
import torchaudio
import pandas as pd

from conf import RAVDESS_DATA_PATH, SAMPLE_RATE_RAVDESS, MAIN_FOLDER, \
    DURATION_AUDIO_RAVDESS

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(device)


def generate_dataset_features(dataset_path):
    """
    Generates dataset from features extracted from wav2vec2 and save it into a CSV file.
    :param dataset_path: string. The path for CSV file with the dataset containing labels and path to audio files.
    :return: None
    """
    path_df = pd.read_csv(dataset_path)
    df_features = pd.DataFrame(columns=['Emotion', 'Features Path', 'Timestamp'])

    dir_path = os.path.join(MAIN_FOLDER, 'datasets', 'RAVDESS_features')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    i = 0
    for idx, row in path_df.iterrows():
        features_np = generate_features(row['Path'])
        # Using timestamp as file name
        name_file = str(row['Timestamp']) + '.npy'
        file_path = os.path.join(dir_path, name_file)
        np.save(file_path, features_np, fix_imports=False)

        # creating and adding the new row on the dataframe
        new_row_dict = {'Emotion': row['Emotion'],
                        'Emotion Intensity': row['Emotion Intensity'],
                        'Gender': row['Gender'],
                        'Features Path': file_path,
                        'Timestamp': row['Timestamp']
                        }
        df_features = pd.concat([df_features, pd.DataFrame([new_row_dict])], ignore_index=True)
        i += 1
        if i >= 15:
            break

    df_features.to_csv(os.path.join(MAIN_FOLDER, 'datasets', 'ravdess_dataset_features_v2.csv'), index=False)


def generate_features(path_audio_file):
    """
    Function to generate the features from a wav file using a wav2vec2 model to generate the features.
    :param path_audio_file: the path where to find the wav audio file
    :return: nparray representation of the tensors, i.e., features extracted from wav file
    """

    waveform, sample_rate = torchaudio.load(path_audio_file)
    waveform = waveform.to(device)

    if SAMPLE_RATE_RAVDESS != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    waveform = truc_audio_signal(waveform)

    # Call wav2vec2 to extract features
    features, _ = model.extract_features(waveform)
    features_np = [f.detach().numpy().astype('float32') for f in features]
    return features_np


def truc_audio_signal(waveform):
    # Making sure all the audio have the same length of duration
    num_rows, sig_len = waveform.shape
    max_len = SAMPLE_RATE_RAVDESS * DURATION_AUDIO_RAVDESS

    if sig_len < max_len:
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with Zeros
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, waveform, pad_end), 1)
        return sig


if __name__ == '__main__':
    path_dataset = os.path.join(RAVDESS_DATA_PATH, 'ravdess_dataset_with_paths.csv')
    generate_dataset_features(path_dataset)
