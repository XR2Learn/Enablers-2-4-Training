import torch
import torchaudio
import pandas as pd
import os
import glob

from conf import DATA_PATH, EMOTIONS, EMOTION_INTENSITY, RAVDESS_DATA_PATH

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_and_save_dataset():
    """
    Function to create and save a dataset in CSV form from the RAVDESS dataset.
    """
    columns_df = ['Emotion', 'Emotion Intensity', 'Gender', 'Path', 'Timestamp']
    df = pd.DataFrame(columns=columns_df)

    path_files = glob.glob(os.path.join(DATA_PATH, '*', '*.wav'))

    for path_file in path_files:
        file_name = os.path.basename(path_file)
        file_name = file_name.replace('.wav', '')
        identifiers = file_name.split('-')
        if int(identifiers[6]) % 2 == 0:
            gender = 'female'
        else:
            gender = 'male'
        new_row_dict = {'Emotion': EMOTIONS[int(identifiers[2])],
                        'Emotion Intensity': EMOTION_INTENSITY[int(identifiers[3])],
                        'Gender': gender,
                        'Path': path_file
                        }
        # new_row = pd.DataFrame.from_dict(new_row_dict, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([new_row_dict])], ignore_index=True)
    df.to_csv(os.path.join(RAVDESS_DATA_PATH, 'ravdess_dataset_with_paths.csv'), index=False)


def generate_dataset_features(path_dataset):
    """
    Generates dataset from features extracted from wav2vec2 and save it into a CSV file.
    :param path_dataset: string. The path for CSV file with the dataset containing labels and path to audio files.
    :return: None
    """
    path_df = pd.read_csv(path_dataset)
    df_features = pd.DataFrame(columns=['Emotion', 'Features'])
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model()
    for row in path_df:
        new_row_features = {}
        waveform, sample_rate = torchaudio.load(row['Path'])
        features = model.extract_features(waveform)
        # Call wav2vec2 to extract features
        df_features = pd.concat([df_features, pd.DataFrame([new_row_features])], ignore_index=True)
        pass

    df_features.to_csv(os.path.join(RAVDESS_DATA_PATH, 'ravdess_dataset_features.csv'), index=False)


if __name__ == '__main__':
    # create_and_save_dataset()
    path_dataset = os.path.join(RAVDESS_DATA_PATH, 'ravdess_dataset_with_paths.csv')
    generate_dataset_features(path_dataset)
