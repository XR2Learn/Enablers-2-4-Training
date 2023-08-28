import os
import glob
import pandas as pd

from conf import DATA_PATH, EMOTIONS_RAVDESS, EMOTION_INTENSITY_RAVDESS, RAVDESS_DATA_PATH


def create_and_save_dataset_audio_paths():
    """
    Function to create and save a dataset in CSV form from the RAVDESS dataset wav files, including labels.
    It parses the files' names and organise the information into a CSV file. The columns of the CSV includes:
    Emotion: labels of the data between seven emotions.
    Emotion Intensity: intensity of the emotion expressed by subject.
    Gender: the subject's gender.
    Path: path to the wav audio file.
    Timestamp: A unique identifier to later synchronise different modalities.
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
        new_row_dict = {'Emotion': EMOTIONS_RAVDESS[int(identifiers[2])],
                        'Emotion Intensity': EMOTION_INTENSITY_RAVDESS[int(identifiers[3])],
                        'Gender': gender,
                        'Path': path_file,
                        'Timestamp': timestamp
                        }
        df = pd.concat([df, pd.DataFrame([new_row_dict])], ignore_index=True)
    if not os.path.exists(RAVDESS_DATA_PATH):
        os.makedirs(RAVDESS_DATA_PATH)
    df.to_csv(os.path.join(RAVDESS_DATA_PATH, 'ravdess_dataset_with_paths.csv'), index=False)


if __name__ == '__main__':
    create_and_save_dataset_audio_paths()
