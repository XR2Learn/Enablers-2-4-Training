import torch
import torchaudio
import pandas as pd
import os
import glob

from conf import DATA_PATH, EMOTIONS

data_frame = pd.DataFrame(columns=['Emotion', 'Emotion intensity', 'Gender', 'Path', 'Timestamp'])

path_files = glob.glob(os.path.join(DATA_PATH, '*', '*.wav'))

for path_file in path_files:
    file_name = os.path.basename(path_file)

for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        file_path = os.path.join('/kaggle/input/', dirname, filename)
        identifiers = filename.split('.')[0].split('-')
        emotion = (int(identifiers[2]))
        if emotion == 8:
            emotion = 0
        if int(identifiers[3]) == 1:
            emotion_intensity = 'normal'
        else:
            emotion_intensity = 'strong'
        if int(identifiers[6]) % 2 == 0:
            gender = 'female'
        else:
            gender = 'male'

        data_frame = data_frame.append({"Emotion": emotion,
                                        "Emotion intensity": emotion_intensity,
                                        "Gender": gender,
                                        "Path": file_path
                                        },
                                       ignore_index=True
                                       )
