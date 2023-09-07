import pathlib
import os.path
import json
from decouple import config

EMOTIONS_RAVDESS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                    5: 'angry', 6: 'fear', 7: 'disgust',
                    8: 'surprise'}
EMOTION_INTENSITY_RAVDESS = {1: 'normal', 2: "strong"}

SAMPLE_RATE_RAVDESS = 48000
DURATION_AUDIO_RAVDESS = 3

MAIN_FOLDER_DEFAULT = pathlib.Path(__file__).parent.parent.absolute()
MAIN_FOLDER = config('MAIN_FOLDER', default=MAIN_FOLDER_DEFAULT)
DATA_PATH = os.path.join(MAIN_FOLDER_DEFAULT, 'datasets', 'RAVDESS', 'audio_speech_actors_01-24')
RAVDESS_DATA_PATH = os.path.join(MAIN_FOLDER_DEFAULT, 'datasets', 'RAVDESS')

# Yet to check if this is really necessary, maybe only for cases where passing values as ENV VARS is too cumbersome
# e.g. [[1, 'a', ],['789', 'o', 9]] would be very annoying to write and parse.
CUSTOM_SETTINGS = {
    'key': {
        'default': 'value',
    }
}
PATH_CUSTOM_SETTINGS = os.path.join(MAIN_FOLDER, 'configuration.json')
if os.path.exists(PATH_CUSTOM_SETTINGS):
    with open(PATH_CUSTOM_SETTINGS, 'r') as f:
        CUSTOM_SETTINGS = json.load(f)
