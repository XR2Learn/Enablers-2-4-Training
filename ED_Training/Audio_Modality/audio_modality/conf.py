import pathlib
import os.path
from decouple import config

EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fear', 7: 'disgust',
            8: 'surprise'}
EMOTION_INTENSITY = {1: 'normal', 2: "strong"}

SAMPLE_RATE = 48000

MAIN_FOLDER_DEFAULT = pathlib.Path(__file__).parent.parent.absolute()
MAIN_FOLDER = config('MAIN_FOLDER', default=MAIN_FOLDER_DEFAULT)
DATA_PATH = os.path.join(MAIN_FOLDER_DEFAULT, 'datasets', 'RAVDESS', 'audio_speech_actors_01-24')
RAVDESS_DATA_PATH = os.path.join(MAIN_FOLDER_DEFAULT, 'datasets', 'RAVDESS')
