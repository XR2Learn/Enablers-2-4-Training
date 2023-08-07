import pathlib
import os.path
from decouple import config

EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fear', 7: 'disgust',
            8: 'surprise'}
SAMPLE_RATE = 48000

main_folder = pathlib.Path(__file__).parent.parent.absolute()
DATA_PATH = os.path.join(main_folder, 'datasets', 'RAVDESS', 'audio_speech_actors_01-24')
