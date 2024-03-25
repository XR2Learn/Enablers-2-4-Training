"""
File to include global variables across the python package and configuration.
All the other files inside the python package can access these variables.
"""
from decouple import config
import os
import pathlib
import json

TESTING = 'Importing a variable from conf.py'

MAIN_FOLDER_DEFAULT = pathlib.Path(__file__).parent.parent.absolute()
MAIN_FOLDER = config('MAIN_FOLDER', default=MAIN_FOLDER_DEFAULT)
outputs_folder = os.path.join(MAIN_FOLDER, 'outputs')
OUTPUTS_FOLDER = config('OUTPUTS_FOLDER', default=outputs_folder)
EXPERIMENT_ID = config('EXPERIMENT_ID', default='dev_model')
datasets_folder = os.path.join(MAIN_FOLDER, 'outputs')
DATASETS_FOLDER = config('DATASETS_FOLDER', default=datasets_folder)
DATA_PATH = os.path.join(MAIN_FOLDER_DEFAULT, 'datasets', 'RAVDESS', 'audio_speech_actors_01-24')
RAVDESS_DATA_PATH = os.path.join(MAIN_FOLDER_DEFAULT, 'datasets', 'RAVDESS')

# Yet to check if this is really necessary, maybe only for cases where passing values as ENV VARS is too cumbersome
# e.g. [[1, 'a', ],['789', 'o', 9]] would be very annoying to write and parse.
CUSTOM_SETTINGS = {
    'dataset_config': {
        'dataset_name': 'default_dataset',
        'modality': 'default_modality',
    },
    'pre_processing': {
        'some_config_preprocessing': 'values',
    }
}

path_custom_settings = os.path.join(MAIN_FOLDER, 'configuration.json')
PATH_CUSTOM_SETTINGS = config('PATH_CUSTOM_SETTINGS', default=path_custom_settings)
if os.path.exists(PATH_CUSTOM_SETTINGS):
    with open(PATH_CUSTOM_SETTINGS, 'r') as f:
        CUSTOM_SETTINGS = json.load(f)


DATA_PATH = os.path.join(DATASETS_FOLDER, CUSTOM_SETTINGS["dataset_config"]["dataset_name"])
# Define components outputs folder
if "modality" in CUSTOM_SETTINGS["dataset_config"]:
    modality = CUSTOM_SETTINGS["dataset_config"]["modality"]
else:
    modality = "default_modality"

MODALITY_FOLDER = os.path.join(
    OUTPUTS_FOLDER,
    CUSTOM_SETTINGS["dataset_config"]["dataset_name"],
    modality,
)