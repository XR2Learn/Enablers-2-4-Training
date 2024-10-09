"""
File to include global variables across the python package and configuration.
All the other files inside the python package can access these variables.
"""
from decouple import config
import os
import pathlib
import json

MAIN_FOLDER_DEFAULT = pathlib.Path(__file__).parent.parent.absolute()
MAIN_FOLDER = config('MAIN_FOLDER', default=MAIN_FOLDER_DEFAULT)
outputs_folder = os.path.join(MAIN_FOLDER, 'outputs')
OUTPUTS_FOLDER = config('OUTPUTS_FOLDER', default=outputs_folder)
EXPERIMENT_ID = config('EXPERIMENT_ID', default='dev_model')
datasets_folder = os.path.join(MAIN_FOLDER, 'datasets')
DATASETS_FOLDER = config('DATASETS_FOLDER', default=datasets_folder)

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
MODALITY = CUSTOM_SETTINGS["dataset_config"].get("modality", "default_modality")

if type(MODALITY) is list and "shimmer" in MODALITY:
    MODALITY = "shimmer"

MODALITY_FOLDER = os.path.join(
    OUTPUTS_FOLDER,
    CUSTOM_SETTINGS["dataset_config"]["dataset_name"],
    MODALITY,
)

COMPONENT_OUTPUT_FOLDER = os.path.join(
    MODALITY_FOLDER,
    'ssl_feature_extraction'
)


BM_LABEL_TO_EMOTION = {
    "01": "BORED",
    "02": "ENGAGED",
    "03": "FRUSTRATED"
}
BM_EMOTION_TO_LABEL = {
    "BORED": "01",
    "ENGAGED": "02",
    "FRUSTRATED": "03"
}
