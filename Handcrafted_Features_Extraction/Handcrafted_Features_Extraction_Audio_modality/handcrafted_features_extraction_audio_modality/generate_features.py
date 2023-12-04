# Python code here
import os

from conf import CUSTOM_SETTINGS, OUTPUTS_FOLDER

from feature_extraction import extract_and_save_features


def extract_handcrafted_features():
    """
    Main function that extracts handcrafted features defined by/in the json configuration file
    """
    processed_data_folder = os.path.join(
        OUTPUTS_FOLDER,
        CUSTOM_SETTINGS['pre_processing_config']['process']
    )

    all_data_paths = os.listdir(processed_data_folder)
    print(f"Found a total of {len(all_data_paths)} files inside the {processed_data_folder} folder.")
    features_to_extract = list(CUSTOM_SETTINGS["handcrafted_features_config"])
    print(f"Features {features_to_extract}, if supported, will be extracted and saved.")

    full_data_paths = [os.path.join(processed_data_folder, data_path) for data_path in all_data_paths]

    extract_and_save_features(
        full_data_paths,
        OUTPUTS_FOLDER,
        CUSTOM_SETTINGS["handcrafted_features_config"]
    )


if __name__ == '__main__':
    extract_handcrafted_features()
