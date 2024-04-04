# Python code here
import os

from conf import CUSTOM_SETTINGS, MODALITY_FOLDER, EXPERIMENT_ID
from generate_and_save import generate_and_save
from utils.init_utils import init_encoder, init_transforms


def generate_ssl_features():
    """
    Function to extract SSL features and save to disk
    Args:
        None
    Returns:
        None
    """

    print(CUSTOM_SETTINGS)

    # currently, use custom train, val, test csv paths
    data_paths = ["train.csv", "val.csv", "test.csv"]

    if "pretrained_path" in CUSTOM_SETTINGS['encoder_config']:
        ckpt_path = CUSTOM_SETTINGS['encoder_config']['pretrained_path']
    elif (
        "pretrained_same_experiment" in CUSTOM_SETTINGS['encoder_config'] and
        CUSTOM_SETTINGS['encoder_config']["pretrained_same_experiment"]
    ):
        modality = CUSTOM_SETTINGS['dataset_config']['modality'] if (
            'modality' in CUSTOM_SETTINGS['dataset_config']
        ) else 'default_modality'

        ckpt_name = (
            f"{EXPERIMENT_ID}_"
            f"{CUSTOM_SETTINGS['dataset_config']['dataset_name']}_"
            f"{modality}_"
            f"{CUSTOM_SETTINGS['ssl_config']['input_type']}_"
            f"{CUSTOM_SETTINGS['encoder_config']['class_name']}_"
        )
        ckpt_path = os.path.join(
            MODALITY_FOLDER,
            "ssl_training",
            f"{ckpt_name}_encoder.pt"
        )
    else:
        raise ValueError("Pre-trained model checkpoint is not provided.")

    try:
        encoder = init_encoder(
            CUSTOM_SETTINGS["encoder_config"],
            ckpt_path
        )
    except:
        raise ValueError("""
                         The encoder cannot be initialized from the provided checkpoint.
                         Make sure that checkpoint model architecture matches the model from configuration
                         """)
    encoder.eval()
    print(encoder)

    if 'transforms' in CUSTOM_SETTINGS.keys():
        train_transforms, test_transforms = init_transforms(CUSTOM_SETTINGS['transforms'])
        transforms = {
            "train": train_transforms,
            "val": train_transforms,
            "test": test_transforms
        }

    for path_ in data_paths:
        generate_and_save(
            encoder,
            path_,
            MODALITY_FOLDER,
            CUSTOM_SETTINGS["ssl_config"]["input_type"],
            f"ssl_features_{os.path.basename(ckpt_path).split('.')[0]}",
            transforms
        )


if __name__ == '__main__':
    generate_ssl_features()
