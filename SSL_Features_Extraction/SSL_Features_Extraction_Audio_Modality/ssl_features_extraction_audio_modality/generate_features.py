# Python code here
from conf import CUSTOM_SETTINGS, OUTPUTS_FOLDER, EXPERIMENT_ID
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

    encoder = init_encoder(
        CUSTOM_SETTINGS["encoder_config"],
        CUSTOM_SETTINGS['encoder_config']['pretrained'] if (
            "pretrained_path" in CUSTOM_SETTINGS['encoder_config']
        ) else f"{OUTPUTS_FOLDER}/ssl_training/{EXPERIMENT_ID}_encoder.pt" if (
            "pretrained_same_experiment" in CUSTOM_SETTINGS['encoder_config'] and
            CUSTOM_SETTINGS['encoder_config']["pretrained_same_experiment"]
        ) else None
    )

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
            OUTPUTS_FOLDER,
            CUSTOM_SETTINGS["ssl_config"]["input_type"],
            "SSL_features",
            transforms
        )


if __name__ == '__main__':
    generate_ssl_features()
