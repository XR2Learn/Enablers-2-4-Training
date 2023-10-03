# Python code here

from conf import CUSTOM_SETTINGS


def generate_ssl_features():
    """
    A basic print function to verify if docker is running.
    :return: None
    """
    model_config = CUSTOM_SETTINGS['pre_processing']['some_config_preprocessing']
    print(f'Docker for features generation Audio has run (Enabler 3). Conf from configuration.json file: {model_config}')

    #load data

    #load model

    #iterate over data
        # get features
        # save features to disk
        # write to csv


if __name__ == '__main__':
    generate_ssl_features()
