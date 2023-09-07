# Python code here

from conf import CUSTOM_SETTINGS


def example_run():
    """
    A basic print function to verify if docker is running.
    :return: None
    """
    model_config = CUSTOM_SETTINGS['model']['some_config_model']
    print(f'Docker for Ed-Training Audio has run. Conf from configuration.json file: {model_config}')


if __name__ == '__main__':
    example_run()
