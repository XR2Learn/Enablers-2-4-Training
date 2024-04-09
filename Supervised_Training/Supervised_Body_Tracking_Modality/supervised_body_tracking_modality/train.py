# Python code here

from conf import PATH_CUSTOM_SETTINGS


def example_run():
    """
    A basic print function to verify if docker is running.
    :return: None
    """
    print(
        f'Docker for Ed-Training Body Tracking has run (Enabler 4). Conf from configuration.json file: {PATH_CUSTOM_SETTINGS}')


if __name__ == '__main__':
    example_run()
