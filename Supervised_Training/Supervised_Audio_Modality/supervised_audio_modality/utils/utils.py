import datetime
import shutil
import yaml


def load_yaml_to_dict(path):
    """ Parses yaml file into a dictionary

    Parameters
    ----------
    path : str
        path to yaml file

    Returns
    -------
    dict
        dictionary based on input YAML
    """    
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)


def generate_experiment_id():
    """ A function for generating unique experiment id based on the current time"""
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')


def copy_file(source, destination):
    """ Copies file from source to destination

    Parameters
    ----------
    source : str
        path to source file
    destination : str
        path to destination file or folder
    """    
    try:
        shutil.copy(source, destination)
        print("File copied successfully.")
    except shutil.SameFileError:
        print("Source and destination are the same file.")
    except PermissionError:
        print("Permission denied.")
    except:
        print("Error occurred while copying file.")