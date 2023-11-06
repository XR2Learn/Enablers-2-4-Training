import importlib
from typing import Any, Dict, Optional

<<<<<<< HEAD

from utils.augmentations.compose_random_augmentations import compose_random_augmentations
=======
>>>>>>> 5aa5d5e (Refactor init_utils and add unittests)
from torchvision import transforms

from .augmentations import compose_random_augmentations


def init_encoder(model_cfg: Dict[str, Any], ckpt_path: Optional[str] = None):
    """ Initialize (pre-trained) encoder from model configuration

    Parameters
    ----------
    model_cfg : dict
        configurations of the model: architecture (e.g., supported by emotion recognition toolbox) and hyperparameter
    ckpt_path : str, optional
        path to the file with the pre-trained model, by default None

    Returns
    -------
    torch.nn.Module
        initialized encoder
    """
    module = importlib.import_module(f"{model_cfg['from_module']}")
    class_ = getattr(module, model_cfg['class_name'])
    return class_(**model_cfg['kwargs'], pretrained=ckpt_path)


# Transforms and augmentations
def init_transforms(transforms_cfg: Dict[str, Any]):
    """ Initialize transforms from the provided configs
    """
    train = []
    test = []
    if transforms_cfg is not None:
        for t in transforms_cfg:
            module = importlib.import_module(f"utils.{t['from_module']}")
            class_ = getattr(module, t['class_name'])

            if "kwargs" in t:
                transform = class_(**t['kwargs'])
            else:
                transform = class_()

            train.append(transform)
            if t['in_test']:
                test.append(transform)

            print(f"added {t['class_name']} transformation")

    composed_train_transform = transforms.Compose(train)
    composed_test_transform = transforms.Compose(test)

    return composed_train_transform, composed_test_transform


def init_augmentations(aug_dict: Dict[str, Any]):
    augmentations = None
    augmentations = compose_random_augmentations(aug_dict)
    augmentations = transforms.Compose(augmentations)
    return augmentations
