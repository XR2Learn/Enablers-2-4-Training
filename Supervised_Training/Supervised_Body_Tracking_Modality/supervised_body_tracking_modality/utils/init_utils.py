import importlib
from typing import Any, Dict, Optional


def init_encoder(model_cfg: Dict[str, Any], ckpt_path: Optional[str] = None):
    """ Initialize (pre-trained) encoder from model configuration

    Parameters
    ----------
    model_cfg : dict
        configurations of the model: architecture and hyperparameters
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
