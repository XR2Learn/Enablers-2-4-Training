import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def setup_callbacks(
        early_stopping_metric,
        no_ckpt, 
        patience=100,
        dirpath=None
    ):
    """ Setup callbacks for pytorch_lightning.Trainer

    Parameters
    ----------
    early_stopping_metric : str
        name of the metric for early stopping
    no_ckpt : bool
        flag for not using checkpoints. If True, the model is not saved.
    patience : int, optional
        number of epochs with no improvement for early stopping, by default 100

    Returns
    -------
    list
        list of callbacks
    """    
    callbacks = []
    mode = 'min' if 'loss' in early_stopping_metric else 'max'
    callbacks.append(setup_early_stopping_callback(early_stopping_metric, mode=mode, patience=patience))
    if not no_ckpt:
        callbacks.append(setup_model_checkpoint_callback_last(dirpath=dirpath))
    return callbacks


def setup_model_checkpoint_callback_last(dirpath=None):
    """ Setup last epoch model checkpoint

    Returns
    -------
    pytorch_lightning.callbacks.ModelCheckpoint
        initialized callback
    """
    return ModelCheckpoint(
        dirpath=dirpath,
        save_last=True,
        filename="{epoch}"
    )	


def setup_early_stopping_callback(
        metric, 
        min_delta=0.00, 
        patience=50, 
        mode="min"
    ):
    """ Initialize early stopping callback

    Parameters
    ----------
    metric : str
        _metric to monitor
    min_delta : float, optional
        minimum change in the monitored quantity to qualify as an improvement, by default 0.00
    patience : int, optional
        number of epochs with no improvement, by default 50
    mode : str, optional
        mode: min or max, by default "min"

    Returns
    -------
    pytorch_lightning.callbacks.EarlyStopping
        initialized callback
    """    
    return EarlyStopping(monitor=metric, min_delta=min_delta, patience=patience, verbose=False, mode=mode)