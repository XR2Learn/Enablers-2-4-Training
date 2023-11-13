from typing import Optional

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def setup_callbacks(
        early_stopping_metric: str,
        no_ckpt: bool,
        patience: int = 100,
        dirpath: Optional[str] = None,
        monitor: str = "val_loss",
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        checkpoint_filename: Optional[str] = None,
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
    mode_early_stopping = 'min' if 'loss' in early_stopping_metric else 'max'
    mode_checkpoint = 'min' if 'loss' in monitor else 'max'
    callbacks.append(setup_early_stopping_callback(early_stopping_metric, mode=mode_early_stopping, patience=patience))
    if not no_ckpt:
        callbacks.append(setup_model_checkpoint_callback(
            dirpath=dirpath,
            monitor=monitor,
            save_last=save_last,
            save_top_k=save_top_k,
            checkpoint_filename=checkpoint_filename,
            mode=mode_checkpoint,
        ))
    return callbacks


def setup_model_checkpoint_callback(
        dirpath: Optional[str] = None,
        monitor: str = "val_loss",
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        checkpoint_filename: Optional[str] = None,
        mode: str = "min",
):
    """ Setup last epoch model checkpoint

    Returns
    -------
    pytorch_lightning.callbacks.ModelCheckpoint
        initialized callback
    """
    return ModelCheckpoint(
        dirpath=dirpath,
        monitor=monitor,
        save_last=save_last,
        save_top_k=save_top_k,
        filename=checkpoint_filename,
        mode=mode
    )


def setup_early_stopping_callback(
        metric: str,
        min_delta: float = 0.00,
        patience: int = 50,
        mode: str = "min"
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
