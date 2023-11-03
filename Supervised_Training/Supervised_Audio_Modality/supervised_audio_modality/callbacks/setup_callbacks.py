from typing import Optional

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .log_classifier_metrics import LogClassifierMetrics


def setup_callbacks(
        early_stopping_metric: str,
        no_ckpt: bool,
        num_classes: int = 3,
        patience: int = 100,
        dirpath: Optional[str] = None
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
    callbacks.append(setup_classifier_metrics_logger(num_classes=num_classes))
    if not no_ckpt:
        callbacks.append(setup_model_checkpoint_callback_last(dirpath=dirpath))
    return callbacks


def setup_model_checkpoint_callback_last(dirpath: Optional[str] = None):
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


def setup_classifier_metrics_logger(
        num_classes,
        metric_names=['accuracy', 'f1-score-macro', 'f1-score-micro', 'f1-score-weighted', 'precision', 'recall'],
        average='macro'
):
    """ Setup classifier metrics

    Parameters
    ----------
    num_classes : int
        number of classes
    metric_names : list, optional
        metrics to log, by default ['accuracy', 'f1-score','f1-score-micro','f1-score-weighted', 'precision', 'recall']
    average : str, optional
        type of averaging, by default 'macro'

    Returns
    -------
    callbacks.log_classifier_metrics.LogClassifierMetrics
        initialized callback
    """
    return LogClassifierMetrics(num_classes, metric_names, average=average)
