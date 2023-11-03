import pytorch_lightning as pl
from torch import nn
import torch
import torchmetrics


class LogClassifierMetrics(pl.Callback):
    """
    A callback which logs one or more classifier-specific metrics at the end of each validation and test epoch, 
    to all available loggers. The available metrics are: accuracy, precision, recall, F1-score.

    The class implements the necessary pytorch_lightning.Callback hooks.
    """
    def __init__(
            self,
            num_classes,
            metric_names=['accuracy', 'f1-score-macro', 'f1-score-micro', 'f1-score-weighted', 'precision', 'recall'],
            average='macro',
    ):
        self.metric_names = metric_names
        self.task = 'binary' if num_classes <= 2 else 'multiclass'
        self.metric_dict = nn.ModuleDict({
            'accuracy': torchmetrics.Accuracy(num_classes=num_classes, task=self.task),
            'f1-score-macro': torchmetrics.F1Score(num_classes=num_classes, task=self.task, average='macro'),
            'f1-score-micro': torchmetrics.F1Score(num_classes=num_classes, task=self.task, average='micro'),
            'f1-score-weighted': torchmetrics.F1Score(num_classes=num_classes, task=self.task, average='weighted'),
            'precision': torchmetrics.Precision(num_classes=num_classes, task=self.task, average=average),
            'recall': torchmetrics.Recall(num_classes=num_classes, task=self.task, average=average)
        })
        self._reset_state()

    def _reset_state(self):
        self.labels = []
        self.preds = []

    def on_test_epoch_start(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule
    ) -> None:
        self._reset_state()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._reset_state()

    def _cache_preds_labels(self, outputs, batch):
        self.labels += batch[1].tolist()
        self.preds += outputs['preds'].tolist()

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs, batch,
            batch_idx,
            dataloader_idx
    ) -> None:
        self._cache_preds_labels(outputs, batch)

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs, batch,
            batch_idx,
            dataloader_idx
    ) -> None:
        self._cache_preds_labels(outputs, batch)

    def _shared_eval(self, trainer, prefix):
        labels_tensor = torch.Tensor(self.labels).int()
        preds_tensor = torch.Tensor(self.preds).int()
        for metric_name in self.metric_names:
            if metric_name in self.metric_dict:
                metric_val = self.metric_dict[metric_name](preds_tensor, labels_tensor)
                self.log(f"{prefix}_{metric_name}", metric_val)

    def on_validation_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule
    ) -> None:
        self._shared_eval(trainer, "val")
