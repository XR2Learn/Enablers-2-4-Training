import shutil
import tempfile
import unittest

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import torch
from pytorch_lightning import Callback, LightningModule

from supervised_audio_modality.callbacks.setup_callbacks import setup_callbacks, setup_model_checkpoint_callback
from supervised_audio_modality.callbacks.log_classifier_metrics import LogClassifierMetrics


class SetupCallbacksTestCase(unittest.TestCase):
    def test_setup_all_calbacks(self):
        callbacks = setup_callbacks(
            early_stopping_metric="val_loss",
            no_ckpt=False,
            patience=50,
            num_classes=3
        )
        for callback in callbacks:
            self.assertTrue(isinstance(callback, Callback))

    def test_checkpoint_setup(self):
        test_dir = tempfile.mkdtemp()
        model_filename = "test_checkpoint.ckpt"
        # Check correct saving modes
        monitor_loss = "val_loss"
        checkpoint_callback_loss = setup_model_checkpoint_callback(
            dirpath=test_dir,
            monitor=monitor_loss,
            checkpoint_filename=model_filename,
            mode="min" if "loss" in monitor_loss else False
        )
        self.assertTrue(checkpoint_callback_loss.mode == "min", "Wrong mode: expected min for loss")

        monitor_acc = "val_accuracy"
        checkpoint_callback_acc = setup_model_checkpoint_callback(
            dirpath=test_dir,
            monitor=monitor_acc,
            checkpoint_filename=model_filename,
            mode="min" if "loss" in monitor_acc else "max"
        )
        self.assertTrue(checkpoint_callback_acc.mode == "max", "Wrong mode: expected max for accuracy")

        shutil.rmtree(test_dir)

    def test_metrics_callback(self):
        metric_names = ['accuracy', 'f1-score-macro', 'f1-score-micro', 'f1-score-weighted', 'precision', 'recall']
        average = "macro"
        metrics_callback = LogClassifierMetrics(
            num_classes=3,
            metric_names=['accuracy', 'f1-score-macro', 'f1-score-micro', 'f1-score-weighted', 'precision', 'recall'],
            average=average
        )
        # Test 1: perfect classification performance
        test_labels = torch.tensor(
            [
                [0, 0, 0, 1, 2],
                [1, 0, 1, 2, 1],
                [2, 2, 2, 1, 0]
            ]
        )
        test_preds = torch.tensor(
            [
                [0, 0, 0, 1, 2],
                [1, 0, 1, 2, 1],
                [2, 2, 2, 1, 0]
            ]
        )
        for i in range(test_preds.shape[0]):
            batch_test_preds = {"preds": test_preds[i]}
            batch_test_labels = (None, test_labels[i])
            metrics_callback.on_validation_batch_end(None, None, batch_test_preds, batch_test_labels, None, None)

        class DummyModule(LightningModule):
            def __init__(self):
                super().__init__()

        lm = DummyModule()

        metrics = metrics_callback._shared_eval(lm, "val")
        for metric in metrics:
            self.assertTrue(metric.startswith("val_"))
            self.assertTrue(metric.split("_")[1] in metric_names)
            self.assertTrue(metrics[metric] == 1)

        # Test 2: check with scikit-learn metrics
        test_preds = torch.randint(0, 3, (50, 64))
        test_labels = torch.randint(0, 3, (50, 64))

        for i in range(test_preds.shape[0]):
            batch_test_preds = {"preds": test_preds[i]}
            batch_test_labels = (None, test_labels[i])
            metrics_callback.on_validation_batch_end(None, None, batch_test_preds, batch_test_labels, None, None)

        metrics = metrics_callback._shared_eval(lm, "val")
        # stack all predictions and labels into single arrays
        test_preds = test_preds.reshape(-1, 1).cpu().numpy()
        test_labels = test_labels.reshape(-1, 1).cpu().numpy()

        self.assertAlmostEqual(
            accuracy_score(test_labels, test_preds),
            float(metrics["val_accuracy"]),
            delta=0.01
        )
        self.assertAlmostEqual(
            recall_score(test_labels, test_preds, average="macro"),
            float(metrics["val_recall"]),
            delta=0.01
        )
        self.assertAlmostEqual(
            precision_score(test_labels, test_preds, average="macro"),
            float(metrics["val_precision"]),
            delta=0.01
        )
        for avr in ["micro", "macro", "weighted"]:
            self.assertAlmostEqual(
                f1_score(test_labels, test_preds, average=f"{avr}"),
                float(metrics[f"val_f1-score-{avr}"]),
                delta=0.01
            )
