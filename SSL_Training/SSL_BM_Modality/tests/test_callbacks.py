import shutil
import tempfile
import unittest


from pytorch_lightning import Callback

from ssl_bm_modality.callbacks.setup_callbacks import setup_callbacks, setup_model_checkpoint_callback


class SetupCallbacksTestCase(unittest.TestCase):
    def test_setup_all_calbacks(self):
        callbacks = setup_callbacks(
            early_stopping_metric="val_loss",
            no_ckpt=False,
            patience=50,
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
        shutil.rmtree(test_dir)
