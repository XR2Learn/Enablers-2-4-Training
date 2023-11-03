import os
import shutil
import tempfile
import unittest

import torch
from pytorch_lightning import Trainer

from supervised_audio_modality.classifiers.linear import LinearClassifier
from supervised_audio_modality.classifiers.mlp import MLPClassifier


class MLPClassifierTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.input_dim = 512
        self.hidden_dimensions = [256, 128, 64]
        self.out_size = 4
        self.mlp = MLPClassifier(
            in_size=self.input_dim,
            out_size=self.out_size,
            hidden=self.hidden_dimensions
        )

        self.input = torch.rand(self.batch_size, self.input_dim)

    def test_number_of_layers(self):
        self.assertEqual(len(self.mlp.hidden_blocks), len(self.hidden_dimensions))

    def test_forward_pass_shape(self):
        output = self.mlp(self.input)
        self.assertEqual(output.shape, (self.batch_size, self.out_size))

    def test_no_dropout(self):
        output_train = self.mlp(self.input)
        self.mlp.eval()
        output_eval = self.mlp(self.input)
        self.assertTrue(torch.allclose(output_train, output_eval))

    def test_dropout_applied(self):
        mlp_dropout = MLPClassifier(
            in_size=self.input_dim,
            out_size=self.out_size,
            hidden=self.hidden_dimensions,
            p_dropout=0.5
        )
        output_train = mlp_dropout(self.input)
        mlp_dropout.eval()
        output_eval = mlp_dropout(self.input)
        self.assertFalse(torch.allclose(output_train, output_eval))

    def test_correct_model_load(self):
        test_dir = tempfile.mkdtemp()
        model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

        trainer = Trainer(default_root_dir=test_dir)
        trainer.strategy.connect(self.mlp)

        trainer.save_checkpoint(model_path)

        mlp_loaded_lightning = MLPClassifier.load_from_checkpoint(model_path)

        # one way to check if models are the same is to check if they produce the same output for the same input
        self.mlp.eval()
        mlp_loaded_lightning.eval()

        output_mlp_default = self.mlp(self.input)
        output_mlp_lightning = mlp_loaded_lightning(self.input)

        self.assertTrue(torch.allclose(output_mlp_default, output_mlp_lightning))

        shutil.rmtree(test_dir)


class LinearClassifierTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.input_dim = 512
        self.out_size = 4
        self.mlp = LinearClassifier(
            in_size=self.input_dim,
            out_size=self.out_size,
        )

        self.input = torch.rand(self.batch_size, self.input_dim)

    def test_forward_pass_shape(self):
        output = self.mlp(self.input)
        self.assertEqual(output.shape, (self.batch_size, self.out_size))

    def test_correct_model_load(self):
        test_dir = tempfile.mkdtemp()
        model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

        trainer = Trainer(default_root_dir=test_dir)
        trainer.strategy.connect(self.mlp)

        trainer.save_checkpoint(model_path)

        mlp_loaded_lightning = LinearClassifier.load_from_checkpoint(model_path)

        # one way to check if models are the same is to check if they produce the same output for the same input
        self.mlp.eval()
        mlp_loaded_lightning.eval()

        output_mlp_default = self.mlp(self.input)
        output_mlp_lightning = mlp_loaded_lightning(self.input)

        self.assertTrue(torch.allclose(output_mlp_default, output_mlp_lightning))

        shutil.rmtree(test_dir)
