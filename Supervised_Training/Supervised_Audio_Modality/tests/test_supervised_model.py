import os
import shutil
import tempfile
import unittest

import torch
from pytorch_lightning import Trainer

from supervised_audio_modality.classifiers.mlp import MLPClassifier
from supervised_audio_modality.encoders.cnn1d import CNN1D
from supervised_audio_modality.classification_model import SupervisedModel


class SupervisedTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.in_channels = 10
        self.len_seq = 1000
        self.out_channels = [16, 32, 64]
        self.kernel_sizes = [3, 5, 7]
        self.hidden_dimensions = [256, 128, 64]
        self.out_size = 4
        self.cnn = CNN1D(
            in_channels=self.in_channels,
            len_seq=self.len_seq,
            out_channels=self.out_channels,
            kernel_sizes=self.kernel_sizes
        )

        self.mlp = MLPClassifier(
            in_size=self.cnn.out_size,
            out_size=self.out_size,
            hidden=self.hidden_dimensions,
        )

        self.input = torch.rand(self.batch_size, self.in_channels, self.len_seq)

    def test_init_supervised_model_from_components(self):
        supervised_model = SupervisedModel(
            self.cnn,
            self.mlp,
            freeze_encoder=False
        )
        self.assertTrue(isinstance(supervised_model.classifier, MLPClassifier))
        self.assertTrue(isinstance(supervised_model.encoder, CNN1D))

    def test_forward_pass_shape(self):
        supervised_model = SupervisedModel(
            self.cnn,
            self.mlp,
            freeze_encoder=False
        )
        output = supervised_model(self.input)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.out_size)
        )

    def test_model_freeze(self):
        supervised_model = SupervisedModel(
            self.cnn,
            self.mlp,
            freeze_encoder=True
        )
        
        for param in supervised_model.encoder.parameters():
            self.assertFalse(param.requires_grad)

    def test_correct_model_load_from_checkpoint(self):
        supervised_model = SupervisedModel(
            self.cnn,
            self.mlp,
            freeze_encoder=False
        )
        test_dir = tempfile.mkdtemp()
        model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

        trainer = Trainer(default_root_dir=test_dir)
        trainer.strategy.connect(supervised_model)

        trainer.save_checkpoint(model_path)

        # Intialize new CNN and MLP with random weights
        new_cnn = CNN1D(
            in_channels=self.in_channels,
            len_seq=self.len_seq,
            out_channels=self.out_channels,
            kernel_sizes=self.kernel_sizes
        )

        new_mlp = MLPClassifier(
            in_size=self.cnn.out_size,
            out_size=self.out_size,
            hidden=self.hidden_dimensions,
        )

        # Use structure of these newly initialzied models to load weights from the pretrained supervised model
        # This simulates inference scenario,
        # i.e. when we initialize a model using previously saved weights of the whole supervised model
        supervised_model_loaded_lightning = SupervisedModel.load_from_checkpoint(
            model_path,
            encoder=new_cnn,
            classifier=new_mlp
        )

        # one way to check if models are the same is to check if they produce the same output for the same input
        supervised_model.eval()
        supervised_model_loaded_lightning.eval()

        output_supervised = supervised_model(self.input)
        output_supervised_lightning = supervised_model_loaded_lightning(self.input)

        self.assertTrue(torch.allclose(output_supervised, output_supervised_lightning))

        shutil.rmtree(test_dir)

    def test_correct_model_load_from_pretrained_encoder(self):
        supervised_model = SupervisedModel(
            self.cnn,
            self.mlp,
            freeze_encoder=False
        )

        test_dir = tempfile.mkdtemp()
        model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

        trainer = Trainer(default_root_dir=test_dir)
        trainer.strategy.connect(self.cnn)

        trainer.save_checkpoint(model_path)
        cnn_loaded_lightning = CNN1D.load_from_checkpoint(
            model_path,
        )

        supervised_model_encoder_checkpoint = SupervisedModel(
            cnn_loaded_lightning,
            self.mlp,
            freeze_encoder=False
        )

        # one way to check if models are the same is to check if they produce the same output for the same input
        supervised_model.eval()
        supervised_model_encoder_checkpoint.eval()

        output_supervised = supervised_model(self.input)
        output_supervised_encoder_checkpoint = supervised_model_encoder_checkpoint(self.input)

        self.assertTrue(torch.allclose(output_supervised, output_supervised_encoder_checkpoint))

        shutil.rmtree(test_dir)
