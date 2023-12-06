import os
import shutil
import tempfile
import unittest

import torch

from pytorch_lightning import LightningModule, Trainer

from supervised_audio_modality.encoders.cnn1d import CNN1D
from supervised_audio_modality.encoders.w2v import Wav2Vec2CNN


class CNN1DTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.in_channels = 10
        self.len_seq = 1000
        self.out_channels = [16, 32, 64, 128, 256]
        self.kernel_sizes = [3, 5, 7, 5, 3]
        self.cnn = CNN1D(
            in_channels=self.in_channels,
            len_seq=self.len_seq,
            out_channels=self.out_channels,
            kernel_sizes=self.kernel_sizes
        )

        self.input = torch.rand(self.batch_size, self.in_channels, self.len_seq)

    def test_number_of_layers(self):
        self.assertEqual(len(self.cnn.convolutional_blocks), self.cnn.num_layers)

    def test_output_size_computation(self):
        # Test 1
        out_size1 = self.cnn._compute_out_size(
            sample_length=20,
            padding=1,
            kernel_sizes=[3],
            stride=1,
            num_layers=1,
            num_channels=10,
            pool_size=2,
            pool_padding=2,
        )
        # after conv: (20 + 2 * 1 - 3) / 1 + 1 = 20
        # after pool: (20 + 2 * 2) / 2 = 12
        # flattened: 12 * 10 = 120
        self.assertEqual(out_size1, 120)

        # Test 2
        out_size2 = self.cnn._compute_out_size(
            sample_length=20,
            padding=1,
            kernel_sizes=[3, 5],
            stride=1,
            num_layers=2,
            num_channels=20,
            pool_size=2,
            pool_padding=2,
        )

        # Test 3
        # after layer 1: (12, 10)
        # layer 2 conv: (12 + 2 * 1 - 5) / 1 + 1 = 10
        # layer 2 pool: (10 + 2 * 2) / 2 = 7
        # flattened: 7 * 20 = 140
        self.assertEqual(out_size2, 140)

        out_size3 = self.cnn._compute_out_size(
            sample_length=100,
            padding=2,
            kernel_sizes=[5, 3, 7],
            stride=2,
            num_layers=3,
            num_channels=64,
            pool_size=2,
            pool_padding=0,
        )
        # layer 1: (100 + 2 * 2 - 5) / 2 + 1 = 50 / 2 = 25
        # layer 2: (25 + 2 * 2 - 3) / 2 + 1 = 14 / 2 = 7
        # layer 3: (7 + 2 * 2 - 7) / 2 + 1 = 3 / 2 = 1
        # flattened: 1 * 64 = 64
        self.assertEqual(out_size3, 64)

    def test_forward_pass_shape(self):
        output = self.cnn(self.input)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.out_channels[-1], self.cnn.out_size // self.out_channels[-1])
        )

    def test_correct_model_load(self):
        test_dir = tempfile.mkdtemp()
        model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

        trainer = Trainer(default_root_dir=test_dir)
        trainer.strategy.connect(self.cnn)

        trainer.save_checkpoint(model_path)
        cnn_loaded_from_state_dict = CNN1D(
            in_channels=self.in_channels,
            len_seq=self.len_seq,
            out_channels=self.out_channels,
            kernel_sizes=self.kernel_sizes,
            pretrained=model_path,
        )
        cnn_loaded_lightning = CNN1D.load_from_checkpoint(
            model_path,
        )

        # one way to check if models are the same is to check if they produce the same output for the same input
        self.cnn.eval()
        cnn_loaded_from_state_dict.eval()
        cnn_loaded_lightning.eval()

        output_cnn_default = self.cnn(self.input)
        output_cnn_from_state_dict = cnn_loaded_from_state_dict(self.input)
        output_cnn_lightning = cnn_loaded_from_state_dict(self.input)

        self.assertTrue(
            torch.allclose(output_cnn_default, output_cnn_from_state_dict),
            "Unexpected result for a model loaded from state_dict()"
        )
        self.assertTrue(
            torch.allclose(output_cnn_default, output_cnn_lightning),
            "Unexpected result for a model loaded from lightning checkpoint"
        )

        shutil.rmtree(test_dir)


class Wav2Vec2CNNTestCase(unittest.TestCase):
    def setUp(self):
        # small batch size to make test less computationally expensive
        self.batch_size = 2
        self.length_samples = 10
        self.sample_rate = 16000
        self.out_channels = [128, 128]
        self.kernel_sizes = [1, 1]
        self.w2v2_model = Wav2Vec2CNN(
            length_samples=self.length_samples,
            sample_rate=self.sample_rate,
            w2v2_type='base',
            freeze=True,
            out_channels=self.out_channels,
            kernel_sizes=self.kernel_sizes,
        )

        self.input = torch.rand(self.batch_size, self.length_samples * self.sample_rate)

    def test_init_correct(self):
        self.assertTrue(isinstance(self.w2v2_model.wav2vec2, LightningModule))
        self.assertTrue(isinstance(self.w2v2_model.cnn, LightningModule))

    def test_forward_pass_shape(self):
        output = self.w2v2_model(self.input)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.w2v2_model.out_size)
        )

    def test_correct_model_load(self):
        test_dir = tempfile.mkdtemp()
        model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

        trainer = Trainer(default_root_dir=test_dir)
        trainer.strategy.connect(self.w2v2_model)

        trainer.save_checkpoint(model_path)
        w2v2_loaded_from_state_dict = Wav2Vec2CNN(
            length_samples=self.length_samples,
            sample_rate=self.sample_rate,
            w2v2_type='base',
            freeze=True,
            out_channels=self.out_channels,
            kernel_sizes=self.kernel_sizes,
            pretrained=model_path
        )
        w2v2_loaded_lightning = Wav2Vec2CNN.load_from_checkpoint(
            model_path,
        )

        # one way to check if models are the same is to check if they produce the same output for the same input
        self.w2v2_model.eval()
        w2v2_loaded_from_state_dict.eval()
        w2v2_loaded_lightning.eval()

        output_w2v2_default = self.w2v2_model(self.input)
        output_w2v2_from_state_dict = w2v2_loaded_from_state_dict(self.input)
        output_w2v2_lightning = w2v2_loaded_lightning(self.input)

        self.assertTrue(
            torch.allclose(output_w2v2_default, output_w2v2_from_state_dict),
            "Unexpected result for a model loaded from state_dict()"
        )
        self.assertTrue(
            torch.allclose(output_w2v2_default, output_w2v2_lightning),
            "Unexpected result for a model loaded from lightning checkpoint"
        )

        shutil.rmtree(test_dir)


class LargeWav2Vec2CNNTestCase(unittest.TestCase):
    def setUp(self):
        # small batch size to make test less computationally expensive
        self.batch_size = 2
        self.length_samples = 10
        self.sample_rate = 16000
        self.out_channels = [128, 128]
        self.kernel_sizes = [1, 1]
        self.w2v2_model = Wav2Vec2CNN(
            length_samples=self.length_samples,
            sample_rate=self.sample_rate,
            w2v2_type='large',
            freeze=True,
            out_channels=self.out_channels,
            kernel_sizes=self.kernel_sizes,
        )

        self.input = torch.rand(self.batch_size, self.length_samples * self.sample_rate)

    def test_init_correct(self):
        self.assertTrue(isinstance(self.w2v2_model.wav2vec2, LightningModule))
        self.assertTrue(isinstance(self.w2v2_model.cnn, LightningModule))

    def test_forward_pass_shape(self):
        output = self.w2v2_model(self.input)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.w2v2_model.out_size)
        )

    def test_correct_model_load(self):
        test_dir = tempfile.mkdtemp()
        model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

        trainer = Trainer(default_root_dir=test_dir)
        trainer.strategy.connect(self.w2v2_model)

        trainer.save_checkpoint(model_path)
        w2v2_loaded_from_state_dict = Wav2Vec2CNN(
            length_samples=self.length_samples,
            sample_rate=self.sample_rate,
            w2v2_type='large',
            freeze=True,
            out_channels=self.out_channels,
            kernel_sizes=self.kernel_sizes,
            pretrained=model_path
        )
        w2v2_loaded_lightning = Wav2Vec2CNN.load_from_checkpoint(
            model_path,
        )

        # one way to check if models are the same is to check if they produce the same output for the same input
        self.w2v2_model.eval()
        w2v2_loaded_from_state_dict.eval()
        w2v2_loaded_lightning.eval()

        output_w2v2_default = self.w2v2_model(self.input)
        output_w2v2_from_state_dict = w2v2_loaded_from_state_dict(self.input)
        output_w2v2_lightning = w2v2_loaded_lightning(self.input)

        self.assertTrue(
            torch.allclose(output_w2v2_default, output_w2v2_from_state_dict),
            "Unexpected result for a model loaded from state_dict()"
        )
        self.assertTrue(
            torch.allclose(output_w2v2_default, output_w2v2_lightning),
            "Unexpected result for a model loaded from lightning checkpoint"
        )

        shutil.rmtree(test_dir)
