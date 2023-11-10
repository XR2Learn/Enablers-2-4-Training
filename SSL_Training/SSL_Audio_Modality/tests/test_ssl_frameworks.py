import os
import shutil
import tempfile
import unittest

import torch
from pytorch_lightning import Trainer

from ssl_audio_modality.encoders.cnn1d import CNN1D
from ssl_audio_modality.encoders.w2v import Wav2Vec2CNN
from ssl_audio_modality.ssl_methods.SimCLR import SimCLR
from ssl_audio_modality.ssl_methods.VICReg import VICReg


def _init_model(class_constructor, config):
    return class_constructor(**config)


class SimCLRTestCase(unittest.TestCase):
    """ Implements a set of basic tests for SimCLR framework exploiting different
        combinations of encoders and projection heads. These tests are encoder and classifier agnostic, i.e.
        they are the same regardless of encoder and classifier used.

        Each test for each combination of encoder and projection head is executed as
        a separate. Thus, if a test fails, the combination of encoder and projection config
        will be printed out along with the error.

        A new encoder, once implemented, can be easily integrated into this test case.
        It is required to provide the following inputs for <encoder>:
            * self.encoder_config : kwargs dictionary that can be used to init the encoder
            * self.encoder_input_shape: input shape expected by the encoder (batch_size, ..., ...)
            * self.encoder_input = torch.randn(*self.encoder_input_shape)
            * self.encoder = _init_model(encoder_cls, self.encoder_config)
                encoder cls is the class implementing the encoder, e.g. CNN1D

        These should be then added to self.encoders dict as a new entry:
        self.encoders = [
            ...
            {
                "encoder": self.encoder,
                "encoder_cls": encoder_cls,
                "encoder_config": self.cnn_config,
                "encoder_input": self.cnn_input
            },
            ...
        ]

        In order to encorporate new classifiers, define:
        self.projection_config = {
           "projection_out": 256,
           ...
        }
        Append this config to self.projection_configs = [..., self.projection_config]

        If more custom tests are needed to perform more sophisticated tests,
            the tests can be added as functions to this unit case.

    """
    def setUp(self):
        self.simclr_config = {
            "ssl_batch_size": 2,
            "temperature": 0.1,
            "n_views": 2
        }
        # Encoders
        # CNN1D configs and init
        self.cnn_config = {
            "in_channels": 10,
            "len_seq": 1000,
            "out_channels": [16, 32, 64],
            "kernel_sizes": [3, 5, 7],
        }
        self.cnn_input_shape = (
            self.simclr_config["ssl_batch_size"],
            self.cnn_config["in_channels"],
            self.cnn_config["len_seq"]
        )
        cnn_random_input = torch.rand(*self.cnn_input_shape)
        self.cnn_input = torch.cat((
            cnn_random_input,
            cnn_random_input + torch.normal(0, 0.5, cnn_random_input.shape)
        ))

        self.cnn = _init_model(CNN1D, self.cnn_config)
        # Wav2Vec2-BASE configs and init
        self.w2v2_base_config = {
            "length_samples": 10,
            "sample_rate": 16000,
            "w2v2_type": 'base',
            "freeze": True,
            "out_channels": [128, 128],
            "kernel_sizes": [1, 1],
        }
        self.w2v2_base_input_shape = (
            self.simclr_config["ssl_batch_size"],
            1,
            self.w2v2_base_config["length_samples"] * self.w2v2_base_config["sample_rate"]
        )

        w2v2_random_input = torch.rand(*self.w2v2_base_input_shape)
        self.w2v2_base_input = torch.cat((
            w2v2_random_input,
            w2v2_random_input + torch.normal(0, 0.5, w2v2_random_input.shape)
        ))
        self.w2v2_base = _init_model(Wav2Vec2CNN, self.w2v2_base_config)
        # encoder mappings
        self.encoders = [
            {
                "encoder": self.cnn,
                "encoder_cls": CNN1D,
                "encoder_config": self.cnn_config,
                "encoder_input": self.cnn_input
            },
            {
                "encoder": self.w2v2_base,
                "encoder_cls": Wav2Vec2CNN,
                "encoder_config": self.w2v2_base_config,
                "encoder_input": self.w2v2_base_input
            }
        ]

        # Projections are initialized inside of the
        # Thus, we only provide their hyperparameters
        self.non_linear_projection_config = {
            "projection_out": 256,
            "projection_hidden": [256]
        }
        self.projection_configs = [self.non_linear_projection_config]

        # Combinations of encoders and classifiers
        self.simclr_combinations = []
        for projection_cfg in self.projection_configs:
            for encoder_mapping in self.encoders:
                self.simclr_combinations.append({
                    "encoder": encoder_mapping["encoder"],
                    "encoder_cls": encoder_mapping["encoder_cls"],
                    "encoder_config": encoder_mapping["encoder_config"],
                    "projection_config": projection_cfg,
                    "input": encoder_mapping["encoder_input"]
                })

    def test_init_simclr_model_from_components(self):
        for i, combination in enumerate(self.simclr_combinations):
            simclr_model = SimCLR(
                combination["encoder"],
                **self.simclr_config,
                **combination["projection_config"]
            )
            with self.subTest(
                f"""
                    Encoder: {simclr_model.encoder.__class__.__name__}
                    Projection config: {str(combination['projection_config'])}
                """,
                i=i
            ):
                self.assertTrue(
                    isinstance(simclr_model.encoder, combination["encoder_cls"]),
                    "Encoder class does not match configurations"
                )

    def test_forward_pass_shape(self):
        for i, combination in enumerate(self.simclr_combinations):
            simclr_model = SimCLR(
                combination["encoder"],
                **self.simclr_config,
                **combination["projection_config"]
            )
            with self.subTest(
                f"""
                    Encoder: {simclr_model.encoder.__class__.__name__}
                    Projection config: {str(combination['projection_config'])}
                """,
                i=i
            ):
                output = simclr_model(combination["input"])
                self.assertEqual(
                    output.shape,
                    (combination["input"].shape[0], combination["projection_config"]["projection_out"])
                )

    def test_training_step_loss_computed(self):
        for i, combination in enumerate(self.simclr_combinations):
            simclr_model = SimCLR(
                combination["encoder"],
                **self.simclr_config,
                **combination["projection_config"]
            )
            with self.subTest(
                f"""
                    Encoder: {simclr_model.encoder.__class__.__name__}
                    Projection config: {str(combination['projection_config'])}
                """,
                i=i
            ):
                output = simclr_model.training_step(
                    (
                        combination["input"][:self.simclr_config["ssl_batch_size"]],
                        None,
                        combination["input"][self.simclr_config["ssl_batch_size"]:]
                    ),
                    0
                )
                self.assertEqual(output.numel(), 1)
                self.assertTrue(torch.is_tensor(output))
                output.backward()

    def test_correct_model_load_from_checkpoint(self):
        for i, combination in enumerate(self.simclr_combinations):
            simclr_model = SimCLR(
                combination["encoder"],
                **self.simclr_config,
                **combination["projection_config"]
            )
            with self.subTest(
                f"""
                    Encoder: {simclr_model.encoder.__class__.__name__}
                    Projection config: {str(combination['projection_config'])}
                """,
                i=i
            ):
                test_dir = tempfile.mkdtemp()
                model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

                trainer = Trainer(default_root_dir=test_dir)
                trainer.strategy.connect(simclr_model)

                trainer.save_checkpoint(model_path)

                # Intialize new CNN and MLP with random weights
                new_encoder = combination["encoder_cls"](
                    **combination["encoder_config"]
                )

                # Use structure of these newly initialzied models to load weights from the pretrained simclr model
                # This simulates inference scenario,
                # i.e. when we initialize a model using previously saved weights of the whole simclr model
                simclr_model_loaded_lightning = SimCLR.load_from_checkpoint(
                    model_path,
                    encoder=new_encoder,
                )

                # one way to check if models are the same is to check if they produce the same output for the same input
                simclr_model.eval()
                simclr_model_loaded_lightning.eval()

                output_simclr = simclr_model(combination["input"])
                output_simclr_lightning = simclr_model_loaded_lightning(combination["input"])

                self.assertTrue(torch.allclose(output_simclr, output_simclr_lightning))

                shutil.rmtree(test_dir)

    def test_correct_encoder_load_from_checkpoint(self):
        for i, combination in enumerate(self.simclr_combinations):
            simclr_model = SimCLR(
                combination["encoder"],
                **self.simclr_config,
                **combination["projection_config"]
            )
            with self.subTest(
                f"""
                    Encoder: {simclr_model.encoder.__class__.__name__}
                    Projection config: {str(combination['projection_config'])}
                """,
                i=i
            ):
                test_dir = tempfile.mkdtemp()
                model_path = os.path.join(test_dir, "test_checkpoint.pt")

                torch.save(simclr_model.encoder.state_dict(), model_path)

                # Intialize new encoder using weights
                new_encoder = combination["encoder_cls"](
                    **combination["encoder_config"],
                    pretrained=model_path
                )

                # one way to check if models are the same is to check if they produce the same output for the same input
                simclr_model.encoder.eval()
                new_encoder.eval()

                output_simclr = simclr_model.encoder(combination["input"])
                output_simclr_lightning = new_encoder(combination["input"])

                self.assertTrue(torch.allclose(output_simclr, output_simclr_lightning))

                shutil.rmtree(test_dir)


class VICRegTestCase(unittest.TestCase):
    """ Implements a set of basic tests for VICReg framework exploiting different
        combinations of encoders and projection heads. These tests are encoder and classifier agnostic, i.e.
        they are the same regardless of encoder and classifier used.

        Each test for each combination of encoder and projection head is executed as
        a separate. Thus, if a test fails, the combination of encoder and projection config
        will be printed out along with the error.

        A new encoder, once implemented, can be easily integrated into this test case.
        It is required to provide the following inputs for <encoder>:
            * self.encoder_config : kwargs dictionary that can be used to init the encoder
            * self.encoder_input_shape: input shape expected by the encoder (batch_size, ..., ...)
            * self.encoder_input = torch.randn(*self.encoder_input_shape)
            * self.encoder = _init_model(encoder_cls, self.encoder_config)
                encoder cls is the class implementing the encoder, e.g. CNN1D

        These should be then added to self.encoders dict as a new entry:
        self.encoders = [
            ...
            {
                "encoder": self.encoder,
                "encoder_cls": encoder_cls,
                "encoder_config": self.cnn_config,
                "encoder_input": self.cnn_input
            },
            ...
        ]

        In order to encorporate new classifiers, define:
        self.projection_config = {
           "projection_out": 256,
           ...
        }
        Append this config to self.projection_configs = [..., self.projection_config]

        If more custom tests are needed to perform more sophisticated tests,
            the tests can be added as functions to this unit case.

    """
    def setUp(self):
        self.vicreg_config = {
            "ssl_batch_size": 2,
            "sim_coeff": 25,
            "std_coeff": 25,
            "cov_coeff": 1
        }
        # Encoders
        # CNN1D configs and init
        self.cnn_config = {
            "in_channels": 10,
            "len_seq": 1000,
            "out_channels": [16, 32, 64],
            "kernel_sizes": [3, 5, 7],
        }
        self.cnn_input_shape = (
            self.vicreg_config["ssl_batch_size"],
            self.cnn_config["in_channels"],
            self.cnn_config["len_seq"]
        )

        self.cnn_input_view1 = torch.rand(*self.cnn_input_shape)
        self.cnn_input_view2 = self.cnn_input_view1 + torch.normal(0, 0.5, self.cnn_input_view1.shape)

        self.cnn = _init_model(CNN1D, self.cnn_config)
        # Wav2Vec2-BASE configs and init
        self.w2v2_base_config = {
            "length_samples": 10,
            "sample_rate": 16000,
            "w2v2_type": 'base',
            "freeze": True,
            "out_channels": [128, 128],
            "kernel_sizes": [1, 1],
        }
        self.w2v2_base_input_shape = (
            self.vicreg_config["ssl_batch_size"],
            1,
            self.w2v2_base_config["length_samples"] * self.w2v2_base_config["sample_rate"]
        )

        self.w2v2_base_input_view1 = torch.rand(*self.w2v2_base_input_shape)
        self.w2v2_base_input_view2 = self.w2v2_base_input_view1 + \
            torch.normal(0, 0.5, self.w2v2_base_input_view1.shape)
        self.w2v2_base = _init_model(Wav2Vec2CNN, self.w2v2_base_config)

        # encoder mappings
        self.encoders = [
            {
                "encoder": self.cnn,
                "encoder_cls": CNN1D,
                "encoder_config": self.cnn_config,
                "encoder_input_view1": self.cnn_input_view1,
                "encoder_input_view2": self.cnn_input_view2
            },
            {
                "encoder": self.w2v2_base,
                "encoder_cls": Wav2Vec2CNN,
                "encoder_config": self.w2v2_base_config,
                "encoder_input_view1": self.w2v2_base_input_view1,
                "encoder_input_view2": self.w2v2_base_input_view2
            }
        ]

        # Projections are initialized inside of the
        # Thus, we only provide their hyperparameters
        self.non_linear_projection_config = {
            "projection_out": 256,
            "projection_hidden": [256]
        }
        self.projection_configs = [self.non_linear_projection_config]

        # Combinations of encoders and classifiers
        self.vicreg_combinations = []
        for projection_cfg in self.projection_configs:
            for encoder_mapping in self.encoders:
                self.vicreg_combinations.append({
                    "encoder": encoder_mapping["encoder"],
                    "encoder_cls": encoder_mapping["encoder_cls"],
                    "encoder_config": encoder_mapping["encoder_config"],
                    "projection_config": projection_cfg,
                    "input_view1": encoder_mapping["encoder_input_view1"],
                    "input_view2": encoder_mapping["encoder_input_view2"]
                })

    def test_init_vicreg_model_from_components(self):
        for i, combination in enumerate(self.vicreg_combinations):
            vicreg_model = VICReg(
                combination["encoder"],
                **self.vicreg_config,
                **combination["projection_config"]
            )
            with self.subTest(
                f"""
                    Encoder: {vicreg_model.encoder.__class__.__name__}
                    Projection config: {str(combination['projection_config'])}
                """,
                i=i
            ):
                self.assertTrue(
                    isinstance(vicreg_model.encoder, combination["encoder_cls"]),
                    "Encoder class does not match configurations"
                )

    def test_forward_pass_shape(self):
        for i, combination in enumerate(self.vicreg_combinations):
            vicreg_model = VICReg(
                combination["encoder"],
                **self.vicreg_config,
                **combination["projection_config"]
            )
            with self.subTest(
                f"""
                    Encoder: {vicreg_model.encoder.__class__.__name__}
                    Projection config: {str(combination['projection_config'])}
                """,
                i=i
            ):
                output = vicreg_model(combination["input_view1"], combination["input_view2"])
                self.assertEqual(
                    output[0].shape,
                    (combination["input_view1"].shape[0], combination["projection_config"]["projection_out"])
                )
                self.assertEqual(
                    output[1].shape,
                    (combination["input_view1"].shape[0], combination["projection_config"]["projection_out"])
                )

    def test_training_step_loss_computed(self):
        for i, combination in enumerate(self.vicreg_combinations):
            vicreg_model = VICReg(
                combination["encoder"],
                **self.vicreg_config,
                **combination["projection_config"]
            )
            with self.subTest(
                f"""
                    Encoder: {vicreg_model.encoder.__class__.__name__}
                    Projection config: {str(combination['projection_config'])}
                """,
                i=i
            ):
                output = vicreg_model.training_step(
                    (
                        combination["input_view1"],
                        None,
                        combination["input_view2"]
                    ),
                    0
                )
                self.assertEqual(output.numel(), 1)
                self.assertTrue(torch.is_tensor(output))
                output.backward()

    def test_correct_model_load_from_checkpoint(self):
        for i, combination in enumerate(self.vicreg_combinations):
            vicreg_model = VICReg(
                combination["encoder"],
                **self.vicreg_config,
                **combination["projection_config"]
            )
            with self.subTest(
                f"""
                    Encoder: {vicreg_model.encoder.__class__.__name__}
                    Projection config: {str(combination['projection_config'])}
                """,
                i=i
            ):
                test_dir = tempfile.mkdtemp()
                model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

                trainer = Trainer(default_root_dir=test_dir)
                trainer.strategy.connect(vicreg_model)

                trainer.save_checkpoint(model_path)

                # Intialize new CNN and MLP with random weights
                new_encoder = combination["encoder_cls"](
                    **combination["encoder_config"]
                )

                # Use structure of these newly initialzied models to load weights from the pretrained vicreg model
                # This simulates inference scenario,
                # i.e. when we initialize a model using previously saved weights of the whole vicreg model
                vicreg_model_loaded_lightning = VICReg.load_from_checkpoint(
                    model_path,
                    encoder=new_encoder,
                )

                # one way to check if models are the same is to check if they produce the same output for the same input
                vicreg_model.eval()
                vicreg_model_loaded_lightning.eval()

                output_vicreg = vicreg_model(combination["input_view1"], combination["input_view2"])
                output_vicreg_lightning = vicreg_model_loaded_lightning(
                    combination["input_view1"],
                    combination["input_view2"]
                )

                self.assertTrue(torch.allclose(output_vicreg[0], output_vicreg_lightning[0]))
                self.assertTrue(torch.allclose(output_vicreg[1], output_vicreg_lightning[1]))

                shutil.rmtree(test_dir)

    def test_correct_encoder_load_from_checkpoint(self):
        for i, combination in enumerate(self.vicreg_combinations):
            vicreg_model = VICReg(
                combination["encoder"],
                **self.vicreg_config,
                **combination["projection_config"]
            )
            with self.subTest(
                f"""
                    Encoder: {vicreg_model.encoder.__class__.__name__}
                    Projection config: {str(combination['projection_config'])}
                """,
                i=i
            ):
                test_dir = tempfile.mkdtemp()
                model_path = os.path.join(test_dir, "test_checkpoint.pt")

                torch.save(vicreg_model.encoder.state_dict(), model_path)

                # Intialize new encoder using weights
                new_encoder = combination["encoder_cls"](
                    **combination["encoder_config"],
                    pretrained=model_path
                )

                # one way to check if models are the same is to check if they produce the same output for the same input
                vicreg_model.encoder.eval()
                new_encoder.eval()

                output_vicreg = vicreg_model.encoder(combination["input_view1"])
                output_vicreg_lightning = new_encoder(combination["input_view1"])

                self.assertTrue(torch.allclose(output_vicreg, output_vicreg_lightning))

                shutil.rmtree(test_dir)
