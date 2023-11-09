import os
import shutil
import tempfile
import unittest

import torch
from pytorch_lightning import Trainer

from supervised_audio_modality.classifiers.linear import LinearClassifier
from supervised_audio_modality.classifiers.mlp import MLPClassifier
from supervised_audio_modality.encoders.cnn1d import CNN1D
from supervised_audio_modality.encoders.w2v import Wav2Vec2CNN
from supervised_audio_modality.classification_model import SupervisedModel


def _init_model(class_constructor, config):
    return class_constructor(**config)


class SupervisedTestCase(unittest.TestCase):
    """ Implements a set of basic tests for supervised models comprising of different combinations of
        encoders and classifiers. These tests are encoder and classifier agnostic, i.e.
        they are the same regardless of encoder and classifier used.

        Each test for each combination of encoder and classifier is executed as
        a separate sub-test with name [encoder_cls-classifier_cls]. Thus, if a test fails,
        the combination of encoder and classifier will be printed out along with the error.

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
        self.classifier_config = {
            "class": classifier_cls,
            "out_size": integer_value,
            **other_kwargs needed to init the classifier
        }
        Append this classifier config to self.classifier_configs = [..., self.classifier_config]

        If more custom tests are needed to perform more sophisticated tests,
            the tests can be added as functions to this unit case.

    """
    def setUp(self):
        # Encoders
        # CNN1D configs and init
        self.cnn_config = {
            "in_channels": 10,
            "len_seq": 1000,
            "out_channels": [16, 32, 64],
            "kernel_sizes": [3, 5, 7],
        }
        self.cnn_input_shape = (64, self.cnn_config["in_channels"], self.cnn_config["len_seq"])
        self.cnn_input = torch.rand(*self.cnn_input_shape)
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
            2,
            1,
            self.w2v2_base_config["length_samples"] * self.w2v2_base_config["sample_rate"]
        )
        self.w2v2_base_input = torch.rand(*self.w2v2_base_input_shape)
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

        # Classifiers
        # The logic is slightly different for classifier initialization, as they require
        # input size to be initialized. It is available from encoder.out_size
        self.mlp_classifier_config = {
            "class": MLPClassifier,
            "out_size": 4,
            "hidden": [256, 128, 64]
        }

        self.linear_classifier_config = {
            "class": LinearClassifier,
            "out_size": 4,
        }
        self.classifier_configs = [self.mlp_classifier_config, self.linear_classifier_config]

        # Combinations of encoders and classifiers
        self.supervised_combinations = []
        for classifier_cfg in self.classifier_configs:
            classifier_cls = classifier_cfg.pop("class")
            for encoder_mapping in self.encoders:
                # some parameters "in_size" will be overwritten, thus create a copy
                cur_classifier_cfg = classifier_cfg.copy()
                # out_size should be either hard-coded in config or be available as argument of encoder class
                cur_classifier_cfg["in_size"] = encoder_mapping["out_size"] if (
                    "out_size" in encoder_mapping
                ) else encoder_mapping["encoder"].out_size
                classifier = classifier_cls(**cur_classifier_cfg)
                self.supervised_combinations.append({
                    "encoder": encoder_mapping["encoder"],
                    "encoder_cls": encoder_mapping["encoder_cls"],
                    "classifier": classifier,
                    "encoder_config": encoder_mapping["encoder_config"],
                    "classifier_config": cur_classifier_cfg,
                    "classifier_cls": classifier_cls,
                    "input": encoder_mapping["encoder_input"]
                })

    def test_init_supervised_model_from_components(self):
        for i, combination in enumerate(self.supervised_combinations):
            supervised_model = SupervisedModel(
                combination["encoder"],
                combination["classifier"],
                freeze_encoder=False
            )
            with self.subTest(
                f"{supervised_model.encoder.__class__.__name__}-{supervised_model.classifier.__class__.__name__}",
                i=i
            ):
                self.assertTrue(
                    isinstance(supervised_model.classifier, combination["classifier_cls"]),
                    "Classifier class does not match configurations"
                )
                self.assertTrue(
                    isinstance(supervised_model.encoder, combination["encoder_cls"]),
                    "Classifier class does not match configurations"
                )

    def test_forward_pass_shape(self):
        for i, combination in enumerate(self.supervised_combinations):
            supervised_model = SupervisedModel(
                combination["encoder"],
                combination["classifier"],
                freeze_encoder=False
            )
            with self.subTest(
                f"{supervised_model.encoder.__class__.__name__}-{supervised_model.classifier.__class__.__name__}",
                i=i
            ):
                output = supervised_model(combination["input"])
                self.assertEqual(
                    output.shape,
                    (combination["input"].shape[0], combination["classifier_config"]["out_size"])
                )

    def test_model_freeze(self):
        for i, combination in enumerate(self.supervised_combinations):
            supervised_model = SupervisedModel(
                combination["encoder"],
                combination["classifier"],
                freeze_encoder=True
            )
            with self.subTest(
                f"{supervised_model.encoder.__class__.__name__}-{supervised_model.classifier.__class__.__name__}",
                i=i
            ):
                for param in supervised_model.encoder.parameters():
                    self.assertFalse(param.requires_grad)

    def test_correct_model_load_from_checkpoint(self):
        for i, combination in enumerate(self.supervised_combinations):
            supervised_model = SupervisedModel(
                combination["encoder"],
                combination["classifier"],
                freeze_encoder=False
            )
            with self.subTest(
                f"{supervised_model.encoder.__class__.__name__}-{supervised_model.classifier.__class__.__name__}",
                i=i
            ):
                test_dir = tempfile.mkdtemp()
                model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

                trainer = Trainer(default_root_dir=test_dir)
                trainer.strategy.connect(supervised_model)

                trainer.save_checkpoint(model_path)

                # Intialize new CNN and MLP with random weights
                new_encoder = combination["encoder_cls"](
                    **combination["encoder_config"]
                )

                new_classifier = combination["classifier_cls"](
                    **combination["classifier_config"]
                )

                # Use structure of these newly initialzied models to load weights from the pretrained supervised model
                # This simulates inference scenario,
                # i.e. when we initialize a model using previously saved weights of the whole supervised model
                supervised_model_loaded_lightning = SupervisedModel.load_from_checkpoint(
                    model_path,
                    encoder=new_encoder,
                    classifier=new_classifier
                )

                # one way to check if models are the same is to check if they produce the same output for the same input
                supervised_model.eval()
                supervised_model_loaded_lightning.eval()

                output_supervised = supervised_model(combination["input"])
                output_supervised_lightning = supervised_model_loaded_lightning(combination["input"])

                self.assertTrue(torch.allclose(output_supervised, output_supervised_lightning))

                shutil.rmtree(test_dir)

    def test_correct_model_load_from_pretrained_encoder(self):
        for i, combination in enumerate(self.supervised_combinations):
            supervised_model = SupervisedModel(
                combination["encoder"],
                combination["classifier"],
                freeze_encoder=False
            )
            with self.subTest(
                f"{supervised_model.encoder.__class__.__name__}-{supervised_model.classifier.__class__.__name__}",
                i=i
            ):
                test_dir = tempfile.mkdtemp()
                model_path = os.path.join(test_dir, "test_checkpoint.ckpt")

                trainer = Trainer(default_root_dir=test_dir)
                trainer.strategy.connect(combination["encoder"])

                trainer.save_checkpoint(model_path)
                encoder_loaded_lightning = combination["encoder_cls"].load_from_checkpoint(
                    model_path,
                )

                supervised_model_encoder_checkpoint = SupervisedModel(
                    encoder_loaded_lightning,
                    combination["classifier"],
                    freeze_encoder=False
                )

                # one way to check if models are the same is to check if they produce the same output for the same input
                supervised_model.eval()
                supervised_model_encoder_checkpoint.eval()

                output_supervised = supervised_model(combination["input"])
                output_supervised_encoder_checkpoint = supervised_model_encoder_checkpoint(combination["input"])

                self.assertTrue(torch.allclose(output_supervised, output_supervised_encoder_checkpoint))

                shutil.rmtree(test_dir)
