import os
import shutil
import tempfile
import unittest

import numpy as np
import scipy

from pre_processing_audio_modality.preprocessing_utils import (
    resample_audio_signal,
    no_preprocessing,
    normalize,
    standardize,
    process_dataset
)


class PreprocessingUtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.initial_sr = 48000
        self.len_seq = 10
        self.target_sr = 16000
        self.input_audio = np.random.rand(self.initial_sr * self.len_seq)
        self.eps = 1e-3

    def test_resample_audio_signal(self):
        signal_resampled = resample_audio_signal(
            self.input_audio,
            self.initial_sr,
            self.target_sr
        )
        self.assertAlmostEqual(
            len(signal_resampled),
            self.target_sr * self.len_seq,
            msg="Mismatch in size after resampling."
        )

    def test_no_preprocessing(self):
        signal_resampled = no_preprocessing(
            self.input_audio,
        )
        self.assertTrue(
            np.allclose(self.input_audio, signal_resampled)
        )

    def test_normalize(self):
        signal_resampled = resample_audio_signal(
            self.input_audio,
            self.initial_sr,
            self.target_sr
        )

        signal_resampled_normalized = normalize(signal_resampled)

        self.assertTrue(
            np.max(signal_resampled_normalized) < 1 + self.eps,
            "Normalization (min-max): Detected value larger than 1"
        )
        self.assertTrue(
            np.min(signal_resampled_normalized) > -1 - self.eps,
            "Normalization (min-max): Detected value smaller than -1"
        )

    def test_standradize(self):
        signal_resampled = resample_audio_signal(
            self.input_audio,
            self.initial_sr,
            self.target_sr
        )

        signal_resampled_standardized = standardize(signal_resampled)

        self.assertTrue(
            np.abs(np.mean(signal_resampled_standardized)) < self.eps,
            "Standardization (z-norm): Mean is not equal to 0"
        )
        self.assertTrue(
            1 - self.eps < np.std(signal_resampled_standardized) < 1 + self.eps,
            "Standardization (z-norm): Std is not equal to 1"
        )


class PreprocessCustomDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset_size = 100
        self.subjects = ["subject1", "subject2", "subject3", "subject4", "subject5"]
        self.label_to_emotion = {
            "01": "happy",
            "02": "sad",
            "03": "neutral"
        }
        self.initial_sr = 48000
        self.len_seq = 10
        self.target_sr = 16000
        self.data_path = tempfile.mkdtemp()
        self.outputs_folder = tempfile.mkdtemp()
        self._create_dataset()

    def tearDown(self) -> None:
        shutil.rmtree(self.data_path)
        shutil.rmtree(self.outputs_folder)

    def _create_dataset(self):
        input_audio = np.random.rand(self.dataset_size, self.initial_sr * self.len_seq)
        labels = np.random.choice(list(self.label_to_emotion), self.dataset_size)
        subjects = np.random.choice(self.subjects, self.dataset_size)

        for i in range(len(subjects)):
            subject_folder = os.path.join(self.data_path, subjects[i])
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)
            scipy.io.wavfile.write(
                os.path.join(subject_folder, f"{labels[i]}-audio{i}.wav"),
                self.initial_sr,
                input_audio[i]
            )

    def test_preprocessing_signal_only_resample(self):
        custom_settings = {
            "pre_processing_config":
            {
                "process": "only_resample",
                "create_splits": True,
                "target_sr": 16000,
                "padding": False,
                "max_length": 10
            }
        }
        self._common_test_preprocessing_result(custom_settings)

    def test_preprocessing_signal_normalize(self):
        custom_settings = {
            "pre_processing_config":
            {
                "process": "normalize",
                "create_splits": True,
                "target_sr": 16000,
                "padding": False,
                "max_length": 10
            }
        }
        self._common_test_preprocessing_result(custom_settings)

    def test_preprocessing_shorter_signal_standardize(self):
        custom_settings = {
            "pre_processing_config":
            {
                "process": "standardize",
                "create_splits": True,
                "target_sr": 16000,
                "padding": True,
                "max_length": 5
            }
        }
        self._common_test_preprocessing_result(custom_settings)

    def test_preprocessing_longer_signal_padding_standardize(self):
        custom_settings = {
            "pre_processing_config":
            {
                "process": "standardize",
                "create_splits": True,
                "target_sr": 16000,
                "padding": True,
                "max_length": 15
            }
        }
        self._common_test_preprocessing_result(custom_settings)

    def test_preprocessing_no_splits(self):
        custom_settings = {
            "pre_processing_config":
            {
                "process": "standardize",
                "create_splits": False,
                "target_sr": 16000,
                "padding": True,
                "max_length": 10
            }
        }
        self._common_test_preprocessing_result(custom_settings)

    def _common_test_preprocessing_result(self, custom_settings):
        train_split, val_split, test_split = process_dataset(
            self.data_path,
            os.listdir(self.data_path),
            custom_settings["pre_processing_config"],
            self.outputs_folder,
            self.label_to_emotion,
            dataset="custom"
        )

        # Test 1: no intersections between generated splits
        self.assertFalse(
            set(train_split["files"]) & set(val_split["files"]),
            "Train and Val splits overlap"
        )
        self.assertFalse(
            set(test_split["files"]) & set(val_split["files"]),
            "Test and Val splits overlap"
        )
        self.assertFalse(
            set(test_split["files"]) & set(val_split["files"]),
            "Train and Test splits overlap"
        )

        # Test 2: the expected pre-processed folder is generated in outputs
        outputs = os.listdir(self.outputs_folder)
        self.assertTrue(
            custom_settings["pre_processing_config"]["process"] in outputs,
            "Provided processing type is not found in outputs"
        )

        for output in outputs:
            curr_data_path = os.path.join(self.outputs_folder, output)
            # Test 3: Assert dataset size is not changed
            if os.path.isdir(curr_data_path):
                self.assertEqual(
                    len(os.listdir(curr_data_path)),
                    self.dataset_size,
                    "Generated dataset size is smaller than initial one."
                )

            files = [os.path.join(curr_data_path, file_) for file_ in os.listdir(curr_data_path)]
            for file_ in files:
                curr_audio = np.load(file_)

                # Test 4: Check size of the generated audio is correct
                self.assertEqual(
                    curr_audio.size,
                    custom_settings["pre_processing_config"]["max_length"] *
                    custom_settings["pre_processing_config"]["target_sr"],
                    "Unexpected pre-processed audio size"
                )
                # Test 5: Check correctness of (optionally) applied padding
                if (
                    custom_settings["pre_processing_config"]["max_length"] > self.len_seq and
                    custom_settings["pre_processing_config"]["padding"]
                ):
                    self.assertEqual(
                        curr_audio.size,
                        custom_settings["pre_processing_config"]["max_length"] *
                        custom_settings["pre_processing_config"]["target_sr"],
                        "Unexpected pre-processed audio size: no padding applied"
                    )
                    padded_ratio = self.len_seq / custom_settings["pre_processing_config"]["max_length"]
                    zeros_ratio = (curr_audio != 0.0).sum() / curr_audio.size

                    self.assertGreaterEqual(
                        zeros_ratio,
                        padded_ratio,
                        "Unexpected padding values"
                    )

        # Test 6: if not splits created, all samples should be assigned to train
        if not custom_settings["pre_processing_config"]["create_splits"]:
            self.assertFalse(
                test_split["files"],
                "Unexpected splits with create_splits: False"
            )
            self.assertFalse(
                test_split["labels"],
                "Unexpected splits with create_splits: False"
            )
            self.assertFalse(
                val_split["files"],
                "Unexpected splits with create_splits: False"
            )
            self.assertFalse(
                val_split["labels"],
                "Unexpected splits with create_splits: False"
            )
            self.assertEqual(
                len(train_split["files"]),
                self.dataset_size,
                "Unexpected splits with create_splits: False"
            )
            self.assertEqual(
                len(train_split["labels"]),
                self.dataset_size,
                "Unexpected splits with create_splits: False"
            )
