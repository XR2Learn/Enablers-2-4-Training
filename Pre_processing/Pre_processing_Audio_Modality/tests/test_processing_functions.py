import unittest

import numpy as np

from pre_processing_audio_modality.preprocessing_utils import (
    resample_audio_signal,
    no_preprocessing,
    normalize,
    standardize
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
