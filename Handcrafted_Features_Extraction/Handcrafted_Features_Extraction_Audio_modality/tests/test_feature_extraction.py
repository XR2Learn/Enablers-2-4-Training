import unittest

import numpy as np
import opensmile
import torch

from handcrafted_features_extraction_audio_modality.feature_extraction import (
    extract_mfcc,
    extract_egemaps,
    extract_mel_spectrogram
)


class TestHandcraftedFeatureExtractionMethods(unittest.TestCase):
    def setUp(self):
        self.length_samples = 10
        self.sample_rate = 16000
        self.input = np.random.rand(self.length_samples * self.sample_rate).astype(np.float32)

        self.handcrafted_features_config = {
                "MFCC": {
                    "sample_rate": self.sample_rate,
                    "n_mfcc": 13,
                    "melkwargs": {
                        "n_fft": 400,
                        "hop_length": 160,
                        "n_mels": 23,
                        "center": False
                    }
                },
                "eGeMAPs": {
                    "sampling_rate": self.sample_rate
                },
                "MelSpectrogram": {
                    "sample_rate": self.sample_rate,
                    "n_fft": 400,
                    "win_length": int(2.5 * 160),
                    "hop_length": 160,
                    "power": 2.0,
                    "n_mels": 64,
                    "f_min": 60,
                    "f_max": 7800,
                    "window_fn": torch.hamming_window,
                    "normalized": False
                }
            }

    def test_extract_mfcc_output_correct(self):
        mfcc = extract_mfcc(
            self.input,
            self.handcrafted_features_config["MFCC"]
        )
        self.assertTrue(
            isinstance(mfcc, np.ndarray),
            "Mismatch output type"
        )
        self.assertEqual(
            mfcc.shape,
            (998, 13),  # (16000 * 10 / 160) - 2
            "Mismatch output shape"
        )

    def test_extract_egemaps_functional_output_correct(self):
        smile_egemaps = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        egemaps = extract_egemaps(
            smile_egemaps,
            self.input,
            self.handcrafted_features_config["eGeMAPs"]
        )
        self.assertTrue(
            isinstance(egemaps, np.ndarray),
            "Mismatch output type"
        )
        self.assertEqual(
            egemaps.shape,
            (88, 1),
            "Mismatch output shape"
        )

    def test_extract_spectrograms_output_correct(self):
        spectrogram = extract_mel_spectrogram(
            self.input,
            self.handcrafted_features_config["MelSpectrogram"]
        )
        self.assertTrue(
            isinstance(spectrogram, np.ndarray),
            "Mismatch output type"
        )
        self.assertEqual(
            spectrogram.shape,
            (1001, 64),  # (16000 * 10 / 160) + 1
            "Mismatch output shape"
        )
