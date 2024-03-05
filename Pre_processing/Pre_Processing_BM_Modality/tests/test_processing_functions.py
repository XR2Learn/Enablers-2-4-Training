import unittest

import numpy as np
import os
import pandas as pd

from pre_processing_bm_modality.preprocessing_utils import (
    resample_bm,
    no_preprocessing,
    normalize,
    standardize,
    process_session,
    segment_processed_session,
    segment_processed_session_ssl
)


class PreprocessingUtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.initial_sr = 10
        self.num_channels = 2
        self.len_seq = 5
        self.target_sr = 5
        self.input_signal = np.random.rand(self.initial_sr * self.len_seq, self.num_channels) * 100
        self.eps = 1e-3

    def test_resample_audio_signal(self):
        signal_resampled = resample_bm(
            self.input_signal,
            self.initial_sr,
            self.target_sr
        )
        self.assertAlmostEqual(
            len(signal_resampled),
            self.target_sr * self.len_seq,
            msg="Mismatch in size after resampling."
        )

    def test_no_preprocessing(self):
        signal = no_preprocessing(
            self.input_signal,
        )
        self.assertTrue(
            np.allclose(self.input_signal, signal)
        )

    def test_normalize(self):
        signal_resampled = resample_bm(
            self.input_signal,
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
        signal_resampled = resample_bm(
            self.input_signal,
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

        np.testing.assert_almost_equal(
            signal_resampled[:, 0],
            signal_resampled_standardized[:, 0] * signal_resampled[:, 0].std() + signal_resampled[:, 0].mean(),
            err_msg="Standardization (z-norm): cannot reconstruct input"
        )

        np.testing.assert_almost_equal(
            signal_resampled[:, 1],
            signal_resampled_standardized[:, 1] * signal_resampled[:, 1].std() + signal_resampled[:, 1].mean(),
            err_msg="Standardization (z-norm): cannot reconstruct input"
        )


class PreprocessingMagicXRoomTestCase(unittest.TestCase):
    def setUp(self):
        self.session_data_file = "./tests/anonymized_magicxroom/anonymized_data_collection_xxxx_SHIMMER_.csv"
        self.session_annot_file = "./tests/anonymized_magicxroom/data_collection_xxxx_PROGRESS_EVENT_.csv"
        self.subject = "P0"
        self.session = "xxxx"
        self.threshold = 10
        self.offset_hours_data = 0
        self.get_ssl = True
        self.get_stats = True
        self.use_sensors = ["gsr", "ppg"]
        self.all_sensors = ["accel_x", "accel_y", "accel_z", "gsr", "ppg", "hr"]

    # def test_noise(self):
    #     directory = "tests/anonymized_magicxroom"
    #     filename = "data_collection_638409299190572115_SHIMMER_.csv"
    #     data = pd.read_csv(os.path.join(directory, filename))
    #     noise_arr = np.zeros(data.shape)
    #     for i, col in enumerate(data.columns):
    #         if "timestamp" not in col:
    #             data_col = data[col]
    #             mean = data_col.mean()
    #             std = data_col.std()
    #             noise_arr[:, i] = np.random.normal(mean, std, data_col.shape)
    #         else:
    #             try:
    #                 noise_arr[:, i] = np.array(data[col])
    #             except ValueError:
    #                 pass
    #     dataframe = pd.DataFrame(noise_arr)
    #     dataframe.columns = data.columns
    #     dataframe.to_csv(os.path.join(directory, "anonymized_" + filename), index=None)

    def test_process_session(self):
        labeled_data, stats, data = process_session(
            self.session_data_file,
            self.session_annot_file,
            self.subject,
            self.session,
            self.threshold,
            self.offset_hours_data,
            self.get_ssl,
            self.get_stats,
            self.use_sensors,
        )

        assert not labeled_data.empty, "Process session: empty labeled data"
        assert not data.empty, "Process session: empty unlabeled data"

        assert labeled_data.shape[0] <= data.shape[0], \
            "Process session: labeled data is longer than unlabeled data"

        for sensor in self.all_sensors:
            if sensor not in self.use_sensors:
                assert sensor not in labeled_data.columns, "Process session: non-required sensor is in labeled data"
                assert sensor not in data.columns, "Process session: non-required sensor is in unlabeled data"
            else:
                assert sensor in labeled_data.columns, "Process session: required sensor is missing in labeled data"
                assert sensor in data.columns, "Process session: required sensor is missing in unlabeled data"

        assert not labeled_data["label"].isna().any(), "Process session: NAN labels in labeled data"

        # True for the currently used anonymized data
        assert stats["length_seconds_BORED"] > 60, "Process session: BORED samples are smaller than expected"
        assert stats["length_seconds_ENGAGED"] > 70, "Process session: ENGAGED samples are smaller than expected"
        assert stats["length_seconds_FRUSTRATED"] > 110, "Process session: FRUSTRATED samples are smaller than expected"

    def test_session_segmentation(self):
        labeled_data, _, _ = process_session(
            self.session_data_file,
            self.session_annot_file,
            self.subject,
            self.session,
            self.threshold,
            self.offset_hours_data,
            self.get_ssl,
            False,
            self.use_sensors,
        )
        seq_len = 10
        frequency = 10
        segments, labels = segment_processed_session(labeled_data, seq_len=10, overlap=0, frequency=10)
        for segment in segments:
            assert segment.shape == (seq_len * frequency, len(self.use_sensors)), \
                f"""Labeled segmentation:
                Wrong shape {segment.shape}, expected {seq_len * frequency, len(self.use_sensors)}"""

        for label in labels:
            assert label in ["BORED", "ENGAGED", "FRUSTRATED"], \
                "Labeled segmentation: unexpected labels generated"

    def test_session_segmentation_ssl(self):
        labeled_data, _, _ = process_session(
            self.session_data_file,
            self.session_annot_file,
            self.subject,
            self.session,
            self.threshold,
            self.offset_hours_data,
            self.get_ssl,
            False,
            self.use_sensors,
        )
        seq_len = 10
        frequency = 10
        segments = segment_processed_session_ssl(labeled_data, seq_len=10, overlap=0, frequency=10)
        for segment in segments:
            assert segment.shape == (seq_len * frequency, len(self.use_sensors)), \
                f"""Labeled segmentation:
                Wrong shape {segment.shape}, expected {seq_len * frequency, len(self.use_sensors)}"""
