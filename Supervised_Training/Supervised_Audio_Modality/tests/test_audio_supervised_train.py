import unittest
from unittest.mock import patch
from supervised_audio_modality.audio_supervised_train import CSVDataset


class CSVDatasetTestCase(unittest.TestCase):
    # It is called every time you run the tests
    def setUp(self):
        self.dataset = CSVDataset(
            "/Users/annanda/PycharmProjects/XR2Learn-Training/Supervised_Training/Supervised_Audio_Modality/datasets/ravdess_dataset_features.csv");

    def tearDown(self):
        pass

    def test_getitem_dimensions(self):
        row = self.dataset[1]
        self.assertEqual(len(row), 2)
        self.assertEqual(len(row[0][0]), 178)

    def test_instance_exist(self):
        self.assertIsNotNone(self.dataset)
