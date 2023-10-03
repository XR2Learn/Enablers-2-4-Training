import unittest
from unittest.mock import patch
from supervised_audio_modality.audio_supervised_train import CSVDataset


def mocked_load_x_y_from_csv(self, path):
    # give a simplified version of data for the tests, so it is not needed to read files
    # when running the test
    self.X = ["path_01", "path_02", "path_03"]
    self.y = [[1.], [2.], [3.]]


class CSVDatasetTestCase(unittest.TestCase):
    def setUp(self):
        # It is called every time you run the tests
        # TODO put this path as a relative path and import from conf.py
        CSVDataset.load_x_y_from_csv = mocked_load_x_y_from_csv
        self.dataset = CSVDataset("<fake_path>")

    def tearDown(self):
        pass

    @patch('supervised_audio_modality.audio_supervised_train.CSVDataset.get_example_features')
    def test_getitem_has_correct_dimensions(self, mocked_get_example_features):
        mocked_get_example_features.return_value = "Mocked features"
        row = self.dataset[1]
        self.assertEqual(len(row), 2)

    @patch('supervised_audio_modality.audio_supervised_train.CSVDataset.get_example_features')
    def test_getitem_calls_correct_method(self, mocked_get_example_features):
        """
        Method of test to serve as reference of how to create unittests when mocking methods.
        :param mocked_get_example_features:
        :return:
        """
        mocked_get_example_features.return_value = "Mocked features"
        row = self.dataset[1]
        self.assertEqual(row[0], "Mocked features")
        # to test if the mocked object was called
        self.assertTrue(mocked_get_example_features.called)
        # to test if the mocked object was called once with the specific argument
        # (more specific than above)
        mocked_get_example_features.assert_called_once_with(1)

    def test_instance_exist(self):
        self.assertIsNotNone(self.dataset)
