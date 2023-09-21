import unittest
from unittest.mock import patch


class MockedServiceStreamTestCase(unittest.TestCase):
    def setUp(self):
        self.something = Something()  # run before every test

    def tearDown(self):  # run after every test
        pass

    @patch('my_project.MyClass.my_method_to_be_mocked')
    def test_something_calls_method(self, mocked_method):
        self.something.run()
        self.assertTrue(mocked_method.called)
