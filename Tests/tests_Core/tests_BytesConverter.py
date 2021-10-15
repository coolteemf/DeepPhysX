import unittest
from numpy import array

from DeepPhysX_Core.AsyncSocket.BytesConverter import BytesConverter


class TestBytesConverter(unittest.TestCase):

    def setUp(self):
        self.converter = BytesConverter()

    def test_conversions(self):
        # Check conversions for all types except numpy array
        for data in [b'test', 'test', 1, -1, 1., -1., [0.1, 0.1], [[-1, 0], [0, 1]]]:
            conversion = self.converter.data_to_bytes(data)
            recovered_data = self.converter.bytes_to_data(conversion)
            self.assertEqual(data, recovered_data)
            self.assertEqual(type(data), type(recovered_data))
        # Check conversions for numpy array
        for data in [array([0.1, 0.1]), array([[-1, 0], [0, 1]])]:
            conversion = self.converter.data_to_bytes(data)
            recovered_data = self.converter.bytes_to_data(conversion)
            self.assertTrue((data == recovered_data).all())
            self.assertEqual(type(data), type(recovered_data))
