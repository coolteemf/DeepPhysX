import unittest
from numpy import array

from DeepPhysX_Core.AsyncSocket.BytesConverter import BytesConverter


class TestBytesConverter(unittest.TestCase):

    def setUp(self):
        self.converter = BytesConverter()

    def test_conversions(self):
        # Check conversions for all types except numpy array
        for data in [b'test', 'test', True, False, 1, -1, 1., -1., [0.1, 0.1], [[-1, 0], [0, 1]]]:
            size, conversion = self.converter.data_to_bytes(data)
            recovered_data = self.converter.bytes_to_data(conversion)
            print(type(recovered_data), recovered_data)
            self.assertEqual(data, recovered_data)
            self.assertEqual(type(data), type(recovered_data))
            test_size = 4 if type(data) == list else 2
            self.assertEqual(int.from_bytes(size, byteorder='big', signed=True), test_size)
        # Check conversions for numpy array
        for data in [array([0.1, 0.1], dtype=float), array([[-1, 0], [0, 1]], dtype=int)]:
            size, conversion = self.converter.data_to_bytes(data)
            recovered_data = self.converter.bytes_to_data(conversion)
            self.assertTrue((data == recovered_data).all())
            self.assertEqual(type(data), type(recovered_data))
            self.assertEqual(int.from_bytes(size, byteorder='big', signed=True), 4)
