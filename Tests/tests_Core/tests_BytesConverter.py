import unittest
from numpy import array, ndarray

from DeepPhysX_Core.AsyncSocket.BytesConverter import BytesConverter


class TestBytesConverter(unittest.TestCase):

    def setUp(self):
        self.converter = BytesConverter()
        self.types = {bytes: 0, str: 0, bool: 0, int: 0, float: 0, list: 1, ndarray: 2}
        self.nb_fields = [2, 4, 4]
        self.equal = {0: lambda a, b: a == b,
                      1: lambda a, b: a == b,
                      2: lambda a, b: (a == b).all()}

    def test_conversions(self):
        # Check conversions for all types except numpy array
        for data in [b'test', 'test', True, False, 1, -1, 1., -1., [0.1, 0.1], [[-1, 0], [0, 1]],
                     array([0.1, 0.1], dtype=float), array([[-1, 0], [0, 1]], dtype=int)]:
            conversion = self.converter.data_to_bytes(data, as_list=True)
            size = self.converter.size_from_bytes(conversion.pop(0))
            recovered_data = self.converter.bytes_to_data(conversion[size:])
            index_type = self.types[type(recovered_data)]
            self.assertTrue(self.equal[index_type](data, recovered_data))
            self.assertEqual(type(data), type(recovered_data))
            self.assertEqual(size, self.nb_fields[index_type])
