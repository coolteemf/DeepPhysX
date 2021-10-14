# from numpy import frombuffer, array

# from DeepPhysX_Core.AsyncSocket.BytesBaseConverter import BytesBaseConverter
#
#
# class BytesNumpyConverter(BytesBaseConverter):
#
#     def bytes_to_data(self, bytes_field):
#         return frombuffer(bytes_field)
#
#     def data_to_bytes(self, data):
#         return data.tobytes()
#
#     def data_type(self):
#         return type(array([]))
from struct import pack, unpack


class BytesConverter:

    def __init__(self):
        # Data to bytes conversions
        self.__data_to_bytes_conversion = {bytes: lambda d: d,
                                           str: lambda d: d.lower().encode('utf-8'),
                                           int: lambda d: d.to_bytes(length=(8 + (d + (d < 0)).bit_length()) // 8,
                                                                     byteorder='big', signed=True),
                                           float: lambda d: pack('f', d)}
        # Bytes to data conversions
        self.__bytes_to_data_conversion = {bytes.__name__: lambda b: b,
                                           str.__name__: lambda b: b.decode('utf-8'),
                                           int.__name__: lambda b: int.from_bytes(b, byteorder='big', signed=True),
                                           float.__name__: lambda b: unpack('f', b)[0]}

    def data_to_bytes(self, data):
        return self.__data_to_bytes_conversion[type(data)](data)

    def bytes_to_data(self, bytes_field, data_type):
        return self.__bytes_to_data_conversion[data_type](bytes_field)


if __name__ == '__main__':
    converter = BytesConverter()

    for data in [b'test', 'test', 1740, -56, 17.40, -1e-3]:

        sending = [type(data).__name__, data]
        b_sending = [converter.data_to_bytes(s) for s in sending]

        data_reco = converter.bytes_to_data(b_sending[1], converter.bytes_to_data(b_sending[0], str.__name__))
        print(data, b_sending, data_reco)
