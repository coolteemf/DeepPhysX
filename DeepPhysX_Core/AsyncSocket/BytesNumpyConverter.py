from numpy import frombuffer, array

from DeepPhysX_Core.AsyncSocket.BytesBaseConverter import BytesBaseConverter


class BytesNumpyConverter(BytesBaseConverter):

    def bytes_to_data(self, bytes_field):
        return frombuffer(bytes_field)

    def data_to_bytes(self, data):
        return data.tobytes()

    def data_type(self):
        return type(array([]))
