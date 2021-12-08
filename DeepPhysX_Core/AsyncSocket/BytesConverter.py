from numpy import ndarray, array, frombuffer, zeros

from struct import pack, unpack, calcsize


class BytesConverter:

    def __init__(self):
        """
        Convert usual types to bytes and vice versa.
        Available types: bytes, str, bool, signed int, float, list, array
        """

        self.int_size = calcsize("i")
        # Data to bytes conversions
        self.__data_to_bytes_conversion = {type(None): lambda d: b'0',
                                           bytes: lambda d: d,
                                           str: lambda d: d.encode('utf-8'),
                                           bool: lambda d: bytearray(pack('?', d)),
                                           int: lambda d: bytearray(pack('i', d)),
                                           float: lambda d: bytearray(pack('f', d)),
                                           list: lambda d: array(d, dtype=float).tobytes(),
                                           ndarray: lambda d: array(d, dtype=float).tobytes()}

        # Bytes to data conversions
        self.__bytes_to_data_conversion = {type(None).__name__: lambda b: None,
                                           bytes.__name__: lambda b: b,
                                           str.__name__: lambda b: b.decode('utf-8'),
                                           bool.__name__: lambda b: unpack('?', b)[0],
                                           int.__name__: lambda b: unpack('i', b)[0],
                                           float.__name__: lambda b: unpack('f', b)[0],
                                           list.__name__: lambda b, t, s: frombuffer(b).astype(t).reshape(s).tolist(),
                                           ndarray.__name__: lambda b, t, s: frombuffer(b).astype(t).reshape(s)}

        # Size of a bytes field will be encoded on 4 bytes
        self.size_to_bytes = lambda i: self.__data_to_bytes_conversion[int](len(i))
        self.size_from_bytes = lambda b: self.__bytes_to_data_conversion[int.__name__](b)

    def data_to_bytes(self, data, as_list=False):
        """
        Convert data to bytes.
        Available types: bytes, str, bool, signed int, float, list, array.

        :param data: Data to convert.
        :param as_list: (for tests) If True, the return will be a list of bytes fields. If False, the whole bytes
        message is returned
        :return: bytes_fields: Size of tuple in bytes, (Type, Data, *args)
        """

        # Store the sizes of the bytes fields, sizes will have a constant number of 4 bytes
        sizes = ()
        # Convert the type of 'data' from str to bytes
        type_data = self.__data_to_bytes_conversion[str](type(data).__name__)
        sizes += (self.size_to_bytes(type_data),)
        #print(f"BytesConverter : {type(data)} : {data}")
        # Convert 'data' to bytes
        data_bytes = self.__data_to_bytes_conversion[type(data)](data)
        sizes += (self.size_to_bytes(data_bytes),)

        # Additional arguments
        args = ()
        # Shape and data type for list and array
        if type(data) in [list, ndarray]:
            # Get python native datatype of array
            dtype = type(zeros(1, dtype=array(data).dtype).item()).__name__
            # Convert datatype of array from str to bytes
            dtype_bytes = self.__data_to_bytes_conversion[str](dtype)
            args += (dtype_bytes,)
            sizes += (self.size_to_bytes(dtype_bytes),)
            # Convert data shape from array to bytes
            shape_bytes = self.__data_to_bytes_conversion[ndarray](array(data).shape)
            args += (shape_bytes,)
            sizes += (self.size_to_bytes(shape_bytes),)

        # Convert the number of bytes fields to bytes
        nb_fields = self.__data_to_bytes_conversion[int](2 + len(args))

        # Gather all bytes fields in the desired order
        fields = [nb_fields, *sizes, type_data, data_bytes, *args]
        if as_list:
            return fields

        # Concatenate bytes fields
        bytes_message = fields[0]
        for f in fields[1:]:
            bytes_message += f
        return bytes_message

    def bytes_to_data(self, bytes_fields):
        """
        Recover data from bytes fields.
        Available types: bytes, str, bool, signed int, float, list, array.

        :param bytes_fields: (Type, Data, *args)
        :return: Recovered data
        """

        # Recover the data type
        data_type = self.__bytes_to_data_conversion[str.__name__](bytes_fields[0])

        # Recover additional arguments
        args = ()
        # Shape and data type for list and array
        if data_type in [list.__name__, ndarray.__name__]:
            # Recover datatype of array
            args += (self.__bytes_to_data_conversion[str.__name__](bytes_fields[2]),)
            # Recover shape of array
            args += (self.__bytes_to_data_conversion[ndarray.__name__](bytes_fields[3], int, -1),)

        # Convert bytes to data
        return self.__bytes_to_data_conversion[data_type](bytes_fields[1], *args)
