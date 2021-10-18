from numpy import ndarray, array, frombuffer, zeros


class BytesConverter:

    def __init__(self):
        """
        Convert usual types to bytes and vice versa.
        Available types: bytes, str, signed int, float, list, array
        """

        # Data to bytes conversions
        self.__data_to_bytes_conversion = {bytes: lambda d: d,
                                           str: lambda d: d.lower().encode('utf-8'),
                                           int: lambda d: d.to_bytes(length=(8 + (d + (d < 0)).bit_length()) // 8,
                                                                     byteorder='big', signed=True),
                                           float: lambda d: array(d, dtype=float).tobytes(),
                                           list: lambda d: array(d, dtype=float).tobytes(),
                                           ndarray: lambda d: array(d, dtype=float).tobytes()}

        # Bytes to data conversions
        self.__bytes_to_data_conversion = {bytes.__name__: lambda b: b,
                                           str.__name__: lambda b: b.decode('utf-8'),
                                           int.__name__: lambda b: int.from_bytes(b, byteorder='big', signed=True),
                                           float.__name__: lambda b: frombuffer(b).item(),
                                           list.__name__: lambda b, t, s: frombuffer(b).astype(t).reshape(s).tolist(),
                                           ndarray.__name__: lambda b, t, s: frombuffer(b).astype(t).reshape(s)}

    def data_to_bytes(self, data):
        """
        Convert data to bytes.
        Available types: bytes, str, signed int, float, list, array.

        :param data: Data to convert.
        :return: bytes_fields: Size of tuple in bytes, (Type, Data, *args)
        """

        # Convert the type of 'data' from str to bytes
        type_data = self.__data_to_bytes_conversion[str](type(data).__name__)
        # Convert 'data' to bytes
        data_bytes = self.__data_to_bytes_conversion[type(data)](data)

        # Additional arguments
        args = ()
        # Shape and data type for list and array
        if type(data) in [list, ndarray]:
            # Get python native datatype of array
            dtype = type(zeros(1, dtype=array(data).dtype).item()).__name__
            # Convert datatype of array from str to bytes
            args += (self.__data_to_bytes_conversion[str](dtype),)
            # Convert data shape from array to bytes
            args += (self.__data_to_bytes_conversion[ndarray](array(data).shape),)

        # Convert the number of bytes fields to bytes
        size = self.__data_to_bytes_conversion[int](2 + len(args))

        return size, (type_data, data_bytes, *args)

    def bytes_to_data(self, bytes_fields):
        """
        Recover data from bytes fields.
        Available types: bytes, str, signed int, float, list, array.

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
