import numpy as np
import operator
import functools

class BaseDataset:

    def __init__(self, config):
        """
        BaseDataset is a dataset class to store any data from a BaseEnvironment or from files.
        Given data is split into input data and output data.
        Saving data results in multiple partitions of input and output data.

        :param BaseDatasetConfig.BaseDatasetProperties config:  Contains BaseDataset parameters
        """

        self.name = self.__class__.__name__

        # Data storage
        self.data_in, self.data_out = np.array([]), np.array([])
        self.in_shape, self.in_flat_shape = None, None
        self.out_shape, self.out_flat_shape = None, None
        self.max_size = config.max_size
        self.current_sample = 0

    def init_data_size(self, side, shape):
        """
        Keep in the original shape of data and its flat shape.
        Init data_in and data_out as arrays containing each flat sample.

        :param str side: Values at 'in' or anything else. Define if the associated shape is correspond to input shape or output one.
        :param numpy.ndarray shape: Shape of the corresponding tensor
        :return:
        """
        # Init data_in
        if side == 'in':
            self.in_shape = shape
            self.in_flat_shape = functools.reduce(operator.mul, shape, 1)
            self.data_in = np.array([]).reshape((0, self.in_flat_shape))
        # Init data_out
        else:
            self.out_shape = shape
            self.out_flat_shape = functools.reduce(operator.mul, shape, 1)
            self.data_out = np.array([]).reshape((0, self.out_flat_shape))

    def reset(self):
        """
        Empty the dataset.

        :return:
        """
        self.data_in = np.array([]).reshape((0, self.in_flat_shape)) if self.in_flat_shape is not None else np.array([])
        self.data_out = np.array([]).reshape((0, self.out_flat_shape)) if self.out_flat_shape is not None else np.array([])
        self.current_sample = 0

    def memory_size(self):
        """
        :return: Size in bytes of the current dataset.
        """
        return self.data_in.nbytes + self.data_out.nbytes

    def check_data(self, side, data):
        """
        Check if the data is a numpy array.

        :param str side: Values at 'in' or anything else. Define if the associated shape is correspond to input shape or output one.
        :param numpy.ndarray data: Corresponding tensor
        :return:
        """
        side = "inputs" if side == 'in' else "outputs"
        if type(data) != np.ndarray:
            raise TypeError(f"[{self.name}] Wrong data {side}: numpy.ndarray required, got {type(data)}")

    def add(self, side, data, partition_file):
        """
        Add new data to the dataset.

        :param str side: Values at 'in' or anything else. Define if the associated shape is correspond to input shape or output one.
        :param numpy.ndarray data: Corresponding tensor
        :param str partition_file: Path or string to the file in which to write the data
        :return:
        """
        # Check data type
        self.check_data(side, data)
        # Adding input data
        if side == 'in':
            # Init sizes variables
            if self.in_flat_shape is None:
                self.init_data_size(side, data[0].shape)
            data_tensor = self.data_in
        # Adding output data
        else:
            # Init sizes variables
            if self.out_flat_shape is None:
                self.init_data_size(side, data[0].shape)
            data_tensor = self.data_out

        # Store and save each sample in batch
        for sample in data:
            data_tensor = np.concatenate((data_tensor, sample.flatten()[None, :]))
            np.save(partition_file, sample.flatten())
        if side == "in":
            self.data_in = data_tensor
        else:
            self.data_out = data_tensor

        self.current_sample = max(len(self.data_in), len(self.data_out))

    def load(self, side, data):
        """
        Add existing data to the dataset.

        :param str side: Values at 'in' or anything else. Define if the associated shape is correspond to input shape or output one.
        :param numpy.ndarray data: Corresponding tensor
        :return:
        """
        self.check_data(side, data)
        # Adding input data
        if side == 'in':
            # Init sizes variables
            if self.in_flat_shape is None:
                self.init_data_size(side, data)
            # Store sample
            self.data_in = np.concatenate((self.data_in, data[None, :]), axis=0)
        # Adding output data
        else:
            # Init sizes variables
            if self.out_flat_shape is None:
                self.init_data_size(side, data)
            # Store sample
            self.data_out = np.concatenate((self.data_out, data[None, :]), axis=0)
        self.current_sample = max(len(self.data_in), len(self.data_out))

    def shuffle(self):
        """
        Shuffle the current dataset.

        :return:
        """

        if self.in_flat_shape is None and self.out_flat_shape is None:
            return
        # Generate a shuffle pattern
        shuffle_pattern = np.arange(self.data_in.shape[0])
        np.random.shuffle(shuffle_pattern)
        # Permute elements in data_in
        if self.in_flat_shape is not None:
            self.data_in = self.data_in[shuffle_pattern]
        # Permute elements in data_out
        if self.out_flat_shape is not None:
            self.data_out = self.data_out[shuffle_pattern]

    def __str__(self):
        """
        :return: String containing information about the BaseDatasetConfig object
        """
        description = "\n"
        description += f"{self.name}\n"
        description += f"    Max size: {self.max_size}\n"
        description += f"    Input shape, input flat shape: {self.in_shape}, {self.in_flat_shape}\n"
        description += f"    Output shape, output flat shape: {self.out_shape}, {self.out_flat_shape}\n"
        return description
