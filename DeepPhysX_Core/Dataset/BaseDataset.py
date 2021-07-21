import numpy as np


class BaseDataset:

    def __init__(self, config):
        """
        BaseDataset is a dataset class to store any data from a BaseEnvironment or from files.
        Given data is split into input data and output data.
        Saving data results in multiple partitions of input and output data.

        :param config: BaseDatasetConfig.BaseDatasetProperties class containing BaseDataset parameters
        """

        self.name = self.__class__.__name__

        # Data storage
        self.data_in, self.data_out = np.array([]), np.array([])
        self.in_shape, self.in_flat_shape = None, None
        self.out_shape, self.out_flat_shape = None, None
        self.max_size = config.max_size
        self.current_sample = 0

    def init_data_size(self, side, data):
        """
        Keep in the original shape of data and its flat shape.
        Init data_in and data_out as arrays containing each flat sample.

        :param str side: Tells if data is an input or an output
        :param numpy.ndarray data:
        :return:
        """
        # Init data_in
        if side == 'in':
            self.in_shape = data.shape
            self.in_flat_shape = len(data.flatten())
            self.data_in = np.array([]).reshape((0, self.in_flat_shape))
        # Init data_out
        else:
            self.out_shape = data.shape
            self.out_flat_shape = len(data.flatten())
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

        :param str side: Tells if data is an input or an output
        :param numpy.array data:
        :return:
        """
        side = "inputs" if side == 'in' else "outputs"
        if type(data) != np.ndarray:
            raise TypeError(f"[{self.name}] Wrong data {side}: numpy.ndarray required, get {type(data)}")

    def add(self, side, data, partition_file):
        """
        Add new data to the dataset.

        :param str side: Tells if data is an input or an output
        :param numpy.array data:
        :param partition_file:
        :return:
        """
        # Check data type
        self.check_data(side, data)
        # Adding input data
        if side == 'in':
            # Init sizes variables
            if self.in_flat_shape is None:
                self.init_data_size(side, data[0])
            # Store and save each sample in batch
            for i in range(len(data)):
                self.data_in = np.concatenate((self.data_in, data[i].flatten()[None, :]))
                np.save(partition_file, data[i].flatten())
        # Adding output data
        else:
            # Init sizes variables
            if self.out_flat_shape is None:
                self.init_data_size(side, data[0])
            # Store and save each sample in batch
            for i in range(len(data)):
                self.data_out = np.concatenate((self.data_out, data[i].flatten()[None, :]))
                np.save(partition_file, data[i].flatten())
        self.current_sample = max(len(self.data_in), len(self.data_out))

    def load(self, side, data):
        """
        Add existing data to the dataset.

        :param str side: Tells if data is an input or an output
        :param numpy.array data:
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
        # Shuffle the indices of samples in dataset
        indices_in = np.arange(len(self.data_in))
        np.random.shuffle(indices_in)
        indices_out = np.arange(len(self.data_out))
        np.random.shuffle(indices_out)
        # Permute elements in data_in and data_out
        inputs = np.empty_like(self.data_in)
        outputs = np.empty_like(self.data_out)
        for i in range(len(indices_in)):
            inputs[i] = self.data_in[indices_in[i]]
        for i in range(len(indices_out)):
            outputs[i] = self.data_out[indices_out[i]]
        # Update data with shuffled data
        self.data_in = inputs
        self.data_out = outputs

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
