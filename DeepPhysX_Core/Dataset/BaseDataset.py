from numpy import array, ndarray, concatenate, save, arange
from numpy.random import shuffle
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

        # Data fields containers
        self.data = {'input': array([]), 'output': array([])}
        self.shape = {'input': None, 'output': None}
        self.flat = {'input': None, 'output': None}
        self.fields = {'IN': ['input'], 'OUT': ['output']}

        # Indexing
        self.shuffle_pattern = None
        self.current_sample = 0
        self.max_size = config.max_size
        self.batch_per_field = {field: 0 for field in ['input', 'output']}
        self.empty = True

    def is_empty(self):
        """
        Check if the fields of the dataset are empty. A field is considered as non empty if it is filled another time.

        :return:
        """
        # The empty flag is set to False once th Dataset is considered as non empty
        if not self.empty:
            return False
        # Check each registered data field
        for field in self.fields['IN'] + self.fields['OUT']:
            # Dataset is considered as non empty if a field is fed another time
            if self.batch_per_field[field] > 1:
                self.empty = False
                return False
        # If all field are considered as non empty then the Dataset is empty
        return True

    def init_data_size(self, field, shape):
        """
        Keep in the original shape of data and its flat shape.
        Init data_in and data_out as arrays containing each flat sample.

        :param str field: Values at 'input' or anything else. Define if the associated shape is correspond to input
        shape or output one.
        :param numpy.ndarray shape: Shape of the corresponding tensor
        :return:
        """
        self.shape[field] = shape
        self.flat[field] = functools.reduce(lambda x, y: x * y, shape, 1)
        self.data[field] = array([]).reshape((0, self.flat[field]))

    def init_additional_field(self, field, shape):
        """
        Register a new data field.

        :param field:
        :param shape:
        :return:
        """
        # Register the data field
        side = 'IN' if field[:3] == 'IN_' else 'OUT'
        self.fields[side].append(field)
        self.batch_per_field[field] = 0
        # Init the field shape
        self.init_data_size(field, shape)

    def reset(self):
        """
        Empty the dataset.

        :return:
        """
        for field in self.fields['IN'] + self.fields['OUT']:
            self.data[field] = array([]).reshape((0, self.flat[field])) if self.flat[field] is not None else array([])
        self.current_sample = 0

    def memory_size(self, field=None):
        """
        Return the actual memory size of the dataset if field is None. Otherwise, return the actual memory size of the
        field.

        :param str field: Name of the data field
        :return: Size in bytes of the current dataset.
        """
        if field is None:
            return sum([self.data[field].nbytes for field in self.fields['IN'] + self.fields['OUT']])
        return self.data[field].nbytes

    def check_data(self, field, data):
        """
        Check if the data is a numpy array.

        :param str field: Values at 'input' or anything else. Define if the associated shape is correspond to input
        shape or output one.
        :param numpy.ndarray data: Corresponding tensor
        :return:
        """
        if type(data) != ndarray:
            raise TypeError(f"[{self.name}] Wrong data type in field '{field}': numpy array required, got {type(data)}")

    def add(self, field, data, partition_file=None):
        """
        Add new data to the dataset.

        :param str field: Values at 'input' or anything else. Define if the associated shape is correspond to input
        shape or output one.
        :param numpy.ndarray data: Corresponding tensor
        :param str partition_file: Path or string to the file in which to write the data
        :return:
        """
        # Check data type
        self.check_data(field, data)
        # Check if field is registered
        if field not in self.fields['IN'] + self.fields['OUT']:
            if not self.is_empty():
                raise ValueError(f"[{self.name}] A new field {field} tries to be created as Dataset is non empty. This "
                                 f"will lead to a different number of sample for each field of the dataset.")
            self.init_additional_field(field, data[0].shape)
        # Check data size initialization
        if self.flat[field] is None:
            self.init_data_size(field, data[0].shape)
        # Add each sample
        for sample in data:
            self.data[field] = concatenate((self.data[field], sample.flatten()[None, :]))
            if partition_file is not None:
                save(partition_file, sample.flatten())
        # Update sample indexing in dataset
        self.batch_per_field[field] += 1
        self.current_sample = max([len(self.data[f]) for f in self.fields['IN'] + self.fields['OUT']])

    def get(self, field, idx_begin, idx_end):
        """
        Get a batch of data 'field'.

        :param str field: Data field
        :param int idx_begin: Index of the first sample
        :param int idx_end: Index of the last sample
        :return:
        """
        indices = slice(idx_begin, idx_end) if self.shuffle_pattern is None else self.shuffle_pattern[idx_begin:idx_end]
        return self.data[field][indices].reshape((-1, *self.shape[field]))

    def shuffle(self):
        """
        Shuffle the current dataset.

        :return:
        """
        if self.is_empty():
            return
        # Generate a shuffle pattern
        self.shuffle_pattern = arange(self.current_sample)
        shuffle(self.shuffle_pattern)

    def __str__(self):
        """
        :return: String containing information about the BaseDatasetConfig object
        """
        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Max size: {self.max_size}\n"
        for side in ['IN', 'OUT']:
            description += f"    {'in' if side == 'IN' else 'Out'}put data fields: {self.fields[side]}"
            for field in self.fields[side]:
                description += f"      {field} shape: {self.shape[field]} // {field} flat shape: {self.flat[field]}"
        return description
