import numpy as np


class BaseDataset:

    def __init__(self, config):
        # Data storage
        self.data_in, self.data_out = np.array([]), np.array([])
        self.max_size = config.max_size
        self.currentSample = 0
        # Sizes
        self.inShape, self.inFlatShape = None, None
        self.outShape, self.outFlatShape = None, None
        # Description
        self.descriptionName = "BaseDataset"
        self.description = ""

    def init_data_size(self, side, data):
        if side == 'in':
            self.inShape = data.shape
            self.inFlatShape = len(data.flatten())
            self.data_in = np.array([]).reshape((0, self.inFlatShape))
        else:
            self.outShape = data.shape
            self.outFlatShape = len(data.flatten())
            self.data_out = np.array([]).reshape((0, self.outFlatShape))

    def reset(self):
        self.data_in = np.array([]).reshape((0, self.inFlatShape)) if self.inFlatShape is not None else np.array([])
        self.data_out = np.array([]).reshape((0, self.outFlatShape)) if self.outFlatShape is not None else np.array([])
        self.currentSample = 0

    def memory_size(self):
        return self.data_in.nbytes + self.data_out.nbytes

    def check_data(self, side, data):
        side = "inputs" if side == 'in' else "outputs"
        if type(data) != np.ndarray:
            raise TypeError("[BASEDATASET] Loaded {} must be numpy array, {} found.".format(side, type(data)))
        if len(data.shape) < 2:
            raise ValueError("[BASEDATASET] Loaded {} shape must be dim > 2, {} found.".format(side, type(data)))

    def add(self, side, data, partition_file):
        self.check_data(side, data)
        if side == 'in':
            if self.inFlatShape is None:
                self.init_data_size(side, data[0])
            for i in range(len(data)):
                self.data_in = np.concatenate((self.data_in, data[i].flatten()[None, :]))
                np.save(partition_file, data[i].flatten())
        else:
            if self.outFlatShape is None:
                self.init_data_size(side, data[0])
            for i in range(len(data)):
                self.data_out = np.concatenate((self.data_out, data[i].flatten()[None, :]))
                np.save(partition_file, data[i].flatten())

    def load(self, side, data):
        self.check_data(side, data)
        if side == 'in':
            if self.inFlatShape is None:
                self.init_data_size(side, data[0])
            self.data_in = np.concatenate((self.data_in, data), axis=0)
        else:
            if self.outFlatShape is None:
                self.init_data_size(side, data[0])
            self.data_out = np.concatenate((self.data_out, data), axis=0)

    def shuffle(self):
        index = np.arange(len(self.data_in))
        np.random.shuffle(index)
        inputs = np.copy(self.data_in)
        outputs = np.copy(self.data_out)
        for i in range(len(inputs)):
            inputs[i] = self.data_in[index[i]]
            outputs[i] = self.data_out[index[i]]
        self.data_in = inputs
        self.data_out = outputs

    def getDescription(self):
        if len(self.description) == 0:
            self.description += "\n{}\n".format(self.descriptionName)
            self.description += "   Max size: {}\n".format(self.max_size)
            self.description += "   Input shape, input flat shape: {}, {}\n".format(self.inShape, self.inFlatShape)
            self.description += "   Output shape, output flat shape: {}, {}\n".format(self.outShape, self.outFlatShape)
        return self.description
